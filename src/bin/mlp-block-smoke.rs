use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::{ELEMENTWISE_F32_SOURCE, RMSNORM_F32_SOURCE};
use qwen3_hip_runtime::weights::TensorArchive;
use qwen3_hip_runtime::{Error, HipFunction, HipRuntime};

const LAYER_PREFIX: &str = "talker.model.layers.0";

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
        });
    let model_path = model_dir.join("model.safetensors");
    let archive = TensorArchive::open(&model_path)?;

    let gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.post_attention_layernorm.weight"))?;
    let (gate_weight, gate_in, gate_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.gate_proj.weight"))?;
    let (up_weight, up_in, up_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.up_proj.weight"))?;
    let (down_weight, down_in, down_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.down_proj.weight"))?;
    if gate_in != gamma.len() || up_in != gamma.len() || gate_out != up_out {
        return Err(Error::InvalidInput(format!(
            "gate/up shape mismatch: gamma={}, gate=({gate_in},{gate_out}), up=({up_in},{up_out})",
            gamma.len()
        )));
    }
    if down_in != gate_out || down_out != gamma.len() {
        return Err(Error::InvalidInput(format!(
            "down shape mismatch: down=({down_in},{down_out}), hidden={}, intermediate={gate_out}",
            gamma.len()
        )));
    }

    let rows = 2usize;
    let hidden = gamma.len();
    let intermediate = gate_out;
    let epsilon = 1e-6f32;
    let residual = deterministic_hidden(rows * hidden);
    let expected = mlp_block_reference(
        &residual,
        &gamma,
        &gate_weight,
        &up_weight,
        &down_weight,
        rows,
        hidden,
        intermediate,
        epsilon,
    );

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let rms_module = runtime.compile_module("rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
    let elementwise_module =
        runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let rmsnorm = rms_module.function("rmsnorm_f32")?;
    let swiglu = elementwise_module.function("swiglu_f32")?;
    let residual_add = elementwise_module.function("residual_add_f32")?;

    let residual_dev = runtime.buffer_from_slice(&residual)?;
    let gamma_dev = runtime.buffer_from_slice(&gamma)?;
    let gate_weight_dev = runtime.buffer_from_slice(&gate_weight)?;
    let up_weight_dev = runtime.buffer_from_slice(&up_weight)?;
    let down_weight_dev = runtime.buffer_from_slice(&down_weight)?;
    let normed_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let gate_dev = runtime.empty_buffer::<f32>(rows * intermediate)?;
    let up_dev = runtime.empty_buffer::<f32>(rows * intermediate)?;
    let swiglu_dev = runtime.empty_buffer::<f32>(rows * intermediate)?;
    let down_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let output_dev = runtime.empty_buffer::<f32>(rows * hidden)?;

    launch_rmsnorm(
        &rmsnorm,
        residual_dev.as_ptr(),
        gamma_dev.as_ptr(),
        normed_dev.as_mut_ptr(),
        rows,
        hidden,
        epsilon,
    )?;
    blas.sgemm_row_major(
        &normed_dev,
        &gate_weight_dev,
        &gate_dev,
        rows,
        intermediate,
        hidden,
    )?;
    blas.sgemm_row_major(
        &normed_dev,
        &up_weight_dev,
        &up_dev,
        rows,
        intermediate,
        hidden,
    )?;
    launch_ternary(
        &swiglu,
        gate_dev.as_ptr(),
        up_dev.as_ptr(),
        swiglu_dev.as_mut_ptr(),
        rows * intermediate,
    )?;
    blas.sgemm_row_major(
        &swiglu_dev,
        &down_weight_dev,
        &down_dev,
        rows,
        hidden,
        intermediate,
    )?;
    launch_ternary(
        &residual_add,
        residual_dev.as_ptr(),
        down_dev.as_ptr(),
        output_dev.as_mut_ptr(),
        rows * hidden,
    )?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = max_abs(&actual, &expected);
    let mean_abs = mean_abs(&actual, &expected);
    if max_abs > 1e-3 || mean_abs > 1e-4 {
        return Err(Error::InvalidInput(format!(
            "MLP block mismatch: max_abs={max_abs}, mean_abs={mean_abs}"
        )));
    }

    println!(
        "MLP block smoke OK: layer={LAYER_PREFIX}, rows={rows}, hidden={hidden}, intermediate={intermediate}, max_abs={max_abs}, mean_abs={mean_abs}, first8={:?}",
        &actual[..8]
    );
    Ok(())
}

fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut gamma = gamma;
    let mut output = output;
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut epsilon = epsilon;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut gamma as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut epsilon as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (rows as u32, 1, 1),
        (block, 1, 1),
        block * std::mem::size_of::<f32>() as u32,
        &mut params,
    )
}

fn launch_ternary(
    function: &HipFunction,
    input_a: *const c_void,
    input_b: *const c_void,
    output: *mut c_void,
    total: usize,
) -> qwen3_hip_runtime::Result<()> {
    let mut input_a = input_a;
    let mut input_b = input_b;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input_a as *mut *const c_void as *mut c_void,
        &mut input_b as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn deterministic_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 17.0)
        .collect()
}

fn mlp_block_reference(
    residual: &[f32],
    gamma: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    hidden: usize,
    intermediate: usize,
    epsilon: f32,
) -> Vec<f32> {
    let normed = rmsnorm_reference(residual, gamma, rows, hidden, epsilon);
    let gate = row_major_matmul(&normed, gate_weight, rows, intermediate, hidden);
    let up = row_major_matmul(&normed, up_weight, rows, intermediate, hidden);
    let swiglu = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    let down = row_major_matmul(&swiglu, down_weight, rows, hidden, intermediate);
    residual
        .iter()
        .zip(down.iter())
        .map(|(residual, down)| residual + down)
        .collect()
}

fn rmsnorm_reference(
    input: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let sum = input[offset..offset + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inv_rms = (sum / cols as f32 + epsilon).sqrt().recip();
        for col in 0..cols {
            output[offset + col] = input[offset + col] * inv_rms * gamma[col];
        }
    }
    output
}

fn row_major_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32
}
