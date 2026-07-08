use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::RMSNORM_F32_SOURCE;
use qwen3_hip_runtime::weights::TensorArchive;
use qwen3_hip_runtime::{Error, HipRuntime};

const TENSOR_NAME: &str = "talker.model.layers.0.input_layernorm.weight";

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
        });
    let model_path = model_dir.join("model.safetensors");
    let archive = TensorArchive::open(&model_path)?;

    let rows = 4usize;
    let gamma = archive.vector_f32(TENSOR_NAME)?;
    let cols = gamma.len();
    let epsilon = 1e-6f32;
    let input = deterministic_input(rows * cols);
    let expected = rmsnorm_reference(&input, &gamma, rows, cols, epsilon);

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
    let function = module.function("rmsnorm_f32")?;
    let input_dev = runtime.buffer_from_slice(&input)?;
    let gamma_dev = runtime.buffer_from_slice(&gamma)?;
    let output_dev = runtime.empty_buffer::<f32>(rows * cols)?;

    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut epsilon_arg = epsilon;
    let mut input_ptr = input_dev.as_ptr();
    let mut gamma_ptr = gamma_dev.as_ptr();
    let mut output_ptr = output_dev.as_mut_ptr();
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut gamma_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut epsilon_arg as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (rows as u32, 1, 1),
        (block, 1, 1),
        block * std::mem::size_of::<f32>() as u32,
        &mut params,
    )?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 5e-6 {
        return Err(Error::InvalidInput(format!(
            "RMSNorm weight smoke mismatch: max_abs={max_abs}"
        )));
    }

    println!(
        "RMSNorm weight smoke OK: tensor={TENSOR_NAME}, rows={rows}, cols={cols}, max_abs={max_abs}, first8={:?}",
        &actual[..8]
    );
    Ok(())
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 13.0)
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
