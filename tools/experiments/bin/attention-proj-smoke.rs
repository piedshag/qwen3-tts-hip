use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::{LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE, ROPE_BHSD_F32_SOURCE};
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
    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;

    let q_gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.q_norm.weight"))?;
    let k_gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.k_norm.weight"))?;
    let (q_weight, q_in, q_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.q_proj.weight"))?;
    let (k_weight, k_in, k_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.k_proj.weight"))?;
    let (v_weight, v_in, v_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.v_proj.weight"))?;

    let head_dim = q_gamma.len();
    if k_gamma.len() != head_dim || q_in != k_in || q_in != v_in {
        return Err(Error::InvalidInput(format!(
            "attention input shape mismatch: q_in={q_in}, k_in={k_in}, v_in={v_in}, q_norm={}, k_norm={}",
            q_gamma.len(),
            k_gamma.len()
        )));
    }
    if q_out % head_dim != 0 || k_out % head_dim != 0 || v_out != k_out {
        return Err(Error::InvalidInput(format!(
            "attention output shape mismatch: q_out={q_out}, k_out={k_out}, v_out={v_out}, head_dim={head_dim}"
        )));
    }
    let batch = 1usize;
    let steps = 2usize;
    let hidden = q_in;
    let q_heads = q_out / head_dim;
    let kv_heads = k_out / head_dim;
    let rows = batch * steps;
    let offset = 3usize;
    let theta = 10_000.0f32;
    let epsilon = 1e-6f32;
    let hidden_input = deterministic_hidden(rows * hidden);

    let expected = attention_projection_reference(
        &hidden_input,
        &q_weight,
        &k_weight,
        &v_weight,
        &q_gamma,
        &k_gamma,
        batch,
        steps,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        offset,
        theta,
        epsilon,
    );

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let rms_module = runtime.compile_module("rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
    let rope_module = runtime.compile_module("rope_bhsd_f32.cpp", ROPE_BHSD_F32_SOURCE)?;
    let layout_module = runtime.compile_module("layout_f32.cpp", LAYOUT_F32_SOURCE)?;
    let rmsnorm = rms_module.function("rmsnorm_f32")?;
    let rope = rope_module.function("rope_bhsd_f32")?;
    let permute = layout_module.function("permute_bshd_to_bhsd_f32")?;

    let hidden_dev = runtime.buffer_from_slice(&hidden_input)?;
    let q_weight_dev = runtime.buffer_from_slice(&q_weight)?;
    let k_weight_dev = runtime.buffer_from_slice(&k_weight)?;
    let v_weight_dev = runtime.buffer_from_slice(&v_weight)?;
    let q_gamma_dev = runtime.buffer_from_slice(&q_gamma)?;
    let k_gamma_dev = runtime.buffer_from_slice(&k_gamma)?;
    let q_proj_dev = runtime.empty_buffer::<f32>(rows * q_out)?;
    let k_proj_dev = runtime.empty_buffer::<f32>(rows * k_out)?;
    let v_proj_dev = runtime.empty_buffer::<f32>(rows * v_out)?;
    let q_norm_dev = runtime.empty_buffer::<f32>(rows * q_out)?;
    let k_norm_dev = runtime.empty_buffer::<f32>(rows * k_out)?;
    let q_bhsd_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * head_dim)?;
    let k_bhsd_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;
    let v_bhsd_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;
    let q_rope_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * head_dim)?;
    let k_rope_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;

    blas.sgemm_row_major(&hidden_dev, &q_weight_dev, &q_proj_dev, rows, q_out, hidden)?;
    blas.sgemm_row_major(&hidden_dev, &k_weight_dev, &k_proj_dev, rows, k_out, hidden)?;
    blas.sgemm_row_major(&hidden_dev, &v_weight_dev, &v_proj_dev, rows, v_out, hidden)?;
    launch_rmsnorm(
        &rmsnorm,
        q_proj_dev.as_ptr(),
        q_gamma_dev.as_ptr(),
        q_norm_dev.as_mut_ptr(),
        rows * q_heads,
        head_dim,
        epsilon,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        k_proj_dev.as_ptr(),
        k_gamma_dev.as_ptr(),
        k_norm_dev.as_mut_ptr(),
        rows * kv_heads,
        head_dim,
        epsilon,
    )?;
    launch_permute(
        &permute,
        q_norm_dev.as_ptr(),
        q_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        q_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        k_norm_dev.as_ptr(),
        k_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        v_proj_dev.as_ptr(),
        v_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_rope(
        &rope,
        q_bhsd_dev.as_ptr(),
        q_rope_dev.as_mut_ptr(),
        batch * q_heads * steps * head_dim,
        q_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    launch_rope(
        &rope,
        k_bhsd_dev.as_ptr(),
        k_rope_dev.as_mut_ptr(),
        batch * kv_heads * steps * head_dim,
        kv_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    runtime.synchronize()?;

    let q_actual = q_rope_dev.copy_to_host()?;
    let k_actual = k_rope_dev.copy_to_host()?;
    let v_actual = v_bhsd_dev.copy_to_host()?;
    let q_max_abs = max_abs(&q_actual, &expected.q);
    let k_max_abs = max_abs(&k_actual, &expected.k);
    let v_max_abs = max_abs(&v_actual, &expected.v);
    if q_max_abs > 1e-4 || k_max_abs > 1e-4 || v_max_abs > 1e-4 {
        return Err(Error::InvalidInput(format!(
            "attention projection mismatch: q_max_abs={q_max_abs}, k_max_abs={k_max_abs}, v_max_abs={v_max_abs}"
        )));
    }

    println!(
        "Attention projection smoke OK: layer={LAYER_PREFIX}, batch={batch}, steps={steps}, hidden={hidden}, q_heads={q_heads}, kv_heads={kv_heads}, head_dim={head_dim}, q_max_abs={q_max_abs}, k_max_abs={k_max_abs}, v_max_abs={v_max_abs}, q_first8={:?}",
        &q_actual[..8]
    );
    Ok(())
}

struct AttentionProjectionReference {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
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

fn launch_permute(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = batch * steps * heads * head_dim;
    let mut input = input;
    let mut output = output;
    let mut batch_i32 = batch as i32;
    let mut steps_i32 = steps as i32;
    let mut heads_i32 = heads as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

#[allow(clippy::too_many_arguments)]
fn launch_rope(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    total: usize,
    heads: usize,
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut heads_i32 = heads as i32;
    let mut steps_i32 = steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut theta = theta;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut theta as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

#[allow(clippy::too_many_arguments)]
fn attention_projection_reference(
    hidden: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    q_gamma: &[f32],
    k_gamma: &[f32],
    batch: usize,
    steps: usize,
    hidden_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
    epsilon: f32,
) -> AttentionProjectionReference {
    let rows = batch * steps;
    let q_out = q_heads * head_dim;
    let kv_out = kv_heads * head_dim;
    let q = row_major_matmul(hidden, q_weight, rows, q_out, hidden_size);
    let k = row_major_matmul(hidden, k_weight, rows, kv_out, hidden_size);
    let v = row_major_matmul(hidden, v_weight, rows, kv_out, hidden_size);
    let q = rmsnorm_reference(&q, q_gamma, rows * q_heads, head_dim, epsilon);
    let k = rmsnorm_reference(&k, k_gamma, rows * kv_heads, head_dim, epsilon);
    let q = permute_bshd_to_bhsd(&q, batch, steps, q_heads, head_dim);
    let k = permute_bshd_to_bhsd(&k, batch, steps, kv_heads, head_dim);
    let v = permute_bshd_to_bhsd(&v, batch, steps, kv_heads, head_dim);
    AttentionProjectionReference {
        q: rope_reference(&q, q_heads, steps, head_dim, offset, theta),
        k: rope_reference(&k, kv_heads, steps, head_dim, offset, theta),
        v,
    }
}

fn deterministic_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 17.0)
        .collect()
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

fn permute_bshd_to_bhsd(
    input: &[f32],
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for step in 0..steps {
            for head in 0..heads {
                for dim in 0..head_dim {
                    let input_idx =
                        (((batch_index * steps + step) * heads + head) * head_dim) + dim;
                    let output_idx =
                        (((batch_index * heads + head) * steps + step) * head_dim) + dim;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
    output
}

fn rope_reference(
    input: &[f32],
    heads: usize,
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let half = head_dim / 2;
    for idx in 0..input.len() {
        let dim_index = idx % head_dim;
        let step_index = (idx / head_dim) % steps;
        let pair_index = dim_index % half;
        let base = idx - dim_index;
        let first = input[base + pair_index];
        let second = input[base + pair_index + half];
        let exponent = (pair_index * 2) as f32 / head_dim as f32;
        let angle = (offset + step_index) as f32 / theta.powf(exponent);
        let cos = angle.cos();
        let sin = angle.sin();
        output[idx] = if dim_index < half {
            first * cos - second * sin
        } else {
            second * cos + first * sin
        };
    }
    let _ = heads;
    output
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}
