use std::ffi::c_void;

use qwen3_hip_runtime::kernels::ROPE_BHSD_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let batch = 2usize;
    let heads = 5usize;
    let steps = 3usize;
    let head_dim = 128usize;
    let offset = 7usize;
    let theta = 10_000.0f32;
    let total = batch * heads * steps * head_dim;
    let input = deterministic_input(total);
    let expected = rope_reference(&input, heads, steps, head_dim, offset, theta);

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("rope_bhsd_f32.cpp", ROPE_BHSD_F32_SOURCE)?;
    let function = module.function("rope_bhsd_f32")?;
    let input_dev = runtime.buffer_from_slice(&input)?;
    let output_dev = runtime.empty_buffer::<f32>(total)?;

    let mut input_ptr = input_dev.as_ptr();
    let mut output_ptr = output_dev.as_mut_ptr();
    let mut total_i32 = total as i32;
    let mut heads_i32 = heads as i32;
    let mut steps_i32 = steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut theta_arg = theta;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut theta_arg as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 2e-6 {
        return Err(Error::InvalidInput(format!(
            "RoPE smoke mismatch: max_abs={max_abs}"
        )));
    }

    println!(
        "RoPE smoke OK: batch={batch}, heads={heads}, steps={steps}, head_dim={head_dim}, max_abs={max_abs}, first8={:?}",
        &actual[..8]
    );
    Ok(())
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 29) as f32 - 14.0) / 11.0)
        .collect()
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
