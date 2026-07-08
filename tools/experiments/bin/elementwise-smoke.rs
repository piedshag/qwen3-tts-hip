use std::ffi::c_void;

use qwen3_hip_runtime::kernels::ELEMENTWISE_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let total = 4096usize;
    let residual = deterministic_values(total, 19, 9.0);
    let update = deterministic_values(total, 23, 7.0);
    let gate = deterministic_values(total, 29, 11.0);
    let up = deterministic_values(total, 31, 13.0);
    let expected_residual = residual_add_reference(&residual, &update);
    let expected_swiglu = swiglu_reference(&gate, &up);

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let residual_function = module.function("residual_add_f32")?;
    let swiglu_function = module.function("swiglu_f32")?;

    let residual_dev = runtime.buffer_from_slice(&residual)?;
    let update_dev = runtime.buffer_from_slice(&update)?;
    let gate_dev = runtime.buffer_from_slice(&gate)?;
    let up_dev = runtime.buffer_from_slice(&up)?;
    let residual_output_dev = runtime.empty_buffer::<f32>(total)?;
    let swiglu_output_dev = runtime.empty_buffer::<f32>(total)?;

    launch_ternary(
        &residual_function,
        residual_dev.as_ptr(),
        update_dev.as_ptr(),
        residual_output_dev.as_mut_ptr(),
        total,
    )?;
    launch_ternary(
        &swiglu_function,
        gate_dev.as_ptr(),
        up_dev.as_ptr(),
        swiglu_output_dev.as_mut_ptr(),
        total,
    )?;
    runtime.synchronize()?;

    let residual_actual = residual_output_dev.copy_to_host()?;
    let swiglu_actual = swiglu_output_dev.copy_to_host()?;
    let residual_max_abs = max_abs(&residual_actual, &expected_residual);
    let swiglu_max_abs = max_abs(&swiglu_actual, &expected_swiglu);
    if residual_max_abs != 0.0 || swiglu_max_abs > 1e-6 {
        return Err(Error::InvalidInput(format!(
            "elementwise smoke mismatch: residual_max_abs={residual_max_abs}, swiglu_max_abs={swiglu_max_abs}"
        )));
    }

    println!(
        "Elementwise smoke OK: total={total}, residual_max_abs={residual_max_abs}, swiglu_max_abs={swiglu_max_abs}, swiglu_first8={:?}",
        &swiglu_actual[..8]
    );
    Ok(())
}

fn launch_ternary(
    function: &qwen3_hip_runtime::HipFunction,
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

fn deterministic_values(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    let center = (modulus / 2) as f32;
    (0..len)
        .map(|idx| ((idx % modulus) as f32 - center) / scale)
        .collect()
}

fn residual_add_reference(residual: &[f32], update: &[f32]) -> Vec<f32> {
    residual
        .iter()
        .zip(update.iter())
        .map(|(residual, update)| residual + update)
        .collect()
}

fn swiglu_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect()
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}
