use std::ffi::c_void;

use qwen3_hip_runtime::kernels::ELEMENTWISE_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let total = 4096usize;
    let a = deterministic_values(total, 19, 9.0);
    let b = deterministic_values(total, 23, 7.0);
    let expected = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let residual_add = module.function("residual_add_f32")?;
    let stream = runtime.create_stream()?;
    let a_dev = runtime.buffer_from_slice(&a)?;
    let b_dev = runtime.buffer_from_slice(&b)?;
    let output_dev = runtime.empty_buffer::<f32>(total)?;

    let mut input_a = a_dev.as_ptr();
    let mut input_b = b_dev.as_ptr();
    let mut output = output_dev.as_mut_ptr();
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input_a as *mut *const c_void as *mut c_void,
        &mut input_b as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);

    stream.begin_capture()?;
    residual_add.launch_on_stream((grid, 1, 1), (block, 1, 1), 0, &mut params, Some(&stream))?;
    let graph = stream.end_capture()?;
    let exec = graph.instantiate()?;

    for _ in 0..10 {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_abs != 0.0 {
        return Err(Error::InvalidInput(format!(
            "graph smoke mismatch: max_abs={max_abs}"
        )));
    }

    println!(
        "HIP graph smoke OK: total={total}, launches=10, max_abs={max_abs}, first8={:?}",
        &actual[..8]
    );
    Ok(())
}

fn deterministic_values(len: usize, modulus: usize, scale: f32) -> Vec<f32> {
    let center = (modulus / 2) as f32;
    (0..len)
        .map(|idx| ((idx % modulus) as f32 - center) / scale)
        .collect()
}
