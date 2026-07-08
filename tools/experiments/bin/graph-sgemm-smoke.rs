use std::ffi::c_void;

use qwen3_hip_runtime::kernels::ELEMENTWISE_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let m = 2usize;
    let n = 4usize;
    let k = 3usize;
    let a = vec![1.0, -2.0, 0.5, 3.0, 1.5, -1.0];
    let b = vec![
        2.0, 0.5, -1.0, 4.0, -3.0, 2.0, 1.0, -0.5, 0.25, -1.5, 2.5, 3.0,
    ];
    let residual = vec![0.25; m * n];
    let expected_sgemm = row_major_matmul(&a, &b, m, n, k);
    let expected = expected_sgemm
        .iter()
        .zip(residual.iter())
        .map(|(value, residual)| value + residual)
        .collect::<Vec<_>>();

    let runtime = HipRuntime::new(0)?;
    let stream = runtime.create_stream()?;
    let blas = runtime.create_blas_handle()?;
    let module = runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let residual_add = module.function("residual_add_f32")?;
    let a_dev = runtime.buffer_from_slice(&a)?;
    let b_dev = runtime.buffer_from_slice(&b)?;
    let residual_dev = runtime.buffer_from_slice(&residual)?;
    let tmp_dev = runtime.empty_buffer::<f32>(m * n)?;
    let output_dev = runtime.empty_buffer::<f32>(m * n)?;

    blas.set_stream(Some(&stream))?;
    let mut tmp_ptr = tmp_dev.as_ptr();
    let mut residual_ptr = residual_dev.as_ptr();
    let mut output_ptr = output_dev.as_mut_ptr();
    let mut total_i32 = (m * n) as i32;
    let mut params = [
        &mut tmp_ptr as *mut *const c_void as *mut c_void,
        &mut residual_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = ((m * n) as u32).div_ceil(block);

    stream.begin_capture()?;
    blas.sgemm_row_major(&a_dev, &b_dev, &tmp_dev, m, n, k)?;
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
    if max_abs > 1e-6 {
        return Err(Error::InvalidInput(format!(
            "graph SGEMM smoke mismatch: max_abs={max_abs}, actual={actual:?}, expected={expected:?}"
        )));
    }

    println!(
        "HIP graph SGEMM smoke OK: m={m}, n={n}, k={k}, launches=10, max_abs={max_abs}, output={actual:?}"
    );
    Ok(())
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
