use std::ffi::c_void;
use std::time::Instant;

use qwen3_hip_runtime::kernel::HipFunction;
use qwen3_hip_runtime::{Error, HipRuntime, Result};

const GEMV_SOURCE: &str = r#"
extern "C" __global__ void row_vector_matmul_f32(
    const float* a,
    const float* b,
    float* c,
    int n,
    int k
) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];

    float sum = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        sum += a[idx] * b[idx * n + col];
    }
    scratch[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        c[col] = scratch[0];
    }
}
"#;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let n = parse_arg(args.next(), "n")?.unwrap_or(2048);
    let k = parse_arg(args.next(), "k")?.unwrap_or(1024);
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(1000);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(50);
    if n == 0 || k == 0 || iterations == 0 {
        return Err(Error::InvalidInput(
            "n, k, and iterations must be non-zero".to_string(),
        ));
    }

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let module = runtime.compile_module("gemv_bench.cpp", GEMV_SOURCE)?;
    let gemv = module.function("row_vector_matmul_f32")?;

    let a_host = deterministic_data(k, 17);
    let b_host = deterministic_data(k * n, 31);
    let a = runtime.buffer_from_slice(&a_host)?;
    let b = runtime.buffer_from_slice(&b_host)?;
    let blas_out = runtime.empty_buffer::<f32>(n)?;
    let gemv_out = runtime.empty_buffer::<f32>(n)?;
    runtime.synchronize()?;

    blas.sgemm_row_major(&a, &b, &blas_out, 1, n, k)?;
    launch_gemv(&gemv, &a, &b, &gemv_out, n, k)?;
    runtime.synchronize()?;
    let blas_values = blas_out.copy_to_host()?;
    let gemv_values = gemv_out.copy_to_host()?;
    let max_abs = max_abs_diff(&blas_values, &gemv_values);
    if max_abs > 1e-3 {
        return Err(Error::InvalidInput(format!(
            "GEMV mismatch: max_abs={max_abs}"
        )));
    }

    for _ in 0..warmup {
        blas.sgemm_row_major(&a, &b, &blas_out, 1, n, k)?;
    }
    runtime.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        blas.sgemm_row_major(&a, &b, &blas_out, 1, n, k)?;
    }
    runtime.synchronize()?;
    let blas_seconds = start.elapsed().as_secs_f64();

    for _ in 0..warmup {
        launch_gemv(&gemv, &a, &b, &gemv_out, n, k)?;
    }
    runtime.synchronize()?;
    let start = Instant::now();
    for _ in 0..iterations {
        launch_gemv(&gemv, &a, &b, &gemv_out, n, k)?;
    }
    runtime.synchronize()?;
    let gemv_seconds = start.elapsed().as_secs_f64();

    let blas_mean_us = blas_seconds * 1_000_000.0 / iterations as f64;
    let gemv_mean_us = gemv_seconds * 1_000_000.0 / iterations as f64;
    println!(
        "GEMV bench: n={n}, k={k}, iterations={iterations}, warmup={warmup}, max_abs={max_abs:.9}, blas_mean_us={blas_mean_us:.3}, gemv_mean_us={gemv_mean_us:.3}, speedup={:.3}",
        blas_mean_us / gemv_mean_us
    );
    Ok(())
}

fn launch_gemv(
    function: &HipFunction,
    a: &qwen3_hip_runtime::DeviceBuffer<f32>,
    b: &qwen3_hip_runtime::DeviceBuffer<f32>,
    c: &qwen3_hip_runtime::DeviceBuffer<f32>,
    n: usize,
    k: usize,
) -> Result<()> {
    let mut a_ptr = a.as_ptr();
    let mut b_ptr = b.as_ptr();
    let mut c_ptr = c.as_mut_ptr();
    let mut n_i32 = n as i32;
    let mut k_i32 = k as i32;
    let mut params = [
        &mut a_ptr as *mut *const c_void as *mut c_void,
        &mut b_ptr as *mut *const c_void as *mut c_void,
        &mut c_ptr as *mut *mut c_void as *mut c_void,
        &mut n_i32 as *mut i32 as *mut c_void,
        &mut k_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch((n as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn deterministic_data(len: usize, period: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % period) as f32 - period as f32 * 0.5) / period as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn parse_arg(value: Option<String>, name: &str) -> Result<Option<usize>> {
    value
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}
