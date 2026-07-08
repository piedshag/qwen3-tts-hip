use qwen3_hip_runtime::HipRuntime;

fn main() -> qwen3_hip_runtime::Result<()> {
    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;

    let m = 2;
    let n = 4;
    let k = 3;
    let a = [1.0f32, -2.0, 3.0, 4.0, 0.5, -1.0];
    let b = [
        0.25f32, 2.0, -1.0, 0.5, -3.0, 1.5, 2.0, -0.75, 4.0, -2.0, 0.0, 1.0,
    ];

    let a_dev = runtime.buffer_from_slice(&a)?;
    let b_dev = runtime.buffer_from_slice(&b)?;
    let c_dev = runtime.empty_buffer::<f32>(m * n)?;
    blas.sgemm_row_major(&a_dev, &b_dev, &c_dev, m, n, k)?;
    runtime.synchronize()?;
    let c = c_dev.copy_to_host()?;

    let expected = row_major_matmul(&a, &b, m, n, k);
    let max_abs = c
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 1e-5 {
        return Err(qwen3_hip_runtime::Error::InvalidInput(format!(
            "SGEMM mismatch: max_abs={max_abs}, actual={c:?}, expected={expected:?}"
        )));
    }

    println!("SGEMM smoke OK: max_abs={max_abs}, output={c:?}");
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
