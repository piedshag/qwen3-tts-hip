use qwen3_hip_runtime::HipRuntime;

fn main() -> qwen3_hip_runtime::Result<()> {
    let runtime = HipRuntime::new(0)?;
    let device_count = runtime.device_count()?;
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let buffer = runtime.buffer_from_slice(&input)?;
    runtime.synchronize()?;
    let output = buffer.copy_to_host()?;
    let _blas = runtime.create_blas_handle()?;
    println!(
        "HIP smoke OK: devices={}, device={}, roundtrip={:?}",
        device_count,
        runtime.device_index(),
        output
    );
    Ok(())
}
