use std::ffi::c_void;

use qwen3_hip_runtime::kernels::ARGMAX_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let rows = 9usize;
    let cols = 1025usize;
    let input = deterministic_logits(rows, cols);
    let expected = argmax_reference(&input, rows, cols);

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("argmax_f32.cpp", ARGMAX_F32_SOURCE)?;
    let function = module.function("argmax_rows_f32")?;
    let input_dev = runtime.buffer_from_slice(&input)?;
    let output_dev = runtime.empty_buffer::<i32>(rows)?;

    let mut input_ptr = input_dev.as_ptr();
    let mut output_ptr = output_dev.as_mut_ptr();
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let shared = block * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()) as u32;
    function.launch((rows as u32, 1, 1), (block, 1, 1), shared, &mut params)?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    if actual != expected {
        return Err(Error::InvalidInput(format!(
            "argmax smoke mismatch: actual={actual:?}, expected={expected:?}"
        )));
    }

    println!("Argmax smoke OK: rows={rows}, cols={cols}, output={actual:?}");
    Ok(())
}

fn deterministic_logits(rows: usize, cols: usize) -> Vec<f32> {
    let mut data = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            data[row * cols + col] = ((row * 17 + col * 31) % 997) as f32 / 97.0;
        }
        let winner = (row * 113 + 41) % cols;
        data[row * cols + winner] = 50.0 - row as f32;
    }
    data[2 * cols + 10] = 60.0;
    data[2 * cols + 20] = 60.0;
    data
}

fn argmax_reference(input: &[f32], rows: usize, cols: usize) -> Vec<i32> {
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let offset = row * cols;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = 0usize;
        for col in 0..cols {
            let value = input[offset + col];
            if value > best_value || (value == best_value && col < best_index) {
                best_value = value;
                best_index = col;
            }
        }
        output.push(best_index as i32);
    }
    output
}
