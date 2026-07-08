use std::ffi::c_void;

use qwen3_hip_runtime::kernels::SOFTMAX_F32_SOURCE;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let rows = 7usize;
    let cols = 257usize;
    let active_cols = 173usize;
    let input = deterministic_logits(rows * cols);
    let expected = masked_softmax_reference(&input, rows, cols, active_cols);

    let runtime = HipRuntime::new(0)?;
    let module = runtime.compile_module("softmax_f32.cpp", SOFTMAX_F32_SOURCE)?;
    let function = module.function("masked_softmax_f32")?;
    let input_dev = runtime.buffer_from_slice(&input)?;
    let output_dev = runtime.empty_buffer::<f32>(rows * cols)?;

    let mut input_ptr = input_dev.as_ptr();
    let mut output_ptr = output_dev.as_mut_ptr();
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut active_cols_i32 = active_cols as i32;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut active_cols_i32 as *mut i32 as *mut c_void,
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
    let max_abs = max_abs(&actual, &expected);
    let row_sums = row_sums(&actual, rows, cols);
    let max_sum_err = row_sums
        .iter()
        .map(|sum| (sum - 1.0).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 2e-6 || max_sum_err > 2e-6 {
        return Err(Error::InvalidInput(format!(
            "softmax smoke mismatch: max_abs={max_abs}, max_sum_err={max_sum_err}"
        )));
    }

    println!(
        "Softmax smoke OK: rows={rows}, cols={cols}, active_cols={active_cols}, max_abs={max_abs}, max_sum_err={max_sum_err}, first8={:?}",
        &actual[..8]
    );
    Ok(())
}

fn deterministic_logits(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 37) as f32 - 18.0) / 5.0)
        .collect()
}

fn masked_softmax_reference(
    input: &[f32],
    rows: usize,
    cols: usize,
    active_cols: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let row_input = &input[offset..offset + active_cols];
        let row_max = row_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = row_input
            .iter()
            .map(|value| (value - row_max).exp())
            .sum::<f32>();
        for col in 0..active_cols {
            output[offset + col] = (input[offset + col] - row_max).exp() / sum;
        }
    }
    output
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn row_sums(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| input[row * cols..(row + 1) * cols].iter().sum::<f32>())
        .collect()
}
