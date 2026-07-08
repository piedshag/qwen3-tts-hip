use std::path::PathBuf;

use qwen3_hip_runtime::weights::TensorArchive;
use qwen3_hip_runtime::{Error, HipRuntime};

const TENSOR_NAME: &str = "talker.model.layers.0.self_attn.q_proj.weight";

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
        });
    let model_path = model_dir.join("model.safetensors");
    let archive = TensorArchive::open(&model_path)?;
    let (weight_transposed_full, in_dim, out_dim) =
        archive.linear_weight_transposed_f32(TENSOR_NAME)?;
    let selected_out = 8;
    if out_dim < selected_out {
        return Err(Error::InvalidInput(format!(
            "{TENSOR_NAME} output dim {out_dim}, expected at least {selected_out}"
        )));
    }

    let input = deterministic_input(in_dim);
    let weight_transposed =
        select_output_columns(&weight_transposed_full, in_dim, out_dim, selected_out);
    let expected = row_major_matmul(&input, &weight_transposed, 1, selected_out, in_dim);

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let input_dev = runtime.buffer_from_slice(&input)?;
    let weight_dev = runtime.buffer_from_slice(&weight_transposed)?;
    let output_dev = runtime.empty_buffer::<f32>(selected_out)?;
    blas.sgemm_row_major(
        &input_dev,
        &weight_dev,
        &output_dev,
        1,
        selected_out,
        in_dim,
    )?;
    runtime.synchronize()?;
    let actual = output_dev.copy_to_host()?;

    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 5e-3 {
        return Err(Error::InvalidInput(format!(
            "linear smoke mismatch: max_abs={max_abs}, actual={actual:?}, expected={expected:?}"
        )));
    }

    println!(
        "linear weight smoke OK: tensor={TENSOR_NAME}, in_dim={in_dim}, selected_out={selected_out}, max_abs={max_abs}, output={actual:?}"
    );
    Ok(())
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 17) as f32 - 8.0) / 9.0)
        .collect()
}

fn select_output_columns(
    weight_transposed: &[f32],
    in_dim: usize,
    out_dim: usize,
    selected_out: usize,
) -> Vec<f32> {
    let mut transposed = vec![0.0; in_dim * selected_out];
    for in_idx in 0..in_dim {
        for out_idx in 0..selected_out {
            transposed[in_idx * selected_out + out_idx] =
                weight_transposed[in_idx * out_dim + out_idx];
        }
    }
    transposed
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
