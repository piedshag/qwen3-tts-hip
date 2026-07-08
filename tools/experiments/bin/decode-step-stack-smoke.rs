use std::path::PathBuf;

use qwen3_hip_runtime::decode::DecodeStepStack;
use qwen3_hip_runtime::weights::{TensorArchive, tensor_to_f32};
use qwen3_hip_runtime::{Error, HipRuntime};
use safetensors::SafeTensors;

#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_out: usize,
    kv_out: usize,
    intermediate: usize,
    epsilon: f32,
    theta: f32,
    scale: f32,
}

struct LayerWeights {
    input_gamma: Vec<f32>,
    q_gamma: Vec<f32>,
    k_gamma: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    v_weight: Vec<f32>,
    o_weight: Vec<f32>,
    post_attention_gamma: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let layer_count = parse_arg(args.next(), "layer count")?.unwrap_or(2);
    let prefix_steps = parse_arg(args.next(), "prefix steps")?.unwrap_or(2);
    if layer_count == 0 || prefix_steps == 0 {
        return Err(Error::InvalidInput(
            "layer count and prefix steps must be non-zero".to_string(),
        ));
    }
    let total_steps = prefix_steps + 1;

    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
    let (layers, dims) = archive.with_tensors(|tensors| {
        let first = load_layer(tensors, 0)?;
        let dims = infer_dims(&first)?;
        let mut layers = Vec::with_capacity(layer_count);
        layers.push(first);
        for index in 1..layer_count {
            let layer = load_layer(tensors, index)?;
            validate_dims(index, infer_dims(&layer)?, dims)?;
            layers.push(layer);
        }
        Ok((layers, dims))
    })?;

    let hidden = deterministic_hidden(total_steps * dims.hidden);
    let mut expected_full = hidden.clone();
    for layer in &layers {
        expected_full = layer_reference(&expected_full, layer, dims, total_steps);
    }
    let expected = &expected_full[prefix_steps * dims.hidden..];

    let runtime = HipRuntime::new(0)?;
    let stack = DecodeStepStack::load(&runtime, &model_dir, layer_count, total_steps)?;
    let prefix_dev = runtime.buffer_from_slice(&hidden[..prefix_steps * dims.hidden])?;
    let current_dev = runtime.buffer_from_slice(&hidden[prefix_steps * dims.hidden..])?;
    let output_dev = runtime.empty_buffer::<f32>(dims.hidden)?;
    stack.prefill(&prefix_dev, prefix_steps)?;
    stack.decode_step(&current_dev, &output_dev, prefix_steps)?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = max_abs(&actual, expected);
    let mean_abs = mean_abs(&actual, expected);
    if max_abs > 2e-3 || mean_abs > 2e-4 {
        return Err(Error::InvalidInput(format!(
            "decode stack mismatch: layers={layer_count}, max_abs={max_abs}, mean_abs={mean_abs}"
        )));
    }

    println!(
        "Decode-step stack smoke OK: layers={layer_count}, prefix_steps={prefix_steps}, hidden={}, q_heads={}, kv_heads={}, head_dim={}, max_abs={max_abs}, mean_abs={mean_abs}, first8={:?}",
        dims.hidden,
        dims.q_heads,
        dims.kv_heads,
        dims.head_dim,
        &actual[..8]
    );
    Ok(())
}

fn parse_arg(
    value: Option<std::ffi::OsString>,
    name: &str,
) -> qwen3_hip_runtime::Result<Option<usize>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}

fn load_layer(tensors: &SafeTensors<'_>, index: usize) -> qwen3_hip_runtime::Result<LayerWeights> {
    let prefix = format!("talker.model.layers.{index}");
    Ok(LayerWeights {
        input_gamma: vector_f32(tensors, &format!("{prefix}.input_layernorm.weight"))?,
        q_gamma: vector_f32(tensors, &format!("{prefix}.self_attn.q_norm.weight"))?,
        k_gamma: vector_f32(tensors, &format!("{prefix}.self_attn.k_norm.weight"))?,
        q_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.q_proj.weight"),
        )?,
        k_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.k_proj.weight"),
        )?,
        v_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.v_proj.weight"),
        )?,
        o_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.o_proj.weight"),
        )?,
        post_attention_gamma: vector_f32(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        gate_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
        )?,
        up_weight: linear_weight_transposed_f32(tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
        down_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
        )?,
    })
}

fn infer_dims(layer: &LayerWeights) -> qwen3_hip_runtime::Result<Dims> {
    let hidden = layer.input_gamma.len();
    let head_dim = layer.q_gamma.len();
    let q_out = layer.q_weight.len() / hidden;
    let kv_out = layer.k_weight.len() / hidden;
    let intermediate = layer.gate_weight.len() / hidden;
    if head_dim == 0
        || q_out % head_dim != 0
        || kv_out % head_dim != 0
        || layer.k_gamma.len() != head_dim
        || layer.v_weight.len() != hidden * kv_out
        || layer.o_weight.len() != q_out * hidden
        || layer.post_attention_gamma.len() != hidden
        || layer.up_weight.len() != hidden * intermediate
        || layer.down_weight.len() != intermediate * hidden
    {
        return Err(Error::InvalidInput("invalid layer dimensions".to_string()));
    }
    Ok(Dims {
        hidden,
        q_heads: q_out / head_dim,
        kv_heads: kv_out / head_dim,
        head_dim,
        q_out,
        kv_out,
        intermediate,
        epsilon: 1e-6,
        theta: 10_000.0,
        scale: (head_dim as f32).sqrt().recip(),
    })
}

fn validate_dims(index: usize, actual: Dims, expected: Dims) -> qwen3_hip_runtime::Result<()> {
    if actual.hidden != expected.hidden
        || actual.q_heads != expected.q_heads
        || actual.kv_heads != expected.kv_heads
        || actual.head_dim != expected.head_dim
        || actual.intermediate != expected.intermediate
    {
        return Err(Error::InvalidInput(format!(
            "layer {index} dims mismatch: hidden={}, q_heads={}, kv_heads={}, head_dim={}, intermediate={}",
            actual.hidden, actual.q_heads, actual.kv_heads, actual.head_dim, actual.intermediate
        )));
    }
    Ok(())
}

fn vector_f32(tensors: &SafeTensors<'_>, name: &str) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))?;
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(Error::InvalidInput(format!(
            "{name} rank {}, expected 1",
            shape.len()
        )));
    }
    tensor_to_f32(name, tensor.dtype(), tensor.data(), shape[0])
}

fn linear_weight_transposed_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "{name} rank {}, expected 2",
            shape.len()
        )));
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let data = tensor_to_f32(name, tensor.dtype(), tensor.data(), out_dim * in_dim)?;
    let mut transposed = vec![0.0; in_dim * out_dim];
    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            transposed[in_idx * out_dim + out_idx] = data[out_idx * in_dim + in_idx];
        }
    }
    Ok(transposed)
}

fn layer_reference(input: &[f32], layer: &LayerWeights, dims: Dims, steps: usize) -> Vec<f32> {
    let normed = rmsnorm_reference(input, &layer.input_gamma, steps, dims.hidden, dims.epsilon);
    let q = row_major_matmul(&normed, &layer.q_weight, steps, dims.q_out, dims.hidden);
    let k = row_major_matmul(&normed, &layer.k_weight, steps, dims.kv_out, dims.hidden);
    let v = row_major_matmul(&normed, &layer.v_weight, steps, dims.kv_out, dims.hidden);
    let q = rmsnorm_reference(
        &q,
        &layer.q_gamma,
        steps * dims.q_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let k = rmsnorm_reference(
        &k,
        &layer.k_gamma,
        steps * dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let q = rope_reference(
        &permute_bshd_to_bhsd(&q, steps, dims.q_heads, dims.head_dim),
        steps,
        dims.head_dim,
        0,
        dims.theta,
    );
    let k = rope_reference(
        &permute_bshd_to_bhsd(&k, steps, dims.kv_heads, dims.head_dim),
        steps,
        dims.head_dim,
        0,
        dims.theta,
    );
    let v = permute_bshd_to_bhsd(&v, steps, dims.kv_heads, dims.head_dim);
    let scores = attention_scores_reference(&q, &k, dims, steps, steps, 0);
    let probs = softmax_rows(&scores, dims.q_heads * steps, steps);
    let attended = attention_apply_value_reference(&probs, &v, dims, steps, steps);
    let projected = row_major_matmul(&attended, &layer.o_weight, steps, dims.hidden, dims.q_out);
    let attention_output = input
        .iter()
        .zip(projected.iter())
        .map(|(input, projected)| input + projected)
        .collect::<Vec<_>>();
    let post_norm = rmsnorm_reference(
        &attention_output,
        &layer.post_attention_gamma,
        steps,
        dims.hidden,
        dims.epsilon,
    );
    let gate = row_major_matmul(
        &post_norm,
        &layer.gate_weight,
        steps,
        dims.intermediate,
        dims.hidden,
    );
    let up = row_major_matmul(
        &post_norm,
        &layer.up_weight,
        steps,
        dims.intermediate,
        dims.hidden,
    );
    let swiglu = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    let down = row_major_matmul(
        &swiglu,
        &layer.down_weight,
        steps,
        dims.hidden,
        dims.intermediate,
    );
    attention_output
        .iter()
        .zip(down.iter())
        .map(|(hidden, down)| hidden + down)
        .collect()
}

fn deterministic_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 17.0)
        .collect()
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

fn rmsnorm_reference(
    input: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let sum = input[offset..offset + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inv_rms = (sum / cols as f32 + epsilon).sqrt().recip();
        for col in 0..cols {
            output[offset + col] = input[offset + col] * inv_rms * gamma[col];
        }
    }
    output
}

fn permute_bshd_to_bhsd(input: &[f32], steps: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for step in 0..steps {
        for head in 0..heads {
            for dim in 0..head_dim {
                let input_idx = ((step * heads + head) * head_dim) + dim;
                let output_idx = ((head * steps + step) * head_dim) + dim;
                output[output_idx] = input[input_idx];
            }
        }
    }
    output
}

fn rope_reference(
    input: &[f32],
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
    output
}

fn attention_scores_reference(
    q: &[f32],
    k: &[f32],
    dims: Dims,
    query_steps: usize,
    key_steps: usize,
    offset: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0; dims.q_heads * query_steps * key_steps];
    let n_rep = dims.q_heads / dims.kv_heads;
    for q_head in 0..dims.q_heads {
        let kv_head = q_head / n_rep;
        for query_step in 0..query_steps {
            for key_step in 0..key_steps {
                let score_idx = ((q_head * query_steps + query_step) * key_steps) + key_step;
                if key_step > offset + query_step {
                    scores[score_idx] = f32::NEG_INFINITY;
                    continue;
                }
                let q_base = ((q_head * query_steps + query_step) * dims.head_dim) as usize;
                let k_base = ((kv_head * key_steps + key_step) * dims.head_dim) as usize;
                let mut sum = 0.0;
                for dim in 0..dims.head_dim {
                    sum += q[q_base + dim] * k[k_base + dim];
                }
                scores[score_idx] = sum * dims.scale;
            }
        }
    }
    scores
}

fn softmax_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let row_max = input[offset..offset + cols]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = input[offset..offset + cols]
            .iter()
            .map(|value| (value - row_max).exp())
            .sum::<f32>();
        for col in 0..cols {
            output[offset + col] = (input[offset + col] - row_max).exp() / sum;
        }
    }
    output
}

fn attention_apply_value_reference(
    probs: &[f32],
    v: &[f32],
    dims: Dims,
    query_steps: usize,
    key_steps: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; query_steps * dims.q_heads * dims.head_dim];
    let n_rep = dims.q_heads / dims.kv_heads;
    for query_step in 0..query_steps {
        for q_head in 0..dims.q_heads {
            let kv_head = q_head / n_rep;
            for dim in 0..dims.head_dim {
                let mut sum = 0.0;
                for key_step in 0..key_steps {
                    let prob_idx = ((q_head * query_steps + query_step) * key_steps) + key_step;
                    let v_idx = ((kv_head * key_steps + key_step) * dims.head_dim) + dim;
                    sum += probs[prob_idx] * v[v_idx];
                }
                let out_idx = ((query_step * dims.q_heads + q_head) * dims.head_dim) + dim;
                output[out_idx] = sum;
            }
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

fn mean_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32
}
