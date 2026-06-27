use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::{
    ATTENTION_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE,
    ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use qwen3_hip_runtime::weights::TensorArchive;
use qwen3_hip_runtime::{Error, HipFunction, HipRuntime};

const LAYER_PREFIX: &str = "talker.model.layers.0";
const NEXT_LAYER_PREFIX: &str = "talker.model.layers.1";

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
        });
    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;

    let input_gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.input_layernorm.weight"))?;
    let q_gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.q_norm.weight"))?;
    let k_gamma = archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.k_norm.weight"))?;
    let (q_weight, q_in, q_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.q_proj.weight"))?;
    let (k_weight, k_in, k_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.k_proj.weight"))?;
    let (v_weight, v_in, v_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.v_proj.weight"))?;
    let (o_weight, o_in, o_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.o_proj.weight"))?;
    let post_attention_gamma =
        archive.vector_f32(&format!("{LAYER_PREFIX}.post_attention_layernorm.weight"))?;
    let (gate_weight, gate_in, gate_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.gate_proj.weight"))?;
    let (up_weight, up_in, up_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.up_proj.weight"))?;
    let (down_weight, down_in, down_out) =
        archive.linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.down_proj.weight"))?;
    let input_gamma_1 =
        archive.vector_f32(&format!("{NEXT_LAYER_PREFIX}.input_layernorm.weight"))?;
    let q_gamma_1 = archive.vector_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.q_norm.weight"))?;
    let k_gamma_1 = archive.vector_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.k_norm.weight"))?;
    let (q_weight_1, q_in_1, q_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.q_proj.weight"))?;
    let (k_weight_1, k_in_1, k_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.k_proj.weight"))?;
    let (v_weight_1, v_in_1, v_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.v_proj.weight"))?;
    let (o_weight_1, o_in_1, o_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.self_attn.o_proj.weight"))?;
    let post_attention_gamma_1 = archive.vector_f32(&format!(
        "{NEXT_LAYER_PREFIX}.post_attention_layernorm.weight"
    ))?;
    let (gate_weight_1, gate_in_1, gate_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.mlp.gate_proj.weight"))?;
    let (up_weight_1, up_in_1, up_out_1) =
        archive.linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.mlp.up_proj.weight"))?;
    let (down_weight_1, down_in_1, down_out_1) = archive
        .linear_weight_transposed_f32(&format!("{NEXT_LAYER_PREFIX}.mlp.down_proj.weight"))?;

    let hidden = input_gamma.len();
    let head_dim = q_gamma.len();
    if q_in != hidden || k_in != hidden || v_in != hidden || o_out != hidden {
        return Err(Error::InvalidInput(format!(
            "attention hidden shape mismatch: hidden={hidden}, q_in={q_in}, k_in={k_in}, v_in={v_in}, o_out={o_out}"
        )));
    }
    if k_gamma.len() != head_dim || q_out % head_dim != 0 || k_out % head_dim != 0 || v_out != k_out
    {
        return Err(Error::InvalidInput(format!(
            "attention head shape mismatch: q_out={q_out}, k_out={k_out}, v_out={v_out}, q_norm={}, k_norm={}",
            q_gamma.len(),
            k_gamma.len()
        )));
    }

    let batch = 1usize;
    let steps = 2usize;
    let rows = batch * steps;
    let q_heads = q_out / head_dim;
    let kv_heads = k_out / head_dim;
    if o_in != q_heads * head_dim {
        return Err(Error::InvalidInput(format!(
            "o_proj input {o_in} does not match q heads * head dim {}",
            q_heads * head_dim
        )));
    }
    if post_attention_gamma.len() != hidden
        || gate_in != hidden
        || up_in != hidden
        || gate_out != up_out
        || down_in != gate_out
        || down_out != hidden
    {
        return Err(Error::InvalidInput(format!(
            "MLP shape mismatch: post_norm={}, gate=({gate_in},{gate_out}), up=({up_in},{up_out}), down=({down_in},{down_out}), hidden={hidden}",
            post_attention_gamma.len()
        )));
    }
    if input_gamma_1.len() != hidden
        || q_gamma_1.len() != head_dim
        || k_gamma_1.len() != head_dim
        || q_in_1 != hidden
        || k_in_1 != hidden
        || v_in_1 != hidden
        || q_out_1 != q_out
        || k_out_1 != k_out
        || v_out_1 != v_out
        || o_in_1 != o_in
        || o_out_1 != o_out
        || post_attention_gamma_1.len() != hidden
        || gate_in_1 != hidden
        || up_in_1 != hidden
        || gate_out_1 != gate_out
        || up_out_1 != gate_out
        || down_in_1 != gate_out
        || down_out_1 != hidden
    {
        return Err(Error::InvalidInput(format!(
            "second layer shape mismatch: q=({q_in_1},{q_out_1}), k=({k_in_1},{k_out_1}), v=({v_in_1},{v_out_1}), o=({o_in_1},{o_out_1}), gate=({gate_in_1},{gate_out_1}), up=({up_in_1},{up_out_1}), down=({down_in_1},{down_out_1})"
        )));
    }

    let offset = 0usize;
    let theta = 10_000.0f32;
    let epsilon = 1e-6f32;
    let scale = (head_dim as f32).sqrt().recip();
    let hidden_input = deterministic_hidden(rows * hidden);
    let expected_attention = attention_block_reference(
        &hidden_input,
        &input_gamma,
        &q_weight,
        &k_weight,
        &v_weight,
        &o_weight,
        &q_gamma,
        &k_gamma,
        batch,
        steps,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        offset,
        theta,
        epsilon,
        scale,
    );
    let expected_layer0 = mlp_tail_reference(
        &expected_attention,
        &post_attention_gamma,
        &gate_weight,
        &up_weight,
        &down_weight,
        rows,
        hidden,
        gate_out,
        epsilon,
    );
    let expected_attention = attention_block_reference(
        &expected_layer0,
        &input_gamma_1,
        &q_weight_1,
        &k_weight_1,
        &v_weight_1,
        &o_weight_1,
        &q_gamma_1,
        &k_gamma_1,
        batch,
        steps,
        hidden,
        q_heads,
        kv_heads,
        head_dim,
        offset,
        theta,
        epsilon,
        scale,
    );
    let expected_layer = mlp_tail_reference(
        &expected_attention,
        &post_attention_gamma_1,
        &gate_weight_1,
        &up_weight_1,
        &down_weight_1,
        rows,
        hidden,
        gate_out,
        epsilon,
    );

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let rms_module = runtime.compile_module("rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
    let rope_module = runtime.compile_module("rope_bhsd_f32.cpp", ROPE_BHSD_F32_SOURCE)?;
    let layout_module = runtime.compile_module("layout_f32.cpp", LAYOUT_F32_SOURCE)?;
    let softmax_module = runtime.compile_module("softmax_f32.cpp", SOFTMAX_F32_SOURCE)?;
    let attention_module = runtime.compile_module("attention_f32.cpp", ATTENTION_F32_SOURCE)?;
    let elementwise_module =
        runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let rmsnorm = rms_module.function("rmsnorm_f32")?;
    let rope = rope_module.function("rope_bhsd_f32")?;
    let permute = layout_module.function("permute_bshd_to_bhsd_f32")?;
    let softmax = softmax_module.function("masked_softmax_f32")?;
    let scores_kernel = attention_module.function("attention_scores_causal_f32")?;
    let apply_value = attention_module.function("attention_apply_value_f32")?;
    let residual_add = elementwise_module.function("residual_add_f32")?;

    let hidden_dev = runtime.buffer_from_slice(&hidden_input)?;
    let input_gamma_dev = runtime.buffer_from_slice(&input_gamma)?;
    let input_gamma_1_dev = runtime.buffer_from_slice(&input_gamma_1)?;
    let post_attention_gamma_dev = runtime.buffer_from_slice(&post_attention_gamma)?;
    let post_attention_gamma_1_dev = runtime.buffer_from_slice(&post_attention_gamma_1)?;
    let q_gamma_dev = runtime.buffer_from_slice(&q_gamma)?;
    let q_gamma_1_dev = runtime.buffer_from_slice(&q_gamma_1)?;
    let k_gamma_dev = runtime.buffer_from_slice(&k_gamma)?;
    let k_gamma_1_dev = runtime.buffer_from_slice(&k_gamma_1)?;
    let q_weight_dev = runtime.buffer_from_slice(&q_weight)?;
    let q_weight_1_dev = runtime.buffer_from_slice(&q_weight_1)?;
    let k_weight_dev = runtime.buffer_from_slice(&k_weight)?;
    let k_weight_1_dev = runtime.buffer_from_slice(&k_weight_1)?;
    let v_weight_dev = runtime.buffer_from_slice(&v_weight)?;
    let v_weight_1_dev = runtime.buffer_from_slice(&v_weight_1)?;
    let o_weight_dev = runtime.buffer_from_slice(&o_weight)?;
    let o_weight_1_dev = runtime.buffer_from_slice(&o_weight_1)?;
    let gate_weight_dev = runtime.buffer_from_slice(&gate_weight)?;
    let gate_weight_1_dev = runtime.buffer_from_slice(&gate_weight_1)?;
    let up_weight_dev = runtime.buffer_from_slice(&up_weight)?;
    let up_weight_1_dev = runtime.buffer_from_slice(&up_weight_1)?;
    let down_weight_dev = runtime.buffer_from_slice(&down_weight)?;
    let down_weight_1_dev = runtime.buffer_from_slice(&down_weight_1)?;
    let normed_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let q_proj_dev = runtime.empty_buffer::<f32>(rows * q_out)?;
    let k_proj_dev = runtime.empty_buffer::<f32>(rows * k_out)?;
    let v_proj_dev = runtime.empty_buffer::<f32>(rows * v_out)?;
    let q_norm_dev = runtime.empty_buffer::<f32>(rows * q_out)?;
    let k_norm_dev = runtime.empty_buffer::<f32>(rows * k_out)?;
    let q_bhsd_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * head_dim)?;
    let k_bhsd_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;
    let v_bhsd_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;
    let q_rope_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * head_dim)?;
    let k_rope_dev = runtime.empty_buffer::<f32>(batch * kv_heads * steps * head_dim)?;
    let scores_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * steps)?;
    let probs_dev = runtime.empty_buffer::<f32>(batch * q_heads * steps * steps)?;
    let attended_dev = runtime.empty_buffer::<f32>(rows * q_heads * head_dim)?;
    let projected_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let output_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let post_norm_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let gate_dev = runtime.empty_buffer::<f32>(rows * gate_out)?;
    let up_dev = runtime.empty_buffer::<f32>(rows * gate_out)?;
    let swiglu_dev = runtime.empty_buffer::<f32>(rows * gate_out)?;
    let mlp_down_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let layer_output_dev = runtime.empty_buffer::<f32>(rows * hidden)?;
    let final_output_dev = runtime.empty_buffer::<f32>(rows * hidden)?;

    launch_rmsnorm(
        &rmsnorm,
        hidden_dev.as_ptr(),
        input_gamma_dev.as_ptr(),
        normed_dev.as_mut_ptr(),
        rows,
        hidden,
        epsilon,
    )?;
    blas.sgemm_row_major(&normed_dev, &q_weight_dev, &q_proj_dev, rows, q_out, hidden)?;
    blas.sgemm_row_major(&normed_dev, &k_weight_dev, &k_proj_dev, rows, k_out, hidden)?;
    blas.sgemm_row_major(&normed_dev, &v_weight_dev, &v_proj_dev, rows, v_out, hidden)?;
    launch_rmsnorm(
        &rmsnorm,
        q_proj_dev.as_ptr(),
        q_gamma_dev.as_ptr(),
        q_norm_dev.as_mut_ptr(),
        rows * q_heads,
        head_dim,
        epsilon,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        k_proj_dev.as_ptr(),
        k_gamma_dev.as_ptr(),
        k_norm_dev.as_mut_ptr(),
        rows * kv_heads,
        head_dim,
        epsilon,
    )?;
    launch_permute(
        &permute,
        q_norm_dev.as_ptr(),
        q_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        q_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        k_norm_dev.as_ptr(),
        k_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        v_proj_dev.as_ptr(),
        v_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_rope(
        &rope,
        q_bhsd_dev.as_ptr(),
        q_rope_dev.as_mut_ptr(),
        batch * q_heads * steps * head_dim,
        q_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    launch_rope(
        &rope,
        k_bhsd_dev.as_ptr(),
        k_rope_dev.as_mut_ptr(),
        batch * kv_heads * steps * head_dim,
        kv_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    launch_attention_scores(
        &scores_kernel,
        q_rope_dev.as_ptr(),
        k_rope_dev.as_ptr(),
        scores_dev.as_mut_ptr(),
        batch,
        q_heads,
        kv_heads,
        steps,
        steps,
        head_dim,
        offset,
        scale,
    )?;
    launch_softmax(
        &softmax,
        scores_dev.as_ptr(),
        probs_dev.as_mut_ptr(),
        batch * q_heads * steps,
        steps,
    )?;
    launch_apply_value(
        &apply_value,
        probs_dev.as_ptr(),
        v_bhsd_dev.as_ptr(),
        attended_dev.as_mut_ptr(),
        batch,
        q_heads,
        kv_heads,
        steps,
        steps,
        head_dim,
    )?;
    blas.sgemm_row_major(
        &attended_dev,
        &o_weight_dev,
        &projected_dev,
        rows,
        hidden,
        q_heads * head_dim,
    )?;
    launch_ternary(
        &residual_add,
        hidden_dev.as_ptr(),
        projected_dev.as_ptr(),
        output_dev.as_mut_ptr(),
        rows * hidden,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        output_dev.as_ptr(),
        post_attention_gamma_dev.as_ptr(),
        post_norm_dev.as_mut_ptr(),
        rows,
        hidden,
        epsilon,
    )?;
    blas.sgemm_row_major(
        &post_norm_dev,
        &gate_weight_dev,
        &gate_dev,
        rows,
        gate_out,
        hidden,
    )?;
    blas.sgemm_row_major(
        &post_norm_dev,
        &up_weight_dev,
        &up_dev,
        rows,
        gate_out,
        hidden,
    )?;
    let swiglu = elementwise_module.function("swiglu_f32")?;
    launch_ternary(
        &swiglu,
        gate_dev.as_ptr(),
        up_dev.as_ptr(),
        swiglu_dev.as_mut_ptr(),
        rows * gate_out,
    )?;
    blas.sgemm_row_major(
        &swiglu_dev,
        &down_weight_dev,
        &mlp_down_dev,
        rows,
        hidden,
        gate_out,
    )?;
    launch_ternary(
        &residual_add,
        output_dev.as_ptr(),
        mlp_down_dev.as_ptr(),
        layer_output_dev.as_mut_ptr(),
        rows * hidden,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        layer_output_dev.as_ptr(),
        input_gamma_1_dev.as_ptr(),
        normed_dev.as_mut_ptr(),
        rows,
        hidden,
        epsilon,
    )?;
    blas.sgemm_row_major(
        &normed_dev,
        &q_weight_1_dev,
        &q_proj_dev,
        rows,
        q_out,
        hidden,
    )?;
    blas.sgemm_row_major(
        &normed_dev,
        &k_weight_1_dev,
        &k_proj_dev,
        rows,
        k_out,
        hidden,
    )?;
    blas.sgemm_row_major(
        &normed_dev,
        &v_weight_1_dev,
        &v_proj_dev,
        rows,
        v_out,
        hidden,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        q_proj_dev.as_ptr(),
        q_gamma_1_dev.as_ptr(),
        q_norm_dev.as_mut_ptr(),
        rows * q_heads,
        head_dim,
        epsilon,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        k_proj_dev.as_ptr(),
        k_gamma_1_dev.as_ptr(),
        k_norm_dev.as_mut_ptr(),
        rows * kv_heads,
        head_dim,
        epsilon,
    )?;
    launch_permute(
        &permute,
        q_norm_dev.as_ptr(),
        q_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        q_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        k_norm_dev.as_ptr(),
        k_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_permute(
        &permute,
        v_proj_dev.as_ptr(),
        v_bhsd_dev.as_mut_ptr(),
        batch,
        steps,
        kv_heads,
        head_dim,
    )?;
    launch_rope(
        &rope,
        q_bhsd_dev.as_ptr(),
        q_rope_dev.as_mut_ptr(),
        batch * q_heads * steps * head_dim,
        q_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    launch_rope(
        &rope,
        k_bhsd_dev.as_ptr(),
        k_rope_dev.as_mut_ptr(),
        batch * kv_heads * steps * head_dim,
        kv_heads,
        steps,
        head_dim,
        offset,
        theta,
    )?;
    launch_attention_scores(
        &scores_kernel,
        q_rope_dev.as_ptr(),
        k_rope_dev.as_ptr(),
        scores_dev.as_mut_ptr(),
        batch,
        q_heads,
        kv_heads,
        steps,
        steps,
        head_dim,
        offset,
        scale,
    )?;
    launch_softmax(
        &softmax,
        scores_dev.as_ptr(),
        probs_dev.as_mut_ptr(),
        batch * q_heads * steps,
        steps,
    )?;
    launch_apply_value(
        &apply_value,
        probs_dev.as_ptr(),
        v_bhsd_dev.as_ptr(),
        attended_dev.as_mut_ptr(),
        batch,
        q_heads,
        kv_heads,
        steps,
        steps,
        head_dim,
    )?;
    blas.sgemm_row_major(
        &attended_dev,
        &o_weight_1_dev,
        &projected_dev,
        rows,
        hidden,
        q_heads * head_dim,
    )?;
    launch_ternary(
        &residual_add,
        layer_output_dev.as_ptr(),
        projected_dev.as_ptr(),
        output_dev.as_mut_ptr(),
        rows * hidden,
    )?;
    launch_rmsnorm(
        &rmsnorm,
        output_dev.as_ptr(),
        post_attention_gamma_1_dev.as_ptr(),
        post_norm_dev.as_mut_ptr(),
        rows,
        hidden,
        epsilon,
    )?;
    blas.sgemm_row_major(
        &post_norm_dev,
        &gate_weight_1_dev,
        &gate_dev,
        rows,
        gate_out,
        hidden,
    )?;
    blas.sgemm_row_major(
        &post_norm_dev,
        &up_weight_1_dev,
        &up_dev,
        rows,
        gate_out,
        hidden,
    )?;
    launch_ternary(
        &swiglu,
        gate_dev.as_ptr(),
        up_dev.as_ptr(),
        swiglu_dev.as_mut_ptr(),
        rows * gate_out,
    )?;
    blas.sgemm_row_major(
        &swiglu_dev,
        &down_weight_1_dev,
        &mlp_down_dev,
        rows,
        hidden,
        gate_out,
    )?;
    launch_ternary(
        &residual_add,
        output_dev.as_ptr(),
        mlp_down_dev.as_ptr(),
        final_output_dev.as_mut_ptr(),
        rows * hidden,
    )?;
    runtime.synchronize()?;

    let attention_actual = output_dev.copy_to_host()?;
    let attention_max_abs = max_abs(&attention_actual, &expected_attention);
    let attention_mean_abs = mean_abs(&attention_actual, &expected_attention);
    if attention_max_abs > 2e-4 || attention_mean_abs > 2e-5 {
        return Err(Error::InvalidInput(format!(
            "attention block mismatch: max_abs={attention_max_abs}, mean_abs={attention_mean_abs}"
        )));
    }

    let layer_actual = final_output_dev.copy_to_host()?;
    let layer_max_abs = max_abs(&layer_actual, &expected_layer);
    let layer_mean_abs = mean_abs(&layer_actual, &expected_layer);
    if layer_max_abs > 3e-4 || layer_mean_abs > 3e-5 {
        return Err(Error::InvalidInput(format!(
            "decoder stack mismatch: max_abs={layer_max_abs}, mean_abs={layer_mean_abs}"
        )));
    }

    println!(
        "Attention/decoder stack smoke OK: layers={LAYER_PREFIX},{NEXT_LAYER_PREFIX}, batch={batch}, steps={steps}, hidden={hidden}, q_heads={q_heads}, kv_heads={kv_heads}, head_dim={head_dim}, attention_max_abs={attention_max_abs}, attention_mean_abs={attention_mean_abs}, stack_max_abs={layer_max_abs}, stack_mean_abs={layer_mean_abs}, first8={:?}",
        &layer_actual[..8]
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut gamma = gamma;
    let mut output = output;
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut epsilon = epsilon;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut gamma as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut epsilon as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

#[allow(clippy::too_many_arguments)]
fn launch_permute(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = batch * steps * heads * head_dim;
    let mut input = input;
    let mut output = output;
    let mut batch_i32 = batch as i32;
    let mut steps_i32 = steps as i32;
    let mut heads_i32 = heads as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (total as u32).div_ceil(block).into_grid(),
        (block, 1, 1),
        0,
        &mut params,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_rope(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    total: usize,
    heads: usize,
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut heads_i32 = heads as i32;
    let mut steps_i32 = steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut theta = theta;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut theta as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (total as u32).div_ceil(block).into_grid(),
        (block, 1, 1),
        0,
        &mut params,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_scores(
    function: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
) -> qwen3_hip_runtime::Result<()> {
    let total = batch * q_heads * query_steps * key_steps;
    let mut q = q;
    let mut k = k;
    let mut scores = scores;
    let mut batch_i32 = batch as i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut scale = scale;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut q as *mut *const c_void as *mut c_void,
        &mut k as *mut *const c_void as *mut c_void,
        &mut scores as *mut *mut c_void as *mut c_void,
        &mut batch_i32 as *mut i32 as *mut c_void,
        &mut q_heads_i32 as *mut i32 as *mut c_void,
        &mut kv_heads_i32 as *mut i32 as *mut c_void,
        &mut query_steps_i32 as *mut i32 as *mut c_void,
        &mut key_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut scale as *mut f32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (total as u32).div_ceil(block).into_grid(),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_softmax(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut output = output;
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut active_cols_i32 = cols as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut active_cols_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

#[allow(clippy::too_many_arguments)]
fn launch_apply_value(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = batch * query_steps * q_heads * head_dim;
    let mut probs = probs;
    let mut v = v;
    let mut output = output;
    let mut batch_i32 = batch as i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut probs as *mut *const c_void as *mut c_void,
        &mut v as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch_i32 as *mut i32 as *mut c_void,
        &mut q_heads_i32 as *mut i32 as *mut c_void,
        &mut kv_heads_i32 as *mut i32 as *mut c_void,
        &mut query_steps_i32 as *mut i32 as *mut c_void,
        &mut key_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        (total as u32).div_ceil(block).into_grid(),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_ternary(
    function: &HipFunction,
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
    function.launch(
        (total as u32).div_ceil(block).into_grid(),
        (block, 1, 1),
        0,
        &mut params,
    )
}

trait GridExt {
    fn into_grid(self) -> (u32, u32, u32);
}

impl GridExt for u32 {
    fn into_grid(self) -> (u32, u32, u32) {
        (self, 1, 1)
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_block_reference(
    hidden_input: &[f32],
    input_gamma: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    o_weight: &[f32],
    q_gamma: &[f32],
    k_gamma: &[f32],
    batch: usize,
    steps: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
    epsilon: f32,
    scale: f32,
) -> Vec<f32> {
    let rows = batch * steps;
    let q_out = q_heads * head_dim;
    let kv_out = kv_heads * head_dim;
    let normed = rmsnorm_reference(hidden_input, input_gamma, rows, hidden, epsilon);
    let q = row_major_matmul(&normed, q_weight, rows, q_out, hidden);
    let k = row_major_matmul(&normed, k_weight, rows, kv_out, hidden);
    let v = row_major_matmul(&normed, v_weight, rows, kv_out, hidden);
    let q = rmsnorm_reference(&q, q_gamma, rows * q_heads, head_dim, epsilon);
    let k = rmsnorm_reference(&k, k_gamma, rows * kv_heads, head_dim, epsilon);
    let q = rope_reference(
        &permute_bshd_to_bhsd(&q, batch, steps, q_heads, head_dim),
        steps,
        head_dim,
        offset,
        theta,
    );
    let k = rope_reference(
        &permute_bshd_to_bhsd(&k, batch, steps, kv_heads, head_dim),
        steps,
        head_dim,
        offset,
        theta,
    );
    let v = permute_bshd_to_bhsd(&v, batch, steps, kv_heads, head_dim);
    let scores = attention_scores_reference(
        &q, &k, batch, q_heads, kv_heads, steps, steps, head_dim, offset, scale,
    );
    let probs = softmax_rows(&scores, batch * q_heads * steps, steps);
    let attended = attention_apply_value_reference(
        &probs, &v, batch, q_heads, kv_heads, steps, steps, head_dim,
    );
    let projected = row_major_matmul(&attended, o_weight, rows, hidden, q_heads * head_dim);
    hidden_input
        .iter()
        .zip(projected.iter())
        .map(|(hidden, projected)| hidden + projected)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mlp_tail_reference(
    hidden: &[f32],
    gamma: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    rows: usize,
    hidden_size: usize,
    intermediate: usize,
    epsilon: f32,
) -> Vec<f32> {
    let normed = rmsnorm_reference(hidden, gamma, rows, hidden_size, epsilon);
    let gate = row_major_matmul(&normed, gate_weight, rows, intermediate, hidden_size);
    let up = row_major_matmul(&normed, up_weight, rows, intermediate, hidden_size);
    let swiglu = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    let down = row_major_matmul(&swiglu, down_weight, rows, hidden_size, intermediate);
    hidden
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

fn permute_bshd_to_bhsd(
    input: &[f32],
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for step in 0..steps {
            for head in 0..heads {
                for dim in 0..head_dim {
                    let input_idx =
                        (((batch_index * steps + step) * heads + head) * head_dim) + dim;
                    let output_idx =
                        (((batch_index * heads + head) * steps + step) * head_dim) + dim;
                    output[output_idx] = input[input_idx];
                }
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

#[allow(clippy::too_many_arguments)]
fn attention_scores_reference(
    q: &[f32],
    k: &[f32],
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0; batch * q_heads * query_steps * key_steps];
    let n_rep = q_heads / kv_heads;
    for batch_index in 0..batch {
        for q_head in 0..q_heads {
            let kv_head = q_head / n_rep;
            for query_step in 0..query_steps {
                for key_step in 0..key_steps {
                    let score_idx = (((batch_index * q_heads + q_head) * query_steps + query_step)
                        * key_steps)
                        + key_step;
                    if key_step > offset + query_step {
                        scores[score_idx] = f32::NEG_INFINITY;
                        continue;
                    }
                    let q_base = (((batch_index * q_heads + q_head) * query_steps + query_step)
                        * head_dim) as usize;
                    let k_base = (((batch_index * kv_heads + kv_head) * key_steps + key_step)
                        * head_dim) as usize;
                    let mut sum = 0.0;
                    for dim in 0..head_dim {
                        sum += q[q_base + dim] * k[k_base + dim];
                    }
                    scores[score_idx] = sum * scale;
                }
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

#[allow(clippy::too_many_arguments)]
fn attention_apply_value_reference(
    probs: &[f32],
    v: &[f32],
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_steps * q_heads * head_dim];
    let n_rep = q_heads / kv_heads;
    for batch_index in 0..batch {
        for query_step in 0..query_steps {
            for q_head in 0..q_heads {
                let kv_head = q_head / n_rep;
                for dim in 0..head_dim {
                    let mut sum = 0.0;
                    for key_step in 0..key_steps {
                        let prob_idx = (((batch_index * q_heads + q_head) * query_steps
                            + query_step)
                            * key_steps)
                            + key_step;
                        let v_idx = (((batch_index * kv_heads + kv_head) * key_steps + key_step)
                            * head_dim)
                            + dim;
                        sum += probs[prob_idx] * v[v_idx];
                    }
                    let out_idx = (((batch_index * query_steps + query_step) * q_heads + q_head)
                        * head_dim)
                        + dim;
                    output[out_idx] = sum;
                }
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
