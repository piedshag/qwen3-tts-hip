pub const RMSNORM_F32_SOURCE: &str = r#"
extern "C" __global__ void rmsnorm_f32(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];

    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        sum += value * value;
    }
    scratch[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(scratch[0] / (float)cols + epsilon);
    for (int col = tid; col < cols; col += blockDim.x) {
        output[row * cols + col] = input[row * cols + col] * inv_rms * gamma[col];
    }
}
"#;

pub const ROPE_BHSD_F32_SOURCE: &str = r#"
extern "C" __global__ void rope_bhsd_f32(
    const float* input,
    float* output,
    int total,
    int heads,
    int steps,
    int head_dim,
    int offset,
    float theta_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int half = head_dim / 2;
    int dim_index = idx % head_dim;
    int step_index = (idx / head_dim) % steps;
    int pair_index = dim_index % half;
    int base = idx - dim_index;
    int first_index = base + pair_index;
    int second_index = base + pair_index + half;

    float exponent = (float)(pair_index * 2) / (float)head_dim;
    float angle = (float)(offset + step_index) / powf(theta_base, exponent);
    float angle_cos = cosf(angle);
    float angle_sin = sinf(angle);
    float first = input[first_index];
    float second = input[second_index];

    if (dim_index < half) {
        output[idx] = first * angle_cos - second * angle_sin;
    } else {
        output[idx] = second * angle_cos + first * angle_sin;
    }
}
"#;

pub const ELEMENTWISE_F32_SOURCE: &str = r#"
extern "C" __global__ void residual_add_f32(
    const float* residual,
    const float* update,
    float* output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = residual[idx] + update[idx];
    }
}

extern "C" __global__ void bias_add_f32(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        output[idx] = input[idx] + bias[idx % cols];
    }
}

extern "C" __global__ void swiglu_f32(
    const float* gate,
    const float* up,
    float* output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float value = gate[idx];
        output[idx] = value / (1.0f + expf(-value)) * up[idx];
    }
}
"#;

pub const SOFTMAX_F32_SOURCE: &str = r#"
extern "C" __global__ void masked_softmax_f32(
    const float* input,
    float* output,
    int rows,
    int cols,
    int active_cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];
    if (row >= rows) {
        return;
    }

    float local_max = -3.4028234663852886e38f;
    for (int col = tid; col < active_cols; col += blockDim.x) {
        float value = input[row * cols + col];
        local_max = fmaxf(local_max, value);
    }
    scratch[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = scratch[0];

    float local_sum = 0.0f;
    for (int col = tid; col < active_cols; col += blockDim.x) {
        local_sum += expf(input[row * cols + col] - row_max);
    }
    scratch[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / scratch[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        if (col < active_cols) {
            output[row * cols + col] = expf(input[row * cols + col] - row_max) * inv_sum;
        } else {
            output[row * cols + col] = 0.0f;
        }
    }
}
"#;

pub const ARGMAX_F32_SOURCE: &str = r#"
extern "C" __global__ void argmax_rows_f32(
    const float* input,
    int* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned char shared[];
    float* values = (float*)shared;
    int* indices = (int*)&values[blockDim.x];
    if (row >= rows) {
        return;
    }

    float best_value = -3.4028234663852886e38f;
    int best_index = 0;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        if (value > best_value || (value == best_value && col < best_index)) {
            best_value = value;
            best_index = col;
        }
    }
    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            int other_index = indices[tid + stride];
            if (other_value > values[tid] || (other_value == values[tid] && other_index < indices[tid])) {
                values[tid] = other_value;
                indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = indices[0];
    }
}
"#;

pub const EMBEDDING_F32_SOURCE: &str = r#"
extern "C" __global__ void embedding_lookup_f32(
    const float* table,
    const int* token,
    float* output,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        output[idx] = table[token[0] * cols + idx];
    }
}

extern "C" __global__ void store_token_i32(
    const int* token,
    int* output,
    int offset
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[offset] = token[0];
    }
}
"#;

pub const SUPPRESSION_F32_SOURCE: &str = r#"
extern "C" __global__ void suppress_codec_logits_f32(
    const float* input,
    float* output,
    int vocab_size,
    int eos_token
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        int suppress_start = vocab_size - 1024;
        float value = input[idx];
        if (idx >= suppress_start && idx != eos_token) {
            value = -__int_as_float(0x7f800000);
        }
        output[idx] = value;
    }
}

extern "C" __global__ void sample_topk_f32(
    const float* logits,
    int* token_out,
    int vocab_size,
    int sort_size,
    int top_k,
    float temperature,
    float random_value
) {
    extern __shared__ unsigned char shared_bytes[];
    float* values = reinterpret_cast<float*>(shared_bytes);
    int* tokens = reinterpret_cast<int*>(values + sort_size);

    for (int idx = threadIdx.x; idx < sort_size; idx += blockDim.x) {
        if (idx < vocab_size && isfinite(logits[idx])) {
            values[idx] = logits[idx];
            tokens[idx] = idx;
        } else {
            values[idx] = -3.4028234663852886e38f;
            tokens[idx] = idx;
        }
    }
    __syncthreads();

    for (int size = 2; size <= sort_size; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int idx = threadIdx.x; idx < sort_size; idx += blockDim.x) {
                int partner = idx ^ stride;
                if (partner > idx) {
                    bool ascending = (idx & size) != 0;
                    float a = values[idx];
                    float b = values[partner];
                    bool swap = ascending ? (a > b) : (a < b);
                    if (swap) {
                        values[idx] = b;
                        values[partner] = a;
                        int token = tokens[idx];
                        tokens[idx] = tokens[partner];
                        tokens[partner] = token;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int kept = top_k < vocab_size ? top_k : vocab_size;
        float max_logit = values[0];
        float sum = 0.0f;
        for (int idx = 0; idx < kept; ++idx) {
            float prob = expf((values[idx] - max_logit) / temperature);
            values[idx] = prob;
            sum += prob;
        }
        if (!isfinite(sum) || sum <= 0.0f) {
            token_out[0] = tokens[0];
            return;
        }
        float target = random_value * sum;
        for (int idx = 0; idx < kept; ++idx) {
            if (target <= values[idx]) {
                token_out[0] = tokens[idx];
                return;
            }
            target -= values[idx];
        }
        token_out[0] = tokens[kept - 1];
    }
}

extern "C" __global__ void sample_topk_random_buffer_f32(
    const float* logits,
    int* token_out,
    const float* random_values,
    int random_offset,
    int vocab_size,
    int sort_size,
    int top_k,
    float temperature
) {
    extern __shared__ unsigned char shared_bytes[];
    float* values = reinterpret_cast<float*>(shared_bytes);
    int* tokens = reinterpret_cast<int*>(values + sort_size);

    for (int idx = threadIdx.x; idx < sort_size; idx += blockDim.x) {
        if (idx < vocab_size && isfinite(logits[idx])) {
            values[idx] = logits[idx];
            tokens[idx] = idx;
        } else {
            values[idx] = -3.4028234663852886e38f;
            tokens[idx] = idx;
        }
    }
    __syncthreads();

    for (int size = 2; size <= sort_size; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int idx = threadIdx.x; idx < sort_size; idx += blockDim.x) {
                int partner = idx ^ stride;
                if (partner > idx) {
                    bool ascending = (idx & size) != 0;
                    float a = values[idx];
                    float b = values[partner];
                    bool swap = ascending ? (a > b) : (a < b);
                    if (swap) {
                        values[idx] = b;
                        values[partner] = a;
                        int token = tokens[idx];
                        tokens[idx] = tokens[partner];
                        tokens[partner] = token;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int kept = top_k < vocab_size ? top_k : vocab_size;
        float max_logit = values[0];
        float sum = 0.0f;
        for (int idx = 0; idx < kept; ++idx) {
            float prob = expf((values[idx] - max_logit) / temperature);
            values[idx] = prob;
            sum += prob;
        }
        if (!isfinite(sum) || sum <= 0.0f) {
            token_out[0] = tokens[0];
            return;
        }
        float target = random_values[random_offset] * sum;
        for (int idx = 0; idx < kept; ++idx) {
            if (target <= values[idx]) {
                token_out[0] = tokens[idx];
                return;
            }
            target -= values[idx];
        }
        token_out[0] = tokens[kept - 1];
    }
}

extern "C" __global__ void apply_repetition_penalty_f32(
    float* logits,
    const int* tokens,
    int token_count,
    int vocab_size,
    float penalty
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || penalty == 1.0f) {
        return;
    }
    for (int idx = 0; idx < token_count; ++idx) {
        int token = tokens[idx];
        if (token < 0 || token >= vocab_size) {
            continue;
        }
        float value = logits[token];
        logits[token] = value < 0.0f ? value * penalty : value / penalty;
    }
}
"#;

pub const LAYOUT_F32_SOURCE: &str = r#"
extern "C" __global__ void permute_bshd_to_bhsd_f32(
    const float* input,
    float* output,
    int batch,
    int steps,
    int heads,
    int head_dim,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int dim = idx % head_dim;
    int head = (idx / head_dim) % heads;
    int step = (idx / (head_dim * heads)) % steps;
    int batch_index = idx / (head_dim * heads * steps);
    int output_idx = (((batch_index * heads + head) * steps + step) * head_dim) + dim;
    output[output_idx] = input[idx];
}
"#;

pub const CODEC_INITIAL_F32_SOURCE: &str = r#"
extern "C" __global__ void codec_rvq_project_f32(
    const int* codes,
    const float* first_codebook,
    const float* rest_codebooks,
    const float* first_weight,
    const float* rest_weight,
    float* output,
    int frames,
    int code_groups,
    int codebook_size,
    int codebook_dim,
    int projected_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * projected_dim;
    if (idx >= total) {
        return;
    }

    int frame = idx % frames;
    int out_channel = idx / frames;
    int first_code = codes[frame * code_groups] % codebook_size;
    if (first_code < 0) {
        first_code += codebook_size;
    }

    float sum = 0.0f;
    for (int dim = 0; dim < codebook_dim; dim++) {
        float first_value = first_codebook[first_code * codebook_dim + dim];
        float rest_value = 0.0f;
        for (int group = 1; group < code_groups; group++) {
            int code = codes[frame * code_groups + group] % codebook_size;
            if (code < 0) {
                code += codebook_size;
            }
            int table = group - 1;
            rest_value += rest_codebooks[(table * codebook_size + code) * codebook_dim + dim];
        }
        sum += first_weight[out_channel * codebook_dim + dim] * first_value;
        sum += rest_weight[out_channel * codebook_dim + dim] * rest_value;
    }
    output[out_channel * frames + frame] = sum;
}

extern "C" __global__ void codec_causal_conv1d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int causal_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * out_channels;
    if (idx >= total) {
        return;
    }

    int frame = idx % frames;
    int out_channel = idx / frames;
    float sum = bias[out_channel];
    for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        for (int k = 0; k < kernel_size; k++) {
            int source_frame = frame + k - causal_padding;
            if (source_frame >= 0 && source_frame < frames) {
                float value = input[in_channel * frames + source_frame];
                float w = weight[(out_channel * in_channels + in_channel) * kernel_size + k];
                sum += value * w;
            }
        }
    }
    output[out_channel * frames + frame] = sum;
}

extern "C" __global__ void codec_transpose_ct_to_tc_f32(
    const float* input,
    float* output,
    int channels,
    int frames,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int channel = idx / frames;
    int frame = idx % frames;
    output[frame * channels + channel] = input[channel * frames + frame];
}

extern "C" __global__ void codec_transpose_tc_to_ct_f32(
    const float* input,
    float* output,
    int frames,
    int channels,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int frame = idx / channels;
    int channel = idx % channels;
    output[channel * frames + frame] = input[frame * channels + channel];
}

extern "C" __global__ void codec_scaled_residual_add_f32(
    const float* residual,
    const float* update,
    const float* scale,
    float* output,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        output[idx] = residual[idx] + update[idx] * scale[col];
    }
}

extern "C" __global__ void codec_bias_add_f32(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        output[idx] = input[idx] + bias[col];
    }
}

extern "C" __global__ void codec_gelu_f32(
    const float* input,
    float* output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
}

extern "C" __global__ void codec_layernorm_f32(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratch[];
    float* sum_scratch = scratch;
    float* sq_scratch = &scratch[blockDim.x];
    if (row >= rows) {
        return;
    }

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        local_sum += value;
        local_sq += value * value;
    }
    sum_scratch[tid] = local_sum;
    sq_scratch[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
            sq_scratch[tid] += sq_scratch[tid + stride];
        }
        __syncthreads();
    }

    float mean = sum_scratch[0] / (float)cols;
    float variance = fmaxf(sq_scratch[0] / (float)cols - mean * mean, 0.0f);
    float inv_std = rsqrtf(variance + epsilon);
    for (int col = tid; col < cols; col += blockDim.x) {
        float value = input[row * cols + col];
        output[row * cols + col] = (value - mean) * inv_std * gamma[col] + beta[col];
    }
}

extern "C" __global__ void codec_transconv1d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_frames,
    int out_frames,
    int channels,
    int kernel_size,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_frames * channels;
    if (idx >= total) {
        return;
    }
    int out_t = idx % out_frames;
    int out_c = idx / out_frames;
    float sum = bias[out_c];
    for (int k = 0; k < kernel_size; k++) {
        int rem = out_t - k;
        if (rem >= 0 && rem % stride == 0) {
            int in_t = rem / stride;
            if (in_t >= 0 && in_t < in_frames) {
                for (int in_c = 0; in_c < channels; in_c++) {
                    float value = input[in_c * in_frames + in_t];
                    float w = weight[(in_c * channels + out_c) * kernel_size + k];
                    sum += value * w;
                }
            }
        }
    }
    output[out_c * out_frames + out_t] = sum;
}

extern "C" __global__ void codec_depthwise_causal_conv1d_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int frames,
    int channels,
    int kernel_size,
    int causal_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * channels;
    if (idx >= total) {
        return;
    }
    int t = idx % frames;
    int c = idx / frames;
    float sum = bias[c];
    for (int k = 0; k < kernel_size; k++) {
        int source_t = t + k - causal_padding;
        if (source_t >= 0 && source_t < frames) {
            sum += input[c * frames + source_t] * weight[c * kernel_size + k];
        }
    }
    output[c * frames + t] = sum;
}

extern "C" __global__ void codec_convnext_residual_f32(
    const float* residual_ct,
    const float* update_tc,
    const float* gamma,
    float* output_ct,
    int channels,
    int frames,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx / frames;
    int t = idx % frames;
    output_ct[idx] = residual_ct[idx] + update_tc[t * channels + c] * gamma[c];
}

extern "C" __global__ void codec_snake_beta_f32(
    const float* input,
    const float* alpha,
    const float* beta,
    float* output,
    int channels,
    int frames,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int c = idx / frames;
    float x = input[idx];
    float a = expf(alpha[c]);
    float b = expf(beta[c]);
    float s = sinf(x * a);
    output[idx] = x + (s * s) / (b + 1.0e-9f);
}

extern "C" __global__ void codec_causal_conv1d_dilated_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int dilation,
    int causal_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * out_channels;
    if (idx >= total) {
        return;
    }
    int t = idx % frames;
    int out_c = idx / frames;
    float sum = bias[out_c];
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int k = 0; k < kernel_size; k++) {
            int source_t = t + k * dilation - causal_padding;
            if (source_t >= 0 && source_t < frames) {
                float value = input[in_c * frames + source_t];
                float w = weight[(out_c * in_channels + in_c) * kernel_size + k];
                sum += value * w;
            }
        }
    }
    output[out_c * frames + t] = sum;
}

extern "C" __global__ void codec_causal_conv1d_dilated_tiled_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int dilation,
    int causal_padding
) {
    const int oc_tile = 16;
    const int red_lanes = 16;
    int lane_oc = threadIdx.x;
    int lane_red = threadIdx.y;
    int t = blockIdx.x;
    int out_c = blockIdx.y * oc_tile + lane_oc;
    extern __shared__ float scratch[];
    int scratch_idx = lane_red * oc_tile + lane_oc;

    float sum = 0.0f;
    if (out_c < out_channels && t < frames) {
        int reductions = in_channels * kernel_size;
        for (int red = lane_red; red < reductions; red += red_lanes) {
            int k = red % kernel_size;
            int in_c = red / kernel_size;
            int source_t = t + k * dilation - causal_padding;
            if (source_t >= 0 && source_t < frames) {
                float value = input[in_c * frames + source_t];
                float w = weight[(out_c * in_channels + in_c) * kernel_size + k];
                sum += value * w;
            }
        }
    }
    scratch[scratch_idx] = sum;
    __syncthreads();

    for (int stride = red_lanes / 2; stride > 0; stride >>= 1) {
        if (lane_red < stride) {
            scratch[scratch_idx] += scratch[(lane_red + stride) * oc_tile + lane_oc];
        }
        __syncthreads();
    }
    if (lane_red == 0 && out_c < out_channels && t < frames) {
        output[out_c * frames + t] = scratch[lane_oc] + bias[out_c];
    }
}

extern "C" __global__ void codec_transconv1d_channels_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_frames,
    int out_frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_frames * out_channels;
    if (idx >= total) {
        return;
    }
    int out_t = idx % out_frames;
    int out_c = idx / out_frames;
    float sum = bias[out_c];
    for (int k = 0; k < kernel_size; k++) {
        int rem = out_t - k;
        if (rem >= 0 && rem % stride == 0) {
            int in_t = rem / stride;
            if (in_t >= 0 && in_t < in_frames) {
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    float value = input[in_c * in_frames + in_t];
                    float w = weight[(in_c * out_channels + out_c) * kernel_size + k];
                    sum += value * w;
                }
            }
        }
    }
    output[out_c * out_frames + out_t] = sum;
}

extern "C" __global__ void codec_transconv1d_channels_tiled_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int in_frames,
    int out_frames,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride
) {
    const int oc_tile = 16;
    const int red_lanes = 16;
    int lane_oc = threadIdx.x;
    int lane_red = threadIdx.y;
    int out_t = blockIdx.x;
    int out_c = blockIdx.y * oc_tile + lane_oc;
    extern __shared__ float scratch[];
    int scratch_idx = lane_red * oc_tile + lane_oc;

    float sum = 0.0f;
    if (out_c < out_channels && out_t < out_frames) {
        for (int k = 0; k < kernel_size; k++) {
            int rem = out_t - k;
            if (rem >= 0 && rem % stride == 0) {
                int in_t = rem / stride;
                if (in_t >= 0 && in_t < in_frames) {
                    for (int in_c = lane_red; in_c < in_channels; in_c += red_lanes) {
                        float value = input[in_c * in_frames + in_t];
                        float w = weight[(in_c * out_channels + out_c) * kernel_size + k];
                        sum += value * w;
                    }
                }
            }
        }
    }
    scratch[scratch_idx] = sum;
    __syncthreads();

    for (int step = red_lanes / 2; step > 0; step >>= 1) {
        if (lane_red < step) {
            scratch[scratch_idx] += scratch[(lane_red + step) * oc_tile + lane_oc];
        }
        __syncthreads();
    }
    if (lane_red == 0 && out_c < out_channels && out_t < out_frames) {
        output[out_c * out_frames + out_t] = scratch[lane_oc] + bias[out_c];
    }
}

extern "C" __global__ void codec_transconv_prepare_tc_f32(
    const float* input,
    float* current_tc,
    float* previous_tc,
    int frames,
    int channels,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int channel = idx % channels;
    int frame = idx / channels;
    current_tc[idx] = input[channel * frames + frame];
    previous_tc[idx] = frame > 0 ? input[channel * frames + frame - 1] : 0.0f;
}

extern "C" __global__ void codec_transconv_scatter_sum_f32(
    const float* current,
    const float* previous,
    const float* bias,
    float* output,
    int frames,
    int out_channels,
    int out_frames,
    int stride,
    int phase,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int out_c = idx % out_channels;
    int frame = idx / out_channels;
    int out_t = frame * stride + phase;
    output[out_c * out_frames + out_t] = current[idx] + previous[idx] + bias[out_c];
}

extern "C" __global__ void codec_clamp_f32(
    const float* input,
    float* output,
    int total,
    float min_value,
    float max_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = fminf(fmaxf(input[idx], min_value), max_value);
    }
}
"#;

pub const ATTENTION_F32_SOURCE: &str = r#"
extern "C" __global__ void attention_scores_causal_f32(
    const float* q,
    const float* k,
    float* scores,
    int batch,
    int q_heads,
    int kv_heads,
    int query_steps,
    int key_steps,
    int head_dim,
    int offset,
    float scale,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int key_step = idx % key_steps;
    int query_step = (idx / key_steps) % query_steps;
    int q_head = (idx / (key_steps * query_steps)) % q_heads;
    int batch_index = idx / (key_steps * query_steps * q_heads);

    if (key_step > offset + query_step) {
        scores[idx] = -3.4028234663852886e38f;
        return;
    }

    int kv_head = q_head / (q_heads / kv_heads);
    int q_base = (((batch_index * q_heads + q_head) * query_steps + query_step) * head_dim);
    int k_base = (((batch_index * kv_heads + kv_head) * key_steps + key_step) * head_dim);
    float sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        sum += q[q_base + dim] * k[k_base + dim];
    }
    scores[idx] = sum * scale;
}

extern "C" __global__ void attention_apply_value_f32(
    const float* probs,
    const float* v,
    float* output,
    int batch,
    int q_heads,
    int kv_heads,
    int query_steps,
    int key_steps,
    int head_dim,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int dim = idx % head_dim;
    int q_head = (idx / head_dim) % q_heads;
    int query_step = (idx / (head_dim * q_heads)) % query_steps;
    int batch_index = idx / (head_dim * q_heads * query_steps);
    int kv_head = q_head / (q_heads / kv_heads);
    int prob_base = (((batch_index * q_heads + q_head) * query_steps + query_step) * key_steps);
    int v_base = ((batch_index * kv_heads + kv_head) * key_steps * head_dim) + dim;

    float sum = 0.0f;
    for (int key_step = 0; key_step < key_steps; ++key_step) {
        sum += probs[prob_base + key_step] * v[v_base + key_step * head_dim];
    }
    output[idx] = sum;
}

extern "C" __global__ void attention_scores_cache_f32(
    const float* q,
    const float* k,
    float* scores,
    int batch,
    int q_heads,
    int kv_heads,
    int query_steps,
    int key_steps,
    int cache_steps,
    int head_dim,
    int offset,
    float scale,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int key_step = idx % key_steps;
    int query_step = (idx / key_steps) % query_steps;
    int q_head = (idx / (key_steps * query_steps)) % q_heads;
    int batch_index = idx / (key_steps * query_steps * q_heads);

    if (key_step > offset + query_step) {
        scores[idx] = -3.4028234663852886e38f;
        return;
    }

    int kv_head = q_head / (q_heads / kv_heads);
    int q_base = (((batch_index * q_heads + q_head) * query_steps + query_step) * head_dim);
    int k_base = (((batch_index * kv_heads + kv_head) * cache_steps + key_step) * head_dim);
    float sum = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        sum += q[q_base + dim] * k[k_base + dim];
    }
    scores[idx] = sum * scale;
}

extern "C" __global__ void attention_apply_value_cache_f32(
    const float* probs,
    const float* v,
    float* output,
    int batch,
    int q_heads,
    int kv_heads,
    int query_steps,
    int key_steps,
    int cache_steps,
    int head_dim,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int dim = idx % head_dim;
    int q_head = (idx / head_dim) % q_heads;
    int query_step = (idx / (head_dim * q_heads)) % query_steps;
    int batch_index = idx / (head_dim * q_heads * query_steps);
    int kv_head = q_head / (q_heads / kv_heads);
    int prob_base = (((batch_index * q_heads + q_head) * query_steps + query_step) * key_steps);
    int v_base = ((batch_index * kv_heads + kv_head) * cache_steps * head_dim) + dim;

    float sum = 0.0f;
    for (int key_step = 0; key_step < key_steps; ++key_step) {
        sum += probs[prob_base + key_step] * v[v_base + key_step * head_dim];
    }
    output[idx] = sum;
}

extern "C" __global__ void write_kv_cache_f32(
    const float* input,
    float* cache,
    int batch,
    int heads,
    int input_steps,
    int cache_steps,
    int head_dim,
    int offset,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int dim = idx % head_dim;
    int step = (idx / head_dim) % input_steps;
    int head = (idx / (head_dim * input_steps)) % heads;
    int batch_index = idx / (head_dim * input_steps * heads);
    int cache_step = offset + step;
    int cache_idx = (((batch_index * heads + head) * cache_steps + cache_step) * head_dim) + dim;
    cache[cache_idx] = input[idx];
}
"#;
