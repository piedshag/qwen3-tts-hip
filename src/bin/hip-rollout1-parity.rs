use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::talker::HipTalker;
use qwen3_hip_runtime::{Error, HipRuntime};

const HIDDEN: usize = 1024;

struct NpyF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let fixture_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout4")
    });

    let prefill = read_npy_f32(&fixture_dir.join("prefill.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    if prefill.shape.len() != 3 || prefill.shape[0] != 1 || prefill.shape[2] != HIDDEN {
        return Err(Error::InvalidInput(format!(
            "expected prefill shape [1, steps, {HIDDEN}], got {:?}",
            prefill.shape
        )));
    }
    let steps = prefill.shape[1];
    let trailing = read_npy_f32(&fixture_dir.join("trailing_text.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    if trailing.data.len() < HIDDEN {
        return Err(Error::InvalidInput(
            "trailing_text is too short".to_string(),
        ));
    }
    let tts_pad = read_vector(&fixture_dir.join("tts_pad_embed.npy"), "tts_pad_embed")?;
    if tts_pad.len() != HIDDEN {
        return Err(Error::InvalidInput(format!(
            "tts_pad_embed length {} does not match hidden {HIDDEN}",
            tts_pad.len()
        )));
    }
    let rollout = read_npy_f32(&fixture_dir.join("rollout_codes.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    if rollout.shape.len() != 2 || rollout.shape[1] != 16 {
        return Err(Error::InvalidInput(format!(
            "expected rollout_codes shape [frames, 16], got {:?}",
            rollout.shape
        )));
    }
    let expected_frames = rollout
        .data
        .iter()
        .map(|value| value.round() as i32)
        .collect::<Vec<_>>();
    let frame_count = rollout.shape[0];
    let compare_frames = frame_count;
    let rollout_logits = read_npy_f32(&fixture_dir.join("rollout_logits.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    let rollout_hidden = read_npy_f32(&fixture_dir.join("rollout_past_hidden.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    let expected_first_frame = read_vector(&fixture_dir.join("first_frame.npy"), "first_frame")?
        .into_iter()
        .map(|value| value.round() as i32)
        .collect::<Vec<_>>();
    let expected_next_logits = read_vector(
        &fixture_dir.join("forward_next_logits.npy"),
        "forward_next_logits",
    )?;
    let expected_next_semantic =
        read_vector(&fixture_dir.join("next_semantic.npy"), "next_semantic")?
            .first()
            .copied()
            .unwrap_or_default()
            .round() as i32;

    let runtime = HipRuntime::new(0)?;
    let talker = HipTalker::load(&runtime, &model_dir, steps + frame_count)?;
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    let prefill_dev = runtime.buffer_from_slice(&prefill.data)?;
    let cp_prefix = runtime.empty_buffer::<f32>(2 * HIDDEN)?;
    let acoustic_sum = runtime.empty_buffer::<f32>(HIDDEN)?;

    let mut current = talker.prefill(&prefill_dev, steps)?;
    let mut generated = Vec::with_capacity(compare_frames * 16);
    let mut first_next = None;
    for frame in 0..compare_frames {
        talker.prepare_code_predictor_prefix(&cp_prefix)?;
        let acoustic = predictor.generate_to_buffer(&cp_prefix, &acoustic_sum)?;
        generated.push(current.semantic_token);
        generated.extend(acoustic);

        if frame + 1 < compare_frames {
            let trailing_offset = frame * HIDDEN;
            let trailing_dev = if trailing_offset + HIDDEN <= trailing.data.len() {
                runtime
                    .buffer_from_slice(&trailing.data[trailing_offset..trailing_offset + HIDDEN])?
            } else {
                runtime.buffer_from_slice(&tts_pad)?
            };
            talker.build_step_input(&acoustic_sum, &trailing_dev)?;
            current = talker.decode_prepared_step(steps + frame)?;
            let expected_hidden_start = frame * HIDDEN;
            let expected_logits_start = frame * 3072;
            let hidden_max_abs = max_abs(
                &current.hidden,
                &rollout_hidden.data[expected_hidden_start..expected_hidden_start + HIDDEN],
            );
            let logits_max_abs = max_abs(
                &current.logits,
                &rollout_logits.data[expected_logits_start..expected_logits_start + 3072],
            );
            if hidden_max_abs > 2e-3 || logits_max_abs > 5e-3 {
                return Err(Error::InvalidInput(format!(
                    "talker decode drift after frame {frame}: hidden_max_abs={hidden_max_abs}, logits_max_abs={logits_max_abs}"
                )));
            }
            if frame == 0 {
                first_next = Some((current.semantic_token, current.logits.clone()));
            }
        }
    }

    if generated[..16] != expected_first_frame {
        return Err(Error::InvalidInput(format!(
            "first frame mismatch: actual={:?}, expected={expected_first_frame:?}",
            &generated[..16]
        )));
    }
    let expected_prefix = &expected_frames[..compare_frames * 16];
    if generated != expected_prefix {
        return Err(Error::InvalidInput(format!(
            "rollout mismatch: actual={generated:?}, expected={expected_prefix:?}"
        )));
    }
    runtime.synchronize()?;
    let (next_semantic, next_logits) = first_next.ok_or_else(|| {
        Error::InvalidInput("rollout fixture must contain at least two frames".to_string())
    })?;
    let next_logits_max_abs = max_abs(&next_logits, &expected_next_logits);
    let next_logits_mean_abs = mean_abs(&next_logits, &expected_next_logits);
    if next_semantic != expected_next_semantic {
        return Err(Error::InvalidInput(format!(
            "next semantic mismatch: actual={next_semantic}, expected={expected_next_semantic}"
        )));
    }

    println!(
        "HIP rollout parity OK: compared_frames={compare_frames}, fixture_frames={frame_count}, first_frame={:?}, next_semantic={next_semantic}, next_logits_max_abs={next_logits_max_abs:.9}, next_logits_mean_abs={next_logits_mean_abs:.9}",
        &generated[..16]
    );
    Ok(())
}

fn read_vector(path: &Path, name: &str) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.iter().product::<usize>() != npy.data.len() {
        return Err(Error::InvalidInput(format!(
            "{name} shape {:?} does not match data length {}",
            npy.shape,
            npy.data.len()
        )));
    }
    Ok(npy.data)
}

fn read_npy_f32(path: &Path) -> Result<NpyF32, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()).into());
    }
    let major = bytes[6];
    let (header_len, data_offset) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => return Err(format!("unsupported npy version: {major}").into()),
    };
    let header_end = data_offset + header_len;
    let header = std::str::from_utf8(&bytes[data_offset..header_end])?;
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4") {
        return Err(format!("{} is not little-endian float32", path.display()).into());
    }
    let shape = parse_shape(header)?;
    let expected = shape.iter().product::<usize>();
    let data_bytes = &bytes[header_end..];
    if data_bytes.len() < expected * 4 {
        return Err(format!("{} is truncated", path.display()).into());
    }
    let data = data_bytes[..expected * 4]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(NpyF32 { shape, data })
}

fn parse_shape(header: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let shape_key = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or("missing shape in npy header")?;
    let tuple_start = header[shape_key..]
        .find('(')
        .map(|offset| shape_key + offset + 1)
        .ok_or("missing shape tuple start")?;
    let tuple_end = header[tuple_start..]
        .find(')')
        .map(|offset| tuple_start + offset)
        .ok_or("missing shape tuple end")?;
    let shape = header[tuple_start..tuple_end]
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        })
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(shape)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / a.len() as f32
}
