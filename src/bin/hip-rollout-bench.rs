use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::talker::HipTalker;
use qwen3_hip_runtime::{DeviceBuffer, Error, HipRuntime};

const HIDDEN: usize = 1024;
const CODE_GROUPS: usize = 16;

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
        PathBuf::from("python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12")
    });
    let frames_arg = parse_arg(args.next(), "frames")?;
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(20);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(3);
    if iterations == 0 {
        return Err(Error::InvalidInput(
            "iterations must be non-zero".to_string(),
        ));
    }

    let prefill = read_npy_f32(&fixture_dir.join("prefill.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    if prefill.shape.len() != 3 || prefill.shape[0] != 1 || prefill.shape[2] != HIDDEN {
        return Err(Error::InvalidInput(format!(
            "expected prefill shape [1, steps, {HIDDEN}], got {:?}",
            prefill.shape
        )));
    }
    let prefill_steps = prefill.shape[1];
    let trailing = read_npy_f32(&fixture_dir.join("trailing_text.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    let tts_pad = read_vector(&fixture_dir.join("tts_pad_embed.npy"), "tts_pad_embed")?;
    if tts_pad.len() != HIDDEN {
        return Err(Error::InvalidInput(format!(
            "tts_pad_embed length {} does not match hidden {HIDDEN}",
            tts_pad.len()
        )));
    }
    let expected = load_expected_codes(&fixture_dir.join("rollout_codes.npy"))?;
    let frames = frames_arg.unwrap_or_else(|| {
        expected
            .as_ref()
            .map(|codes| codes.len() / CODE_GROUPS)
            .unwrap_or(12)
    });
    if frames == 0 {
        return Err(Error::InvalidInput("frames must be non-zero".to_string()));
    }

    let runtime = HipRuntime::new(0)?;
    let load_start = Instant::now();
    let talker = HipTalker::load(&runtime, &model_dir, prefill_steps + frames)?;
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    let prefill_dev = runtime.buffer_from_slice(&prefill.data)?;
    let cp_prefix = runtime.empty_buffer::<f32>(2 * HIDDEN)?;
    let acoustic_sum = runtime.empty_buffer::<f32>(HIDDEN)?;
    let trailing_devs =
        upload_trailing(&runtime, &trailing.data, &tts_pad, frames.saturating_sub(1))?;
    runtime.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let first = rollout(
        &talker,
        &predictor,
        &prefill_dev,
        &cp_prefix,
        &acoustic_sum,
        &trailing_devs,
        prefill_steps,
        frames,
    )?;
    runtime.synchronize()?;
    if let Some(expected) = expected
        .as_ref()
        .filter(|codes| codes.len() >= frames * CODE_GROUPS)
    {
        let expected = &expected[..frames * CODE_GROUPS];
        if first != expected {
            return Err(Error::InvalidInput(format!(
                "rollout mismatch: actual={first:?}, expected={expected:?}"
            )));
        }
    }

    for _ in 0..warmup {
        let _ = rollout(
            &talker,
            &predictor,
            &prefill_dev,
            &cp_prefix,
            &acoustic_sum,
            &trailing_devs,
            prefill_steps,
            frames,
        )?;
    }
    runtime.synchronize()?;

    let start = Instant::now();
    let mut last = first.clone();
    for _ in 0..iterations {
        last = rollout(
            &talker,
            &predictor,
            &prefill_dev,
            &cp_prefix,
            &acoustic_sum,
            &trailing_devs,
            prefill_steps,
            frames,
        )?;
    }
    runtime.synchronize()?;
    let total_seconds = start.elapsed().as_secs_f64();
    let mean_seconds = total_seconds / iterations as f64;
    let audio_seconds = frames as f64 / 12.0;
    let generation_rtf = mean_seconds / audio_seconds;

    println!(
        "HIP rollout bench: frames={frames}, iterations={iterations}, warmup={warmup}, prefill_steps={prefill_steps}, load_seconds={load_seconds:.6}, total_seconds={total_seconds:.6}, mean_seconds={mean_seconds:.6}, audio_seconds={audio_seconds:.6}, generation_rtf={generation_rtf:.6}, first_frame={:?}, last_frame={:?}",
        &last[..CODE_GROUPS],
        &last[(frames - 1) * CODE_GROUPS..frames * CODE_GROUPS]
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rollout(
    talker: &HipTalker,
    predictor: &HipCodePredictor,
    prefill: &DeviceBuffer<f32>,
    cp_prefix: &DeviceBuffer<f32>,
    acoustic_sum: &DeviceBuffer<f32>,
    trailing: &[DeviceBuffer<f32>],
    prefill_steps: usize,
    frames: usize,
) -> qwen3_hip_runtime::Result<Vec<i32>> {
    let mut semantic = talker.prefill_token(prefill, prefill_steps)?;
    let mut generated = Vec::with_capacity(frames * CODE_GROUPS);
    for frame in 0..frames {
        talker.prepare_code_predictor_prefix(cp_prefix)?;
        let acoustic = predictor.generate_to_buffer(cp_prefix, acoustic_sum)?;
        generated.push(semantic);
        generated.extend(acoustic);
        if frame + 1 < frames {
            talker.build_step_input(acoustic_sum, &trailing[frame])?;
            semantic = talker.decode_prepared_token(prefill_steps + frame)?;
        }
    }
    Ok(generated)
}

fn upload_trailing(
    runtime: &HipRuntime,
    trailing: &[f32],
    tts_pad: &[f32],
    frames: usize,
) -> qwen3_hip_runtime::Result<Vec<DeviceBuffer<f32>>> {
    let mut buffers = Vec::with_capacity(frames);
    for frame in 0..frames {
        let offset = frame * HIDDEN;
        if offset + HIDDEN <= trailing.len() {
            buffers.push(runtime.buffer_from_slice(&trailing[offset..offset + HIDDEN])?);
        } else {
            buffers.push(runtime.buffer_from_slice(tts_pad)?);
        }
    }
    Ok(buffers)
}

fn load_expected_codes(path: &Path) -> qwen3_hip_runtime::Result<Option<Vec<i32>>> {
    if !path.exists() {
        return Ok(None);
    }
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.len() != 2 || npy.shape[1] != CODE_GROUPS {
        return Err(Error::InvalidInput(format!(
            "expected rollout_codes shape [frames, {CODE_GROUPS}], got {:?}",
            npy.shape
        )));
    }
    Ok(Some(
        npy.data.iter().map(|value| value.round() as i32).collect(),
    ))
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
