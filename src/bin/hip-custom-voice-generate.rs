use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::codec::write_wav;
use qwen3_hip_runtime::codec_hip::HipCodecInitial;
use qwen3_hip_runtime::talker::HipTalker;
use qwen3_hip_runtime::text::{CustomVoiceTextPrep, Language, Speaker};
use qwen3_hip_runtime::{DeviceBuffer, Error, HipRuntime};

const HIDDEN: usize = 1024;
const CODE_GROUPS: usize = 16;
const CODEC_EOS_TOKEN: i32 = 2150;

struct NpyF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
    let max_frames = parse_arg(args.next(), "max_frames")?.unwrap_or(12);
    let reference_codes = args.next().map(PathBuf::from);
    let speaker = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Speaker>())
        .transpose()?
        .unwrap_or(Speaker::Ryan);
    let language = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Language>())
        .transpose()?
        .unwrap_or(Language::English);
    let output_wav = args.next().map(PathBuf::from);
    let wav_gain = parse_f32_arg(args.next(), "wav_gain")?.unwrap_or(1.0);
    if max_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames must be non-zero".to_string(),
        ));
    }

    let prep = CustomVoiceTextPrep::load(&model_dir)?;
    let inputs = prep.prepare_custom_voice(&text, speaker, language)?;
    let runtime = HipRuntime::new(0)?;
    let talker = HipTalker::load(&runtime, &model_dir, inputs.prefill_steps + max_frames)?;
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    let prefill = runtime.buffer_from_slice(&inputs.prefill)?;
    let trailing = upload_trailing(
        &runtime,
        &inputs.trailing_text,
        &inputs.tts_pad_embed,
        max_frames.saturating_sub(1),
    )?;
    let cp_prefix = runtime.empty_buffer::<f32>(2 * HIDDEN)?;
    let acoustic_sum = runtime.empty_buffer::<f32>(HIDDEN)?;

    let frames = rollout(
        &talker,
        &predictor,
        &prefill,
        &cp_prefix,
        &acoustic_sum,
        &trailing,
        inputs.prefill_steps,
        max_frames,
    )?;
    runtime.synchronize()?;

    if let Some(path) = reference_codes.as_deref() {
        let expected = load_expected_codes(path)?;
        if expected.len() < frames.len() {
            return Err(Error::InvalidInput(format!(
                "reference has {} codes but generated {}",
                expected.len(),
                frames.len()
            )));
        }
        if frames != expected[..frames.len()] {
            return Err(Error::InvalidInput(format!(
                "generated codes mismatch: actual={frames:?}, expected={:?}",
                &expected[..frames.len()]
            )));
        }
    }

    if let Some(path) = output_wav.as_deref() {
        let decoder = HipCodecInitial::load(&runtime, &model_dir)?;
        let waveform = decode_waveform(&runtime, &decoder, &frames)?;
        write_wav(path, &waveform, 24_000, wav_gain)?;
    }

    println!(
        "HIP custom voice generate OK: text={text:?}, speaker={speaker:?}, language={language:?}, input_tokens={}, content_tokens={}, prefill_steps={}, trailing_steps={}, frames={}, output_wav={:?}, first_frame={:?}, last_frame={:?}",
        inputs.input_ids.len(),
        inputs.content_ids.len(),
        inputs.prefill_steps,
        inputs.trailing_steps,
        frames.len() / CODE_GROUPS,
        output_wav,
        &frames[..CODE_GROUPS.min(frames.len())],
        &frames[frames.len().saturating_sub(CODE_GROUPS)..]
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
    max_frames: usize,
) -> qwen3_hip_runtime::Result<Vec<i32>> {
    let mut semantic = talker.prefill_token(prefill, prefill_steps)?;
    let mut generated = Vec::with_capacity(max_frames * CODE_GROUPS);
    for frame in 0..max_frames {
        if semantic == CODEC_EOS_TOKEN {
            break;
        }
        talker.prepare_code_predictor_prefix(cp_prefix)?;
        let acoustic = predictor.generate_to_buffer(cp_prefix, acoustic_sum)?;
        generated.push(semantic);
        generated.extend(acoustic);
        if frame + 1 < max_frames {
            talker.build_step_input(acoustic_sum, &trailing[frame])?;
            semantic = talker.decode_prepared_token(prefill_steps + frame)?;
        }
    }
    Ok(generated)
}

fn decode_waveform(
    runtime: &HipRuntime,
    decoder: &HipCodecInitial,
    codes: &[i32],
) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let frames = codes.len() / CODE_GROUPS;
    let initial = decoder.run(runtime, codes, frames)?;
    let pre_transformer =
        decoder.run_pre_transformer(runtime, &initial.pre_conv, initial.frames)?;
    let upsample = decoder.run_upsample_stages(runtime, &pre_transformer, initial.frames)?;
    let output = decoder.run_decoder_stages(runtime, &upsample.upsample_1_1, upsample.frames_1)?;
    runtime.synchronize()?;
    output.waveform.copy_to_host()
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

fn load_expected_codes(path: &Path) -> qwen3_hip_runtime::Result<Vec<i32>> {
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.len() != 2 || npy.shape[1] != CODE_GROUPS {
        return Err(Error::InvalidInput(format!(
            "expected rollout_codes shape [frames, {CODE_GROUPS}], got {:?}",
            npy.shape
        )));
    }
    Ok(npy.data.iter().map(|value| value.round() as i32).collect())
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

fn parse_f32_arg(
    value: Option<std::ffi::OsString>,
    name: &str,
) -> qwen3_hip_runtime::Result<Option<f32>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<f32>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
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
