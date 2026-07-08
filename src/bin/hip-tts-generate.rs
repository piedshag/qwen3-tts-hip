use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3_hip_runtime::codec::write_wav;
use qwen3_hip_runtime::generation::{
    DEFAULT_TEXT_LOOKAHEAD_TOKENS, GenerateOptions, HipTtsEngine, Language, Speaker,
    VoiceClonePrompt,
};
use qwen3_hip_runtime::{Error, Result};

const CODE_GROUPS: usize = 16;

struct NpyF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
    let max_frames = parse_arg(args.next(), "max_frames")?.unwrap_or(12);
    let reference_codes = args.next().and_then(|value| {
        let text = value.to_string_lossy();
        (!matches!(text.as_ref(), "none" | "-")).then(|| PathBuf::from(value))
    });
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
    let output_wav = args.next().and_then(|value| {
        let text = value.to_string_lossy();
        (!matches!(text.as_ref(), "none" | "-")).then(|| PathBuf::from(value))
    });
    let wav_gain = parse_f32_arg(args.next(), "wav_gain")?.unwrap_or(1.0);
    let repetition_penalty = parse_f32_arg(args.next(), "repetition_penalty")?.unwrap_or(1.05);
    let do_sample = parse_bool_arg(args.next(), "do_sample")?.unwrap_or(true);
    let top_k = parse_arg(args.next(), "top_k")?.unwrap_or(50);
    let top_p = parse_f32_arg(args.next(), "top_p")?.unwrap_or(1.0);
    let temperature = parse_f32_arg(args.next(), "temperature")?.unwrap_or(0.9);
    let subtalker_dosample = parse_bool_arg(args.next(), "subtalker_dosample")?.unwrap_or(true);
    let subtalker_top_k = parse_arg(args.next(), "subtalker_top_k")?.unwrap_or(50);
    let subtalker_top_p = parse_f32_arg(args.next(), "subtalker_top_p")?.unwrap_or(1.0);
    let subtalker_temperature = parse_f32_arg(args.next(), "subtalker_temperature")?.unwrap_or(0.9);
    let seed = parse_u64_arg(args.next(), "seed")?.unwrap_or(0);
    let text_lookahead_tokens =
        parse_arg(args.next(), "text_lookahead_tokens")?.unwrap_or(DEFAULT_TEXT_LOOKAHEAD_TOKENS);
    let voice_clone_prompt_path = args.next().and_then(|value| {
        let text = value.to_string_lossy();
        (!matches!(text.as_ref(), "none" | "-")).then(|| PathBuf::from(value))
    });
    if max_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames must be non-zero".to_string(),
        ));
    }
    if text_lookahead_tokens == 0 {
        return Err(Error::InvalidInput(
            "text_lookahead_tokens must be non-zero".to_string(),
        ));
    }

    let load_start = Instant::now();
    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_frames)?;
    engine.runtime().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let generation_start = Instant::now();
    let options = GenerateOptions {
        speaker,
        language,
        max_frames,
        decode_audio: false,
        do_sample,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        subtalker_dosample,
        subtalker_top_k,
        subtalker_top_p,
        subtalker_temperature,
        seed,
        text_lookahead_tokens,
    };
    let voice_clone_prompt = voice_clone_prompt_path
        .as_deref()
        .map(VoiceClonePrompt::from_json)
        .transpose()?;
    let generated = if let Some(prompt) = voice_clone_prompt.as_ref() {
        engine.generate_voice_clone_codes(&text, prompt, options)?
    } else {
        engine.generate_codes(&text, options)?
    };
    let generation_seconds = generation_start.elapsed().as_secs_f64();

    if let Some(path) = reference_codes.as_deref() {
        let expected = load_expected_codes(path)?;
        if expected.len() < generated.codes.len() {
            return Err(Error::InvalidInput(format!(
                "reference has {} codes but generated {}",
                expected.len(),
                generated.codes.len()
            )));
        }
        if generated.codes != expected[..generated.codes.len()] {
            return Err(Error::InvalidInput(format!(
                "generated codes mismatch: actual={:?}, expected={:?}",
                generated.codes,
                &expected[..generated.codes.len()]
            )));
        }
    }

    let mut samples = Vec::new();
    let mut decode_seconds = None;
    let mut write_seconds = None;
    if let Some(path) = output_wav.as_deref() {
        let decode_start = Instant::now();
        samples = engine.decode_codes(&generated.codes)?;
        decode_seconds = Some(decode_start.elapsed().as_secs_f64());

        let write_start = Instant::now();
        write_wav(
            path,
            &samples,
            qwen3_hip_runtime::generation::SAMPLE_RATE,
            wav_gain,
        )?;
        write_seconds = Some(write_start.elapsed().as_secs_f64());
    }

    let audio_seconds = (!samples.is_empty())
        .then_some(samples.len() as f64 / qwen3_hip_runtime::generation::SAMPLE_RATE as f64);
    let inference_seconds = generation_seconds + decode_seconds.unwrap_or(0.0);
    println!(
        "HIP {} generate OK: text={text:?}, speaker={speaker:?}, language={language:?}, frames={}, ended_by_eos={}, samples={}, output_wav={:?}, load_seconds={load_seconds:.6}, generation_seconds={generation_seconds:.6}, decode_seconds={}, write_seconds={}, inference_seconds={inference_seconds:.6}, audio_seconds={}, generation_rtf={}, decode_rtf={}, inference_rtf={}, first_frame={:?}, last_frame={:?}",
        if voice_clone_prompt.is_some() {
            "voice clone"
        } else {
            "custom voice"
        },
        generated.frames,
        generated.ended_by_eos,
        samples.len(),
        output_wav,
        format_optional_seconds(decode_seconds),
        format_optional_seconds(write_seconds),
        format_optional_seconds(audio_seconds),
        format_optional_rtf(generation_seconds, audio_seconds),
        format_optional_rtf(
            decode_seconds.unwrap_or(0.0),
            audio_seconds.filter(|_| decode_seconds.is_some())
        ),
        format_optional_rtf(inference_seconds, audio_seconds),
        &generated.codes[..CODE_GROUPS.min(generated.codes.len())],
        &generated.codes[generated.codes.len().saturating_sub(CODE_GROUPS)..]
    );
    Ok(())
}

fn format_optional_seconds(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_rtf(seconds: f64, audio_seconds: Option<f64>) -> String {
    audio_seconds
        .filter(|audio_seconds| *audio_seconds > 0.0)
        .map(|audio_seconds| format!("{:.6}", seconds / audio_seconds))
        .unwrap_or_else(|| "n/a".to_string())
}

fn load_expected_codes(path: &Path) -> Result<Vec<i32>> {
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.len() != 2 || npy.shape[1] != CODE_GROUPS {
        return Err(Error::InvalidInput(format!(
            "expected rollout_codes shape [frames, {CODE_GROUPS}], got {:?}",
            npy.shape
        )));
    }
    Ok(npy.data.iter().map(|value| value.round() as i32).collect())
}

fn parse_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<usize>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}

fn parse_f32_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<f32>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<f32>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}

fn parse_u64_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<u64>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<u64>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}

fn parse_bool_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<bool>> {
    value
        .map(
            |value| match value.to_string_lossy().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                other => Err(Error::InvalidInput(format!("invalid {name}: {other}"))),
            },
        )
        .transpose()
}

fn read_npy_f32(path: &Path) -> std::result::Result<NpyF32, Box<dyn std::error::Error>> {
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

fn parse_shape(header: &str) -> std::result::Result<Vec<usize>, Box<dyn std::error::Error>> {
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
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(shape)
}
