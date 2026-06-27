use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::generation::{GenerateOptions, HipTtsEngine, Language, Speaker};
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
    let output_wav = args.next().map(PathBuf::from);
    let wav_gain = parse_f32_arg(args.next(), "wav_gain")?.unwrap_or(1.0);
    if max_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames must be non-zero".to_string(),
        ));
    }

    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_frames)?;
    let speech = engine.generate(
        &text,
        GenerateOptions {
            speaker,
            language,
            max_frames,
            decode_audio: output_wav.is_some(),
        },
    )?;

    if let Some(path) = reference_codes.as_deref() {
        let expected = load_expected_codes(path)?;
        if expected.len() < speech.codes.len() {
            return Err(Error::InvalidInput(format!(
                "reference has {} codes but generated {}",
                expected.len(),
                speech.codes.len()
            )));
        }
        if speech.codes != expected[..speech.codes.len()] {
            return Err(Error::InvalidInput(format!(
                "generated codes mismatch: actual={:?}, expected={:?}",
                speech.codes,
                &expected[..speech.codes.len()]
            )));
        }
    }

    if let Some(path) = output_wav.as_deref() {
        speech.write_wav(path, wav_gain)?;
    }

    println!(
        "HIP custom voice generate OK: text={text:?}, speaker={speaker:?}, language={language:?}, frames={}, ended_by_eos={}, samples={}, output_wav={:?}, first_frame={:?}, last_frame={:?}",
        speech.frames,
        speech.ended_by_eos,
        speech.samples.len(),
        output_wav,
        &speech.codes[..CODE_GROUPS.min(speech.codes.len())],
        &speech.codes[speech.codes.len().saturating_sub(CODE_GROUPS)..]
    );
    Ok(())
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
