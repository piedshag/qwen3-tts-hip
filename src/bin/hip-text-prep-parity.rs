use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::text::{Language, Speaker, TtsTextPrep};
use qwen3_hip_runtime::{Error, Result};

struct NpyF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let fixture_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
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

    let prep = TtsTextPrep::load(&model_dir)?;
    let inputs = prep.prepare_custom_voice(&text, speaker, language)?;

    compare_ids(
        "input_ids",
        &inputs.input_ids,
        &fixture_dir.join("input_ids.npy"),
    )?;
    compare_ids(
        "content_ids",
        &inputs.content_ids,
        &fixture_dir.join("content_ids.npy"),
    )?;
    compare_f32("prefill", &inputs.prefill, &fixture_dir.join("prefill.npy"))?;
    compare_f32(
        "trailing_text",
        &inputs.trailing_text,
        &fixture_dir.join("trailing_text.npy"),
    )?;
    compare_f32(
        "tts_pad_embed",
        &inputs.tts_pad_embed,
        &fixture_dir.join("tts_pad_embed.npy"),
    )?;
    Ok(())
}

fn compare_ids(name: &str, actual: &[u32], path: &Path) -> Result<()> {
    let expected = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    let expected = expected
        .data
        .iter()
        .map(|value| value.round() as u32)
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(Error::InvalidInput(format!(
            "{name} mismatch: actual={actual:?}, expected={expected:?}"
        )));
    }
    println!("{name}: exact, len={}", actual.len());
    Ok(())
}

fn compare_f32(name: &str, actual: &[f32], path: &Path) -> Result<()> {
    let expected = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if actual.len() != expected.data.len() {
        return Err(Error::InvalidInput(format!(
            "{name} length mismatch: actual={}, expected={}, expected_shape={:?}",
            actual.len(),
            expected.data.len(),
            expected.shape
        )));
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_idx = 0usize;
    for (idx, (&a, &b)) in actual.iter().zip(&expected.data).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = idx;
        }
        sum_abs += diff as f64;
    }
    println!(
        "{name}: shape={:?}, max_abs={max_abs:.9}, mean_abs={:.9}, max_idx={max_idx}, actual={:.9}, expected={:.9}",
        expected.shape,
        sum_abs / actual.len() as f64,
        actual[max_idx],
        expected.data[max_idx]
    );
    Ok(())
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
