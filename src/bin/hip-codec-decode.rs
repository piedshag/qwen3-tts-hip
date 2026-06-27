use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::codec::{decode_codes_to_waveform, write_wav};
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
    let codes_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::InvalidInput("missing codes.npy path".to_string()))?;
    let reference_waveform = args.next().map(PathBuf::from);
    let output_wav = args.next().map(PathBuf::from);
    let wav_gain = parse_f32_arg(args.next(), "wav_gain")?.unwrap_or(1.0);

    let codes = read_codes(&codes_path)?;
    let waveform = decode_codes_to_waveform(&model_dir, &codes)?;
    if let Some(reference) = reference_waveform.as_deref() {
        compare_waveform(&waveform, reference)?;
    }
    if let Some(path) = output_wav.as_deref() {
        write_wav(path, &waveform, 24_000, wav_gain)?;
    }
    println!(
        "HIP codec decode OK: frames={}, samples={}, reference={:?}, output_wav={:?}",
        codes.len(),
        waveform.len(),
        reference_waveform,
        output_wav
    );
    Ok(())
}

fn read_codes(path: &Path) -> Result<Vec<Vec<u32>>> {
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.len() != 2 || npy.shape[1] != CODE_GROUPS {
        return Err(Error::InvalidInput(format!(
            "expected codes shape [frames, {CODE_GROUPS}], got {:?}",
            npy.shape
        )));
    }
    Ok(npy
        .data
        .chunks_exact(CODE_GROUPS)
        .map(|frame| {
            frame
                .iter()
                .map(|value| value.round().max(0.0) as u32)
                .collect()
        })
        .collect())
}

fn compare_waveform(actual: &[f32], path: &Path) -> Result<()> {
    let expected = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if actual.len() != expected.data.len() {
        return Err(Error::InvalidInput(format!(
            "waveform length mismatch: actual={}, expected={}, expected_shape={:?}",
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
        "waveform parity: shape={:?}, max_abs={max_abs:.9}, mean_abs={:.9}, max_idx={max_idx}, actual={:.9}, expected={:.9}",
        expected.shape,
        sum_abs / actual.len() as f64,
        actual[max_idx],
        expected.data[max_idx]
    );
    Ok(())
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
