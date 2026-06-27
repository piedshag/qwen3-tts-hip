use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::{Error, HipRuntime};

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
        PathBuf::from("python-reference/out/custom_voice_0p6b_rocm_long/code_predictor_f32")
    });

    let talker_hidden = read_vector(&fixture_dir.join("talker_hidden.npy"), "talker_hidden")?;
    let semantic_embed = read_vector(&fixture_dir.join("semantic_embed.npy"), "semantic_embed")?;
    let expected_tokens = read_vector(&fixture_dir.join("acoustic_tokens.npy"), "acoustic_tokens")?
        .into_iter()
        .map(|value| value.round() as i32)
        .collect::<Vec<_>>();
    let expected_embedding_sum =
        read_vector(&fixture_dir.join("embedding_sum.npy"), "embedding_sum")?;

    let runtime = HipRuntime::new(0)?;
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    let actual = predictor.generate_from_host(&talker_hidden, &semantic_embed)?;
    runtime.synchronize()?;

    if actual.acoustic_tokens != expected_tokens {
        return Err(Error::InvalidInput(format!(
            "CodePredictor token mismatch: actual={:?}, expected={expected_tokens:?}",
            actual.acoustic_tokens
        )));
    }
    let max_abs = max_abs(&actual.embedding_sum, &expected_embedding_sum);
    let mean_abs = mean_abs(&actual.embedding_sum, &expected_embedding_sum);
    if max_abs > 2e-3 || mean_abs > 2e-4 {
        return Err(Error::InvalidInput(format!(
            "CodePredictor embedding_sum mismatch: max_abs={max_abs}, mean_abs={mean_abs}"
        )));
    }

    println!(
        "CodePredictor parity OK: tokens={:?}, embedding_sum_max_abs={max_abs:.9}, embedding_sum_mean_abs={mean_abs:.9}, first8={:?}",
        actual.acoustic_tokens,
        &actual.embedding_sum[..8]
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
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        2 | 3 => {
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => return Err(format!("unsupported npy version: {major}").into()),
    };
    let header_end = data_offset + header_len;
    let header = std::str::from_utf8(&bytes[data_offset..header_end])?;
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4") {
        return Err(format!("{} is not little-endian float32", path.display()).into());
    }
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true") {
        return Err(format!("{} uses Fortran order", path.display()).into());
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
