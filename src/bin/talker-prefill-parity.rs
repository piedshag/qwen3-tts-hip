use std::fs;
use std::path::{Path, PathBuf};

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

    let prefill_npy = read_npy_f32(&fixture_dir.join("prefill.npy"))
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    if prefill_npy.shape.len() != 3 || prefill_npy.shape[0] != 1 || prefill_npy.shape[2] != HIDDEN {
        return Err(Error::InvalidInput(format!(
            "expected prefill shape [1, steps, {HIDDEN}], got {:?}",
            prefill_npy.shape
        )));
    }
    let steps = prefill_npy.shape[1];
    let expected_hidden = read_vector(&fixture_dir.join("last_hidden.npy"), "last_hidden")?;
    let expected_logits = read_vector(&fixture_dir.join("logits.npy"), "logits")?;
    let expected_semantic = read_vector(&fixture_dir.join("first_semantic.npy"), "first_semantic")?
        .first()
        .copied()
        .unwrap_or_default()
        .round() as i32;

    let runtime = HipRuntime::new(0)?;
    let talker = HipTalker::load(&runtime, &model_dir, steps + 1)?;
    let actual = talker.prefill_from_host(&runtime, &prefill_npy.data, steps)?;
    runtime.synchronize()?;

    let hidden_max_abs = max_abs(&actual.hidden, &expected_hidden);
    let hidden_mean_abs = mean_abs(&actual.hidden, &expected_hidden);
    let logits_max_abs = max_abs(&actual.logits, &expected_logits);
    let logits_mean_abs = mean_abs(&actual.logits, &expected_logits);
    if actual.semantic_token != expected_semantic {
        return Err(Error::InvalidInput(format!(
            "talker semantic mismatch: actual={}, expected={expected_semantic}",
            actual.semantic_token
        )));
    }
    if hidden_max_abs > 2e-3
        || hidden_mean_abs > 2e-4
        || logits_max_abs > 5e-3
        || logits_mean_abs > 5e-4
    {
        return Err(Error::InvalidInput(format!(
            "talker prefill mismatch: hidden_max_abs={hidden_max_abs}, hidden_mean_abs={hidden_mean_abs}, logits_max_abs={logits_max_abs}, logits_mean_abs={logits_mean_abs}"
        )));
    }

    println!(
        "Talker prefill parity OK: steps={steps}, semantic={}, hidden_max_abs={hidden_max_abs:.9}, hidden_mean_abs={hidden_mean_abs:.9}, logits_max_abs={logits_max_abs:.9}, logits_mean_abs={logits_mean_abs:.9}, logits_first8={:?}",
        actual.semantic_token,
        &actual.logits[..8]
    );

    let step_input_path = fixture_dir.join("first_step_input.npy");
    if step_input_path.exists() {
        let step_input = read_vector(&step_input_path, "first_step_input")?;
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
        let actual_next = talker.decode_step_from_host(&runtime, &step_input, steps)?;
        runtime.synchronize()?;
        let next_logits_max_abs = max_abs(&actual_next.logits, &expected_next_logits);
        let next_logits_mean_abs = mean_abs(&actual_next.logits, &expected_next_logits);
        if actual_next.semantic_token != expected_next_semantic {
            return Err(Error::InvalidInput(format!(
                "talker next semantic mismatch: actual={}, expected={expected_next_semantic}",
                actual_next.semantic_token
            )));
        }
        if next_logits_max_abs > 5e-3 || next_logits_mean_abs > 5e-4 {
            return Err(Error::InvalidInput(format!(
                "talker decode-step logits mismatch: max_abs={next_logits_max_abs}, mean_abs={next_logits_mean_abs}"
            )));
        }
        println!(
            "Talker decode-step parity OK: offset={steps}, semantic={}, logits_max_abs={next_logits_max_abs:.9}, logits_mean_abs={next_logits_mean_abs:.9}, logits_first8={:?}",
            actual_next.semantic_token,
            &actual_next.logits[..8]
        );
    }
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
