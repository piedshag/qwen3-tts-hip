use std::fs;
use std::path::{Path, PathBuf};

use qwen3_hip_runtime::codec_hip::HipCodecInitial;
use qwen3_hip_runtime::{Error, HipRuntime, Result};

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
    let fixture_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("python-reference/out/custom_voice_0p6b_rocm_long/codec_stages_rollout39_f32")
    });

    let codes = read_codes(&fixture_dir.join("codes_bqt.npy"))?;
    let frames = codes.len() / CODE_GROUPS;
    let runtime = HipRuntime::new(0)?;
    let codec = HipCodecInitial::load(&runtime, &model_dir)?;
    let output = codec.run(&runtime, &codes, frames)?;
    runtime.synchronize()?;

    let quantized = output.quantized.copy_to_host()?;
    let pre_conv = output.pre_conv.copy_to_host()?;
    compare_f32("quantized", &quantized, &fixture_dir.join("quantized.npy"))?;
    compare_f32("pre_conv", &pre_conv, &fixture_dir.join("pre_conv.npy"))?;
    let pre_transformer = codec.run_pre_transformer(&runtime, &output.pre_conv, output.frames)?;
    runtime.synchronize()?;
    compare_f32(
        "pre_transformer",
        &pre_transformer.copy_to_host()?,
        &fixture_dir.join("pre_transformer.npy"),
    )?;
    let upsample = codec.run_upsample_stages(&runtime, &pre_transformer, output.frames)?;
    runtime.synchronize()?;
    compare_f32(
        "upsample_0_0",
        &upsample.upsample_0_0.copy_to_host()?,
        &fixture_dir.join("upsample_0_0.npy"),
    )?;
    compare_f32(
        "upsample_0_1",
        &upsample.upsample_0_1.copy_to_host()?,
        &fixture_dir.join("upsample_0_1.npy"),
    )?;
    compare_f32(
        "upsample_1_0",
        &upsample.upsample_1_0.copy_to_host()?,
        &fixture_dir.join("upsample_1_0.npy"),
    )?;
    compare_f32(
        "upsample_1_1",
        &upsample.upsample_1_1.copy_to_host()?,
        &fixture_dir.join("upsample_1_1.npy"),
    )?;
    let decoder = codec.run_decoder_stages(&runtime, &upsample.upsample_1_1, upsample.frames_1)?;
    runtime.synchronize()?;
    compare_f32(
        "decoder_0",
        &decoder.decoder_0.copy_to_host()?,
        &fixture_dir.join("decoder_0.npy"),
    )?;
    compare_f32(
        "decoder_1",
        &decoder.decoder_1.copy_to_host()?,
        &fixture_dir.join("decoder_1.npy"),
    )?;
    compare_f32(
        "decoder_2",
        &decoder.decoder_2.copy_to_host()?,
        &fixture_dir.join("decoder_2.npy"),
    )?;
    compare_f32(
        "decoder_3",
        &decoder.decoder_3.copy_to_host()?,
        &fixture_dir.join("decoder_3.npy"),
    )?;
    compare_f32(
        "decoder_4",
        &decoder.decoder_4.copy_to_host()?,
        &fixture_dir.join("decoder_4.npy"),
    )?;
    compare_f32(
        "decoder_5",
        &decoder.decoder_5.copy_to_host()?,
        &fixture_dir.join("decoder_5.npy"),
    )?;
    compare_f32(
        "decoder_6",
        &decoder.decoder_6.copy_to_host()?,
        &fixture_dir.join("decoder_6.npy"),
    )?;
    compare_f32(
        "waveform",
        &decoder.waveform.copy_to_host()?,
        &fixture_dir.join("waveform.npy"),
    )?;
    println!("HIP codec initial parity OK: frames={frames}");
    Ok(())
}

fn read_codes(path: &Path) -> Result<Vec<i32>> {
    let npy = read_npy_f32(path).map_err(|err| Error::InvalidInput(err.to_string()))?;
    if npy.shape.len() != 3 || npy.shape[0] != 1 || npy.shape[2] != CODE_GROUPS {
        return Err(Error::InvalidInput(format!(
            "expected codes shape [1, frames, {CODE_GROUPS}], got {:?}",
            npy.shape
        )));
    }
    Ok(npy.data.iter().map(|value| value.round() as i32).collect())
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
