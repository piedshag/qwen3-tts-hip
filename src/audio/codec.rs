use std::path::Path;

use crate::error::{Error, Result};

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, gain: f32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|err| {
        Error::InvalidInput(format!("failed to create {}: {err}", path.display()))
    })?;
    for &sample in samples {
        let value = (sample * gain).clamp(-1.0, 1.0);
        writer
            .write_sample((value * i16::MAX as f32) as i16)
            .map_err(|err| Error::InvalidInput(format!("failed to write wav sample: {err}")))?;
    }
    writer.finalize().map_err(|err| {
        Error::InvalidInput(format!("failed to finalize {}: {err}", path.display()))
    })?;
    Ok(())
}
