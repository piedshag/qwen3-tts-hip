use std::path::Path;

use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::error::{Error, Result};

pub struct TensorArchive {
    bytes: Vec<u8>,
}

impl TensorArchive {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|err| Error::InvalidInput(format!("failed to read {path:?}: {err}")))?;
        Ok(Self { bytes })
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let tensors = SafeTensors::deserialize(&self.bytes)
            .map_err(|err| Error::InvalidInput(format!("failed to parse safetensors: {err}")))?;
        tensors
            .tensor(name)
            .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))
    }

    pub fn with_tensors<R>(&self, f: impl FnOnce(&SafeTensors<'_>) -> Result<R>) -> Result<R> {
        let tensors = SafeTensors::deserialize(&self.bytes)
            .map_err(|err| Error::InvalidInput(format!("failed to parse safetensors: {err}")))?;
        f(&tensors)
    }

    pub fn vector_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor(name)?;
        let shape = tensor.shape();
        if shape.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "{name} rank {}, expected 1",
                shape.len()
            )));
        }
        tensor_to_f32(name, tensor.dtype(), tensor.data(), shape[0])
    }

    pub fn linear_weight_transposed_f32(&self, name: &str) -> Result<(Vec<f32>, usize, usize)> {
        let tensor = self.tensor(name)?;
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidInput(format!(
                "{name} rank {}, expected 2",
                shape.len()
            )));
        }
        let out_dim = shape[0];
        let in_dim = shape[1];
        let data = tensor_to_f32(name, tensor.dtype(), tensor.data(), out_dim * in_dim)?;
        let mut transposed = vec![0.0; in_dim * out_dim];
        for out_idx in 0..out_dim {
            for in_idx in 0..in_dim {
                transposed[in_idx * out_dim + out_idx] = data[out_idx * in_dim + in_idx];
            }
        }
        Ok((transposed, in_dim, out_dim))
    }
}

pub fn tensor_to_f32(name: &str, dtype: Dtype, data: &[u8], len: usize) -> Result<Vec<f32>> {
    if !matches!(dtype, Dtype::F32 | Dtype::BF16) {
        return Err(Error::InvalidInput(format!(
            "{name} has dtype {dtype:?}, expected F32 or BF16"
        )));
    }
    Ok((0..len).map(|idx| read_value(dtype, data, idx)).collect())
}

pub fn read_value(dtype: Dtype, data: &[u8], idx: usize) -> f32 {
    match dtype {
        Dtype::F32 => {
            let offset = idx * 4;
            f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
        }
        Dtype::BF16 => {
            let offset = idx * 2;
            let bits = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
            f32::from_bits((bits as u32) << 16)
        }
        _ => unreachable!("dtype checked before read"),
    }
}
