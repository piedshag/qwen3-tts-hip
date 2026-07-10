pub mod audio;
pub mod error;
pub mod generation;
pub mod gpu;
pub mod model;

pub mod blas {
    pub use crate::gpu::blas::*;
}

pub mod buffer {
    pub use crate::gpu::buffer::*;
}

pub mod code_predictor {
    pub use crate::model::code_predictor::*;
}

pub mod codec {
    pub use crate::audio::codec::*;
}

pub mod codec_hip {
    pub use crate::audio::codec_hip::*;
}

pub mod config {
    pub use crate::model::config::*;
}

pub mod decode {
    pub use crate::model::decode::*;
}

pub(crate) mod ffi {
    pub(crate) use crate::gpu::ffi::*;
}

pub mod graph {
    pub use crate::gpu::graph::*;
}

pub mod kernel {
    pub use crate::gpu::kernel::*;
}

pub mod profile {
    pub use crate::gpu::profile::*;
}

pub mod kernels {
    pub use crate::gpu::kernels::*;
}

pub mod runtime {
    pub use crate::gpu::runtime::*;
}

pub mod stack {
    pub use crate::model::stack::*;
}

pub mod talker {
    pub use crate::model::talker::*;
}

pub mod text {
    pub use crate::model::text::*;
}

pub mod voice_clone {
    pub use crate::model::voice_clone::*;
}

pub mod weights {
    pub use crate::model::weights::*;
}

pub use blas::RocblasHandle;
pub use buffer::DeviceBuffer;
pub use error::{Error, Result};
pub use generation::{
    EngineOptions, GenerateOptions, GeneratedAudioChunk, GeneratedCodes, GeneratedCodesChunk,
    GeneratedSpeech, GenerationProfile, HipTextStream, HipTtsEngine, HipTtsStream,
    IncrementalAudio, Language, PollingTextStream, ProfiledGeneratedCodes, Speaker, StreamOptions,
    TextStreamInput, TextStreamOptions, VoiceClonePrompt,
};
pub use graph::{HipGraph, HipGraphExec, HipStream};
pub use kernel::{HipFunction, HipModule};
pub use runtime::HipRuntime;
