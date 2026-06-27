use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load ROCm library {library:?}: {source}")]
    LibraryLoad {
        library: PathBuf,
        #[source]
        source: libloading::Error,
    },

    #[error("failed to resolve symbol {symbol} from {library}: {source}")]
    SymbolLoad {
        library: &'static str,
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },

    #[error("could not find ROCm library matching {name}")]
    LibraryNotFound { name: &'static str },

    #[error("HIP call {call} failed with status {status}: {message}")]
    Hip {
        call: &'static str,
        status: i32,
        message: String,
    },

    #[error("rocBLAS call {call} failed with status {status}")]
    Rocblas { call: &'static str, status: i32 },

    #[error("HIPRTC call {call} failed with status {status}: {log}")]
    Hiprtc {
        call: &'static str,
        status: i32,
        log: String,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, Error>;
