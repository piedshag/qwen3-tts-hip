use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::Library;

use crate::error::{Error, Result};

pub(crate) const HIP_SUCCESS: i32 = 0;
pub(crate) const HIP_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub(crate) const HIP_MEMCPY_DEVICE_TO_HOST: u32 = 2;
pub(crate) const HIP_MEMCPY_DEVICE_TO_DEVICE: u32 = 3;
pub(crate) const ROCBLAS_STATUS_SUCCESS: i32 = 0;
pub(crate) const ROCBLAS_OPERATION_NONE: i32 = 111;
pub(crate) const HIPRTC_SUCCESS: i32 = 0;

pub(crate) type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
pub(crate) type HipSetDevice = unsafe extern "C" fn(c_int) -> c_int;
pub(crate) type HipGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
pub(crate) type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
pub(crate) type HipFree = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32) -> c_int;
pub(crate) type HipMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, u32, *mut c_void) -> c_int;
pub(crate) type HipDeviceSynchronize = unsafe extern "C" fn() -> c_int;
pub(crate) type HipStreamCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
pub(crate) type HipStreamDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipStreamBeginCapture = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
pub(crate) type HipStreamEndCapture = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
pub(crate) type HipGraphInstantiate = unsafe extern "C" fn(
    *mut *mut c_void,
    *mut c_void,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
pub(crate) type HipGraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
pub(crate) type HipGraphExecDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipGraphDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
pub(crate) type HipModuleUnload = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type HipModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
pub(crate) type HipModuleLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> c_int;

pub(crate) type HiprtcProgram = *mut c_void;
pub(crate) type HiprtcCreateProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
pub(crate) type HiprtcCompileProgram =
    unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> c_int;
pub(crate) type HiprtcDestroyProgram = unsafe extern "C" fn(*mut HiprtcProgram) -> c_int;
pub(crate) type HiprtcGetProgramLogSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
pub(crate) type HiprtcGetProgramLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;
pub(crate) type HiprtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> c_int;
pub(crate) type HiprtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> c_int;

pub(crate) type RocblasCreateHandle = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
pub(crate) type RocblasDestroyHandle = unsafe extern "C" fn(*mut c_void) -> c_int;
pub(crate) type RocblasSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
pub(crate) type RocblasSgemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> c_int;

#[derive(Debug)]
pub(crate) struct HipApi {
    _hip_lib: Library,
    _rtc_lib: Library,
    pub get_device_count: HipGetDeviceCount,
    pub set_device: HipSetDevice,
    pub get_error_string: HipGetErrorString,
    pub malloc: HipMalloc,
    pub free: HipFree,
    pub memcpy: HipMemcpy,
    pub memcpy_async: HipMemcpyAsync,
    pub device_synchronize: HipDeviceSynchronize,
    pub stream_create: HipStreamCreate,
    pub stream_destroy: HipStreamDestroy,
    pub stream_synchronize: HipStreamSynchronize,
    pub stream_begin_capture: HipStreamBeginCapture,
    pub stream_end_capture: HipStreamEndCapture,
    pub graph_instantiate: HipGraphInstantiate,
    pub graph_launch: HipGraphLaunch,
    pub graph_exec_destroy: HipGraphExecDestroy,
    pub graph_destroy: HipGraphDestroy,
    pub module_load_data: HipModuleLoadData,
    pub module_unload: HipModuleUnload,
    pub module_get_function: HipModuleGetFunction,
    pub module_launch_kernel: HipModuleLaunchKernel,
    pub rtc_create_program: HiprtcCreateProgram,
    pub rtc_compile_program: HiprtcCompileProgram,
    pub rtc_destroy_program: HiprtcDestroyProgram,
    pub rtc_get_program_log_size: HiprtcGetProgramLogSize,
    pub rtc_get_program_log: HiprtcGetProgramLog,
    pub rtc_get_code_size: HiprtcGetCodeSize,
    pub rtc_get_code: HiprtcGetCode,
}

#[derive(Debug)]
pub(crate) struct RocblasApi {
    _lib: Library,
    pub create_handle: RocblasCreateHandle,
    pub destroy_handle: RocblasDestroyHandle,
    pub set_stream: RocblasSetStream,
    pub sgemm: RocblasSgemm,
}

impl HipApi {
    pub(crate) fn load() -> Result<Arc<Self>> {
        let hip_path = find_library("libamdhip64.so")?;
        let rtc_path = find_library("libhiprtc.so")?;
        let hip_lib = load_library(&hip_path)?;
        let rtc_lib = load_library(&rtc_path)?;
        let api = unsafe {
            Self {
                get_device_count: load_symbol(&hip_lib, "libamdhip64", b"hipGetDeviceCount\0")?,
                set_device: load_symbol(&hip_lib, "libamdhip64", b"hipSetDevice\0")?,
                get_error_string: load_symbol(&hip_lib, "libamdhip64", b"hipGetErrorString\0")?,
                malloc: load_symbol(&hip_lib, "libamdhip64", b"hipMalloc\0")?,
                free: load_symbol(&hip_lib, "libamdhip64", b"hipFree\0")?,
                memcpy: load_symbol(&hip_lib, "libamdhip64", b"hipMemcpy\0")?,
                memcpy_async: load_symbol(&hip_lib, "libamdhip64", b"hipMemcpyAsync\0")?,
                device_synchronize: load_symbol(
                    &hip_lib,
                    "libamdhip64",
                    b"hipDeviceSynchronize\0",
                )?,
                stream_create: load_symbol(&hip_lib, "libamdhip64", b"hipStreamCreate\0")?,
                stream_destroy: load_symbol(&hip_lib, "libamdhip64", b"hipStreamDestroy\0")?,
                stream_synchronize: load_symbol(
                    &hip_lib,
                    "libamdhip64",
                    b"hipStreamSynchronize\0",
                )?,
                stream_begin_capture: load_symbol(
                    &hip_lib,
                    "libamdhip64",
                    b"hipStreamBeginCapture\0",
                )?,
                stream_end_capture: load_symbol(&hip_lib, "libamdhip64", b"hipStreamEndCapture\0")?,
                graph_instantiate: load_symbol(&hip_lib, "libamdhip64", b"hipGraphInstantiate\0")?,
                graph_launch: load_symbol(&hip_lib, "libamdhip64", b"hipGraphLaunch\0")?,
                graph_exec_destroy: load_symbol(&hip_lib, "libamdhip64", b"hipGraphExecDestroy\0")?,
                graph_destroy: load_symbol(&hip_lib, "libamdhip64", b"hipGraphDestroy\0")?,
                module_load_data: load_symbol(&hip_lib, "libamdhip64", b"hipModuleLoadData\0")?,
                module_unload: load_symbol(&hip_lib, "libamdhip64", b"hipModuleUnload\0")?,
                module_get_function: load_symbol(
                    &hip_lib,
                    "libamdhip64",
                    b"hipModuleGetFunction\0",
                )?,
                module_launch_kernel: load_symbol(
                    &hip_lib,
                    "libamdhip64",
                    b"hipModuleLaunchKernel\0",
                )?,
                rtc_create_program: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcCreateProgram\0")?,
                rtc_compile_program: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcCompileProgram\0")?,
                rtc_destroy_program: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcDestroyProgram\0")?,
                rtc_get_program_log_size: load_symbol(
                    &rtc_lib,
                    "libhiprtc",
                    b"hiprtcGetProgramLogSize\0",
                )?,
                rtc_get_program_log: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcGetProgramLog\0")?,
                rtc_get_code_size: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcGetCodeSize\0")?,
                rtc_get_code: load_symbol(&rtc_lib, "libhiprtc", b"hiprtcGetCode\0")?,
                _hip_lib: hip_lib,
                _rtc_lib: rtc_lib,
            }
        };
        Ok(Arc::new(api))
    }

    pub(crate) fn check(&self, call: &'static str, status: i32) -> Result<()> {
        if status == HIP_SUCCESS {
            return Ok(());
        }
        let message = unsafe {
            let ptr = (self.get_error_string)(status);
            if ptr.is_null() {
                "unknown HIP error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(Error::Hip {
            call,
            status,
            message,
        })
    }

    pub(crate) fn check_hiprtc(
        &self,
        call: &'static str,
        status: i32,
        program: Option<HiprtcProgram>,
    ) -> Result<()> {
        if status == HIPRTC_SUCCESS {
            return Ok(());
        }
        let log = program
            .map(|program| self.program_log(program))
            .transpose()?
            .unwrap_or_default();
        Err(Error::Hiprtc { call, status, log })
    }

    pub(crate) fn program_log(&self, program: HiprtcProgram) -> Result<String> {
        let mut len = 0usize;
        let status = unsafe { (self.rtc_get_program_log_size)(program, &mut len as *mut usize) };
        self.check_hiprtc("hiprtcGetProgramLogSize", status, None)?;
        if len == 0 {
            return Ok(String::new());
        }
        let mut bytes = vec![0i8; len];
        let status = unsafe { (self.rtc_get_program_log)(program, bytes.as_mut_ptr()) };
        self.check_hiprtc("hiprtcGetProgramLog", status, None)?;
        let bytes = bytes
            .into_iter()
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .to_string())
    }
}

impl RocblasApi {
    pub(crate) fn load() -> Result<Arc<Self>> {
        let path = find_library("librocblas.so")?;
        let lib = load_library(&path)?;
        let api = unsafe {
            Self {
                create_handle: load_symbol(&lib, "librocblas", b"rocblas_create_handle\0")?,
                destroy_handle: load_symbol(&lib, "librocblas", b"rocblas_destroy_handle\0")?,
                set_stream: load_symbol(&lib, "librocblas", b"rocblas_set_stream\0")?,
                sgemm: load_symbol(&lib, "librocblas", b"rocblas_sgemm\0")?,
                _lib: lib,
            }
        };
        Ok(Arc::new(api))
    }

    pub(crate) fn check(call: &'static str, status: i32) -> Result<()> {
        if status == ROCBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::Rocblas { call, status })
        }
    }
}

unsafe fn load_symbol<T: Copy>(
    lib: &Library,
    library: &'static str,
    symbol: &'static [u8],
) -> Result<T> {
    unsafe { lib.get::<T>(symbol) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::SymbolLoad {
            library,
            symbol: std::str::from_utf8(&symbol[..symbol.len() - 1]).unwrap_or("<invalid>"),
            source,
        })
}

fn load_library(path: &Path) -> Result<Library> {
    unsafe { Library::new(path) }.map_err(|source| Error::LibraryLoad {
        library: path.to_path_buf(),
        source,
    })
}

fn find_library(name: &'static str) -> Result<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(path) = std::env::var("ROCM_PATH") {
        dirs.push(PathBuf::from(path).join("lib"));
    }
    if let Ok(path) = std::env::var("HIP_PATH") {
        dirs.push(PathBuf::from(path).join("lib"));
    }
    dirs.push(PathBuf::from("/opt/rocm/lib"));
    dirs.push(PathBuf::from("/opt/rocm-7.2.4/lib"));

    for dir in &dirs {
        let exact = dir.join(name);
        if exact.exists() {
            return Ok(exact);
        }
    }

    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.starts_with(name) {
                return Ok(path);
            }
        }
    }

    Err(Error::LibraryNotFound { name })
}
