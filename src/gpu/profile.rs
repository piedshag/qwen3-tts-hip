use std::ffi::{CString, c_char, c_int};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::Library;

type RoctxRangePush = unsafe extern "C" fn(*const c_char) -> c_int;
type RoctxRangePop = unsafe extern "C" fn() -> c_int;

struct RoctxApi {
    _lib: Library,
    range_push: RoctxRangePush,
    range_pop: RoctxRangePop,
}

pub struct RangeGuard {
    active: bool,
}

pub fn range(name: &'static str) -> RangeGuard {
    let Some(api) = roctx_api() else {
        return RangeGuard { active: false };
    };
    let Ok(name) = CString::new(name) else {
        return RangeGuard { active: false };
    };
    unsafe {
        (api.range_push)(name.as_ptr());
    }
    RangeGuard { active: true }
}

impl Drop for RangeGuard {
    fn drop(&mut self) {
        if let (true, Some(api)) = (self.active, roctx_api()) {
            unsafe {
                (api.range_pop)();
            }
        }
    }
}

fn roctx_api() -> Option<&'static RoctxApi> {
    static API: OnceLock<Option<RoctxApi>> = OnceLock::new();
    API.get_or_init(load_roctx).as_ref()
}

fn load_roctx() -> Option<RoctxApi> {
    if !matches!(
        std::env::var("QWEN3_HIP_ROCTX").as_deref(),
        Ok("1" | "true" | "yes" | "on")
    ) {
        return None;
    }

    for name in ["librocprofiler-sdk-roctx.so", "libroctx64.so"] {
        let Some(path) = find_library(name) else {
            continue;
        };
        let Ok(lib) = (unsafe { Library::new(path) }) else {
            continue;
        };
        let range_push = unsafe { lib.get::<RoctxRangePush>(b"roctxRangePushA\0").ok()? };
        let range_pop = unsafe { lib.get::<RoctxRangePop>(b"roctxRangePop\0").ok()? };
        return Some(RoctxApi {
            range_push: *range_push,
            range_pop: *range_pop,
            _lib: lib,
        });
    }
    None
}

fn find_library(name: &'static str) -> Option<PathBuf> {
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
            return Some(exact);
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
                return Some(path);
            }
        }
    }
    None
}
