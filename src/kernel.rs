use std::ffi::{CString, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::ffi::{HipApi, HiprtcProgram};
use crate::graph::HipStream;

#[derive(Debug)]
pub struct HipModule {
    module: NonNull<c_void>,
    hip: Arc<HipApi>,
    _code: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct HipFunction {
    function: NonNull<c_void>,
    hip: Arc<HipApi>,
}

impl HipModule {
    pub(crate) fn compile(hip: Arc<HipApi>, name: &str, source: &str) -> Result<Self> {
        let source = CString::new(source)
            .map_err(|err| Error::InvalidInput(format!("kernel source contains nul: {err}")))?;
        let name = CString::new(name)
            .map_err(|err| Error::InvalidInput(format!("kernel name contains nul: {err}")))?;
        let mut program: HiprtcProgram = std::ptr::null_mut();
        let status = unsafe {
            (hip.rtc_create_program)(
                &mut program as *mut HiprtcProgram,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        hip.check_hiprtc("hiprtcCreateProgram", status, None)?;

        let option = CString::new("--std=c++17").unwrap();
        let options = [option.as_ptr()];
        let status =
            unsafe { (hip.rtc_compile_program)(program, options.len() as i32, options.as_ptr()) };
        if let Err(err) = hip.check_hiprtc("hiprtcCompileProgram", status, Some(program)) {
            unsafe { (hip.rtc_destroy_program)(&mut program as *mut HiprtcProgram) };
            return Err(err);
        }

        let mut code_size = 0usize;
        let status = unsafe { (hip.rtc_get_code_size)(program, &mut code_size as *mut usize) };
        hip.check_hiprtc("hiprtcGetCodeSize", status, Some(program))?;
        let mut code = vec![0u8; code_size];
        let status = unsafe { (hip.rtc_get_code)(program, code.as_mut_ptr() as *mut i8) };
        hip.check_hiprtc("hiprtcGetCode", status, Some(program))?;
        unsafe { (hip.rtc_destroy_program)(&mut program as *mut HiprtcProgram) };

        let mut module = std::ptr::null_mut();
        let status = unsafe {
            (hip.module_load_data)(
                &mut module as *mut *mut c_void,
                code.as_ptr() as *const c_void,
            )
        };
        hip.check("hipModuleLoadData", status)?;
        let module = NonNull::new(module)
            .ok_or_else(|| Error::InvalidInput("hipModuleLoadData returned null".to_string()))?;
        Ok(Self {
            module,
            hip,
            _code: code,
        })
    }

    pub fn function(&self, name: &str) -> Result<HipFunction> {
        let name = CString::new(name)
            .map_err(|err| Error::InvalidInput(format!("function name contains nul: {err}")))?;
        let mut function = std::ptr::null_mut();
        let status = unsafe {
            (self.hip.module_get_function)(
                &mut function as *mut *mut c_void,
                self.module.as_ptr(),
                name.as_ptr(),
            )
        };
        self.hip.check("hipModuleGetFunction", status)?;
        let function = NonNull::new(function)
            .ok_or_else(|| Error::InvalidInput("hipModuleGetFunction returned null".to_string()))?;
        Ok(HipFunction {
            function,
            hip: self.hip.clone(),
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        let _ = unsafe { (self.hip.module_unload)(self.module.as_ptr()) };
    }
}

impl HipFunction {
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        self.launch_on_stream(grid, block, shared_mem_bytes, params, None)
    }

    pub fn launch_on_stream(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
        stream: Option<&HipStream>,
    ) -> Result<()> {
        let status = unsafe {
            (self.hip.module_launch_kernel)(
                self.function.as_ptr(),
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream
                    .map(HipStream::as_mut_ptr)
                    .unwrap_or(std::ptr::null_mut()),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.hip.check("hipModuleLaunchKernel", status)
    }
}
