use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::buffer::DeviceBuffer;
use crate::error::{Error, Result};
use crate::ffi::{ROCBLAS_OPERATION_NONE, RocblasApi};
use crate::graph::HipStream;

#[derive(Debug)]
pub struct RocblasHandle {
    handle: NonNull<c_void>,
    api: Arc<RocblasApi>,
    stream: Cell<*mut c_void>,
}

impl RocblasHandle {
    pub(crate) fn new(api: Arc<RocblasApi>) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { (api.create_handle)(&mut handle as *mut *mut c_void) };
        RocblasApi::check("rocblas_create_handle", status)?;
        let handle = NonNull::new(handle).ok_or_else(|| {
            Error::InvalidInput("rocblas_create_handle returned null".to_string())
        })?;
        Ok(Self {
            handle,
            api,
            stream: Cell::new(std::ptr::null_mut()),
        })
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }

    pub fn set_stream(&self, stream: Option<&HipStream>) -> Result<()> {
        let stream_ptr = stream
            .map(HipStream::as_mut_ptr)
            .unwrap_or(std::ptr::null_mut());
        if self.stream.get() == stream_ptr {
            return Ok(());
        }
        let status = unsafe { (self.api.set_stream)(self.handle.as_ptr(), stream_ptr) };
        RocblasApi::check("rocblas_set_stream", status)?;
        self.stream.set(stream_ptr);
        Ok(())
    }

    pub fn sgemm_row_major(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.set_stream(None)?;
        self.sgemm_row_major_impl(a, b, c, m, n, k)
    }

    pub fn sgemm_row_major_on_stream(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        stream: &HipStream,
    ) -> Result<()> {
        self.set_stream(Some(stream))?;
        self.sgemm_row_major_impl(a, b, c, m, n, k)
    }

    fn sgemm_row_major_impl(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() < m * k {
            return Err(Error::InvalidInput(format!(
                "A length {} is smaller than m*k {}",
                a.len(),
                m * k
            )));
        }
        if b.len() < k * n {
            return Err(Error::InvalidInput(format!(
                "B length {} is smaller than k*n {}",
                b.len(),
                k * n
            )));
        }
        if c.len() < m * n {
            return Err(Error::InvalidInput(format!(
                "C length {} is smaller than m*n {}",
                c.len(),
                m * n
            )));
        }

        let alpha = 1.0f32;
        let beta = 0.0f32;
        let status = unsafe {
            (self.api.sgemm)(
                self.handle.as_ptr(),
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                n as c_int,
                m as c_int,
                k as c_int,
                &alpha as *const f32,
                b.as_ptr() as *const f32,
                n as c_int,
                a.as_ptr() as *const f32,
                k as c_int,
                &beta as *const f32,
                c.as_mut_ptr() as *mut f32,
                n as c_int,
            )
        };
        RocblasApi::check("rocblas_sgemm", status)
    }
}

impl Drop for RocblasHandle {
    fn drop(&mut self) {
        let _ = unsafe { (self.api.destroy_handle)(self.handle.as_ptr()) };
    }
}
