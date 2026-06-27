use std::ffi::c_int;
use std::sync::Arc;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::error::{Error, Result};
use crate::ffi::{HipApi, RocblasApi};
use crate::graph::HipStream;
use crate::kernel::HipModule;

#[derive(Clone, Debug)]
pub struct HipRuntime {
    pub(crate) hip: Arc<HipApi>,
    pub(crate) rocblas: Arc<RocblasApi>,
    device_index: i32,
}

impl HipRuntime {
    pub fn new(device_index: i32) -> Result<Self> {
        let hip = HipApi::load()?;
        let rocblas = RocblasApi::load()?;
        let runtime = Self {
            hip,
            rocblas,
            device_index,
        };
        runtime.set_device()?;
        Ok(runtime)
    }

    pub fn device_count(&self) -> Result<i32> {
        let mut count = 0 as c_int;
        let status = unsafe { (self.hip.get_device_count)(&mut count as *mut c_int) };
        self.hip.check("hipGetDeviceCount", status)?;
        Ok(count)
    }

    pub fn device_index(&self) -> i32 {
        self.device_index
    }

    pub fn set_device(&self) -> Result<()> {
        let status = unsafe { (self.hip.set_device)(self.device_index as c_int) };
        self.hip.check("hipSetDevice", status)
    }

    pub fn synchronize(&self) -> Result<()> {
        let status = unsafe { (self.hip.device_synchronize)() };
        self.hip.check("hipDeviceSynchronize", status)
    }

    pub fn create_blas_handle(&self) -> Result<RocblasHandle> {
        RocblasHandle::new(self.rocblas.clone())
    }

    pub fn buffer_from_slice<T: Copy>(&self, data: &[T]) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_slice(self.hip.clone(), data)
    }

    pub fn empty_buffer<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        if len == 0 {
            return Err(Error::InvalidInput(
                "device buffer length must be non-zero".to_string(),
            ));
        }
        DeviceBuffer::uninitialized(self.hip.clone(), len)
    }

    pub fn compile_module(&self, name: &str, source: &str) -> Result<HipModule> {
        HipModule::compile(self.hip.clone(), name, source)
    }

    pub fn create_stream(&self) -> Result<HipStream> {
        HipStream::new(self.hip.clone())
    }
}
