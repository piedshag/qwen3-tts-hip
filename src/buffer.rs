use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::ffi::{
    HIP_MEMCPY_DEVICE_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE, HipApi,
};
use crate::graph::HipStream;

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    ptr: NonNull<c_void>,
    len: usize,
    hip: Arc<HipApi>,
    _marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    pub(crate) fn uninitialized(hip: Arc<HipApi>, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidInput(
                "device buffer length must be non-zero".to_string(),
            ));
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::InvalidInput("device buffer size overflow".to_string()))?;
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { (hip.malloc)(&mut ptr as *mut *mut c_void, bytes) };
        hip.check("hipMalloc", status)?;
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::InvalidInput("hipMalloc returned null".to_string()))?;
        Ok(Self {
            ptr,
            len,
            hip,
            _marker: PhantomData,
        })
    }

    pub(crate) fn from_slice(hip: Arc<HipApi>, data: &[T]) -> Result<Self> {
        let buffer = Self::uninitialized(hip, data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }

    pub(crate) fn from_same_context(&self, data: &[T]) -> Result<Self> {
        Self::from_slice(self.hip.clone(), data)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr_at(&self, offset: usize) -> Result<*mut c_void> {
        if offset > self.len {
            return Err(Error::InvalidInput(format!(
                "buffer offset {offset} exceeds length {}",
                self.len
            )));
        }
        Ok(unsafe { (self.ptr.as_ptr() as *mut T).add(offset) as *mut c_void })
    }

    pub fn as_ptr_at(&self, offset: usize) -> Result<*const c_void> {
        if offset > self.len {
            return Err(Error::InvalidInput(format!(
                "buffer offset {offset} exceeds length {}",
                self.len
            )));
        }
        Ok(unsafe { (self.ptr.as_ptr() as *const T).add(offset) as *const c_void })
    }

    pub fn copy_from_host(&self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(Error::InvalidInput(format!(
                "host length {} does not match device length {}",
                data.len(),
                self.len
            )));
        }
        let bytes = self.byte_len();
        let status = unsafe {
            (self.hip.memcpy)(
                self.ptr.as_ptr(),
                data.as_ptr() as *const c_void,
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        self.hip.check("hipMemcpy(H2D)", status)
    }

    pub fn copy_to_host(&self) -> Result<Vec<T>> {
        let mut data = Vec::<T>::with_capacity(self.len);
        let status = unsafe {
            (self.hip.memcpy)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr.as_ptr(),
                self.byte_len(),
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        self.hip.check("hipMemcpy(D2H)", status)?;
        unsafe { data.set_len(self.len) };
        Ok(data)
    }

    pub fn copy_from_device(&self, source: &Self) -> Result<()> {
        if source.len != self.len {
            return Err(Error::InvalidInput(format!(
                "source length {} does not match destination length {}",
                source.len, self.len
            )));
        }
        let status = unsafe {
            (self.hip.memcpy)(
                self.ptr.as_ptr(),
                source.ptr.as_ptr(),
                self.byte_len(),
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        self.hip.check("hipMemcpy(D2D)", status)
    }

    pub fn copy_from_device_range(
        &self,
        source: &Self,
        source_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.copy_from_device_range_at(0, source, source_offset, len)
    }

    pub fn copy_from_device_range_at(
        &self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        len: usize,
    ) -> Result<()> {
        if destination_offset + len > self.len || source_offset + len > source.len {
            return Err(Error::InvalidInput(format!(
                "invalid D2D range: destination length {}, source length {}, destination_offset={destination_offset}, source_offset={source_offset}, len={len}",
                self.len, source.len
            )));
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::InvalidInput("device copy size overflow".to_string()))?;
        let destination_ptr = unsafe { (self.ptr.as_ptr() as *mut T).add(destination_offset) };
        let source_ptr = unsafe { (source.ptr.as_ptr() as *const T).add(source_offset) };
        let status = unsafe {
            (self.hip.memcpy)(
                destination_ptr as *mut c_void,
                source_ptr as *const c_void,
                bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        self.hip.check("hipMemcpy(D2D range)", status)
    }

    pub fn copy_from_device_on_stream(&self, source: &Self, stream: &HipStream) -> Result<()> {
        if source.len != self.len {
            return Err(Error::InvalidInput(format!(
                "source length {} does not match destination length {}",
                source.len, self.len
            )));
        }
        let status = unsafe {
            (self.hip.memcpy_async)(
                self.ptr.as_ptr(),
                source.ptr.as_ptr(),
                self.byte_len(),
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                stream.as_mut_ptr(),
            )
        };
        self.hip.check("hipMemcpyAsync(D2D)", status)
    }

    fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _ = unsafe { (self.hip.free)(self.ptr.as_ptr()) };
    }
}
