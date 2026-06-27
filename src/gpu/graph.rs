use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::ffi::HipApi;

const HIP_STREAM_CAPTURE_MODE_GLOBAL: i32 = 0;

#[derive(Debug)]
pub struct HipStream {
    stream: NonNull<c_void>,
    hip: Arc<HipApi>,
}

#[derive(Debug)]
pub struct HipGraph {
    graph: NonNull<c_void>,
    hip: Arc<HipApi>,
}

#[derive(Debug)]
pub struct HipGraphExec {
    exec: NonNull<c_void>,
    hip: Arc<HipApi>,
}

impl HipStream {
    pub(crate) fn new(hip: Arc<HipApi>) -> Result<Self> {
        let mut stream = std::ptr::null_mut();
        let status = unsafe { (hip.stream_create)(&mut stream as *mut *mut c_void) };
        hip.check("hipStreamCreate", status)?;
        let stream = NonNull::new(stream)
            .ok_or_else(|| Error::InvalidInput("hipStreamCreate returned null".to_string()))?;
        Ok(Self { stream, hip })
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    pub fn synchronize(&self) -> Result<()> {
        let status = unsafe { (self.hip.stream_synchronize)(self.stream.as_ptr()) };
        self.hip.check("hipStreamSynchronize", status)
    }

    pub fn begin_capture(&self) -> Result<()> {
        let status = unsafe {
            (self.hip.stream_begin_capture)(self.stream.as_ptr(), HIP_STREAM_CAPTURE_MODE_GLOBAL)
        };
        self.hip.check("hipStreamBeginCapture", status)
    }

    pub fn end_capture(&self) -> Result<HipGraph> {
        let mut graph = std::ptr::null_mut();
        let status = unsafe {
            (self.hip.stream_end_capture)(self.stream.as_ptr(), &mut graph as *mut *mut c_void)
        };
        self.hip.check("hipStreamEndCapture", status)?;
        let graph = NonNull::new(graph)
            .ok_or_else(|| Error::InvalidInput("hipStreamEndCapture returned null".to_string()))?;
        Ok(HipGraph {
            graph,
            hip: self.hip.clone(),
        })
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        let _ = unsafe { (self.hip.stream_destroy)(self.stream.as_ptr()) };
    }
}

impl HipGraph {
    pub fn instantiate(&self) -> Result<HipGraphExec> {
        let mut exec = std::ptr::null_mut();
        let mut error_node = std::ptr::null_mut();
        let status = unsafe {
            (self.hip.graph_instantiate)(
                &mut exec as *mut *mut c_void,
                self.graph.as_ptr(),
                &mut error_node as *mut *mut c_void,
                std::ptr::null_mut(),
                0,
            )
        };
        self.hip.check("hipGraphInstantiate", status)?;
        let exec = NonNull::new(exec)
            .ok_or_else(|| Error::InvalidInput("hipGraphInstantiate returned null".to_string()))?;
        Ok(HipGraphExec {
            exec,
            hip: self.hip.clone(),
        })
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        let _ = unsafe { (self.hip.graph_destroy)(self.graph.as_ptr()) };
    }
}

impl HipGraphExec {
    pub fn launch(&self, stream: &HipStream) -> Result<()> {
        let status = unsafe { (self.hip.graph_launch)(self.exec.as_ptr(), stream.as_mut_ptr()) };
        self.hip.check("hipGraphLaunch", status)
    }
}

impl Drop for HipGraphExec {
    fn drop(&mut self) {
        let _ = unsafe { (self.hip.graph_exec_destroy)(self.exec.as_ptr()) };
    }
}
