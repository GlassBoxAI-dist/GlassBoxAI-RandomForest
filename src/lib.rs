//! @file
//! @ingroup RF_Internal_Logic
//
// Matthew Abbott 2025
// Random Forest Library - Multi-Backend (CUDA, OpenCL, CPU, Hybrid)
//

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(kani)]
mod kani;

pub const MAX_FEATURES: usize = 100;
pub const MAX_SAMPLES: usize = 10000;
pub const MAX_TREES: usize = 500;
pub const MAX_DEPTH_DEFAULT: i32 = 10;
pub const MIN_SAMPLES_LEAF_DEFAULT: i32 = 1;
pub const MIN_SAMPLES_SPLIT_DEFAULT: i32 = 2;
pub const MAX_NODES: usize = 4096;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(i32)]
pub enum TaskType {
    Classification = 0,
    Regression = 1,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SplitCriterion {
    Gini,
    Entropy,
    MSE,
    VarianceReduction,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AggregationMethod {
    MajorityVote,
    WeightedVote,
    Mean,
    WeightedMean,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BackendKind {
    Auto,
    Cuda,
    OpenCl,
    Cpu,
    Hybrid,
}

impl BackendKind {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" => BackendKind::Cuda,
            "opencl" | "cl" => BackendKind::OpenCl,
            "cpu" => BackendKind::Cpu,
            "hybrid" | "mixed" => BackendKind::Hybrid,
            _ => BackendKind::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct FlatTreeNode {
    pub is_leaf: i32,
    pub feature_index: i32,
    pub threshold: f64,
    pub prediction: f64,
    pub class_label: i32,
    pub left_child: i32,
    pub right_child: i32,
}

#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for FlatTreeNode {}

#[derive(Clone)]
pub struct FlatTree {
    pub nodes: Vec<FlatTreeNode>,
    pub num_nodes: usize,
    pub oob_indices: Vec<bool>,
    pub num_oob_indices: usize,
}

impl FlatTree {
    fn new() -> Self {
        Self {
            nodes: vec![FlatTreeNode::default(); MAX_NODES],
            num_nodes: 0,
            oob_indices: vec![false; MAX_SAMPLES],
            num_oob_indices: 0,
        }
    }
}

#[allow(dead_code)]
struct TreeNode {
    is_leaf: bool,
    feature_index: i32,
    threshold: f64,
    prediction: f64,
    class_label: i32,
    impurity: f64,
    num_samples: usize,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new_leaf(prediction: f64, class_label: i32, impurity: f64, num_samples: usize) -> Self {
        Self {
            is_leaf: true,
            feature_index: -1,
            threshold: 0.0,
            prediction,
            class_label,
            impurity,
            num_samples,
            left: None,
            right: None,
        }
    }

    fn new_split(
        feature_index: i32,
        threshold: f64,
        prediction: f64,
        class_label: i32,
        impurity: f64,
        num_samples: usize,
    ) -> Self {
        Self {
            is_leaf: false,
            feature_index,
            threshold,
            prediction,
            class_label,
            impurity,
            num_samples,
            left: None,
            right: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NodeInfo {
    pub node_id: i32,
    pub depth: i32,
    pub is_leaf: bool,
    pub feature_index: i32,
    pub threshold: f64,
    pub prediction: f64,
    pub class_label: i32,
    pub impurity: f64,
    pub num_samples: i32,
    pub left_child_id: i32,
    pub right_child_id: i32,
}

#[derive(Clone, Debug, Default)]
pub struct TreeInfo {
    pub tree_id: i32,
    pub num_nodes: i32,
    pub max_depth: i32,
    pub num_leaves: i32,
    pub features_used: Vec<bool>,
    pub num_features_used: i32,
    pub oob_error: f64,
    pub nodes: Vec<NodeInfo>,
}

#[derive(Clone, Debug, Default)]
pub struct FeatureStats {
    pub feature_index: i32,
    pub times_used: i32,
    pub trees_used_in: i32,
    pub avg_importance: f64,
    pub total_importance: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SampleTrackInfo {
    pub sample_index: i32,
    pub trees_influenced: Vec<bool>,
    pub num_trees_influenced: i32,
    pub oob_trees: Vec<bool>,
    pub num_oob_trees: i32,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct OOBTreeInfo {
    pub tree_id: i32,
    pub num_oob_samples: i32,
    pub oob_error: f64,
    pub oob_accuracy: f64,
}

struct FlattenedForest {
    #[allow(dead_code)]
    all_nodes: Vec<FlatTreeNode>,
    #[allow(dead_code)]
    tree_offsets: Vec<i32>,
    #[allow(dead_code)]
    total_nodes: usize,
}

// ═══════════════════════════════════════════════════════════════════
// Compute Backend
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    d_all_tree_nodes: CudaSlice<FlatTreeNode>,
    d_tree_node_offsets: CudaSlice<i32>,
    #[allow(dead_code)]
    total_gpu_nodes: usize,
}

#[cfg(feature = "opencl")]
pub struct OpenClBackend {
    context: opencl3::context::Context,
    queue: opencl3::command_queue::CommandQueue,
    program: opencl3::program::Program,
    d_all_tree_nodes: opencl3::memory::Buffer<opencl3::types::cl_uchar>,
    d_tree_node_offsets: opencl3::memory::Buffer<opencl3::types::cl_int>,
    total_gpu_nodes: usize,
}

pub struct CpuBackend;

pub struct HybridBackend {
    gpu: Box<ComputeBackend>,
    gpu_fraction: f64,
}

pub enum ComputeBackend {
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClBackend),
    Cpu(CpuBackend),
    Hybrid(HybridBackend),
    None,
}

impl ComputeBackend {
    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            ComputeBackend::Cuda(_) => "CUDA",
            #[cfg(feature = "opencl")]
            ComputeBackend::OpenCl(_) => "OpenCL",
            ComputeBackend::Cpu(_) => "CPU",
            ComputeBackend::Hybrid(h) => {
                match h.gpu.as_ref() {
                    #[cfg(feature = "cuda")]
                    ComputeBackend::Cuda(_) => "Hybrid (CUDA+CPU)",
                    #[cfg(feature = "opencl")]
                    ComputeBackend::OpenCl(_) => "Hybrid (OpenCL+CPU)",
                    _ => "Hybrid (CPU)",
                }
            }
            ComputeBackend::None => "None",
        }
    }

    fn detect_and_create(kind: BackendKind) -> ComputeBackend {
        match kind {
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match Self::try_create_cuda() {
                        Some(b) => {
                            println!("Backend: CUDA");
                            b
                        }
                        None => {
                            eprintln!("Error: CUDA backend requested but not available");
                            ComputeBackend::None
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("Error: CUDA support not compiled in (enable 'cuda' feature)");
                    ComputeBackend::None
                }
            }
            BackendKind::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    match Self::try_create_opencl() {
                        Some(b) => {
                            println!("Backend: OpenCL");
                            b
                        }
                        None => {
                            eprintln!("Error: OpenCL backend requested but not available");
                            ComputeBackend::None
                        }
                    }
                }
                #[cfg(not(feature = "opencl"))]
                {
                    eprintln!("Error: OpenCL support not compiled in (enable 'opencl' feature)");
                    ComputeBackend::None
                }
            }
            BackendKind::Cpu => {
                println!("Backend: CPU");
                ComputeBackend::Cpu(CpuBackend)
            }
            BackendKind::Hybrid => {
                let gpu = Self::auto_detect_gpu();
                match gpu {
                    ComputeBackend::None | ComputeBackend::Cpu(_) => {
                        println!("Backend: CPU (no GPU found for hybrid mode)");
                        ComputeBackend::Cpu(CpuBackend)
                    }
                    gpu_backend => {
                        let name = gpu_backend.backend_name();
                        println!("Backend: Hybrid ({}+CPU)", name);
                        ComputeBackend::Hybrid(HybridBackend {
                            gpu: Box::new(gpu_backend),
                            gpu_fraction: 0.8,
                        })
                    }
                }
            }
            BackendKind::Auto => {
                let backend = Self::auto_detect_gpu();
                let name = backend.backend_name();
                println!("Backend: {} (auto-detected)", name);
                backend
            }
        }
    }

    fn auto_detect_gpu() -> ComputeBackend {
        #[cfg(feature = "cuda")]
        {
            if let Some(b) = Self::try_create_cuda() {
                return b;
            }
        }
        #[cfg(feature = "opencl")]
        {
            if let Some(b) = Self::try_create_opencl() {
                return b;
            }
        }
        ComputeBackend::Cpu(CpuBackend)
    }

    #[cfg(feature = "cuda")]
    fn try_create_cuda() -> Option<ComputeBackend> {
        match CudaDevice::new(0) {
            Ok(_) => Some(ComputeBackend::None),
            Err(_) => None,
        }
    }

    #[cfg(feature = "opencl")]
    fn try_create_opencl() -> Option<ComputeBackend> {
        use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU};
        use opencl3::context::Context;
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};

        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .ok()
            .and_then(|devs| if devs.is_empty() { None } else { Some(devs[0]) })
            .or_else(|| {
                get_all_devices(CL_DEVICE_TYPE_CPU)
                    .ok()
                    .and_then(|devs| if devs.is_empty() { None } else { Some(devs[0]) })
            })?;

        let device = opencl3::device::Device::new(device_id);
        let context = Context::from_device(&device).ok()?;
        let queue = CommandQueue::create_default_with_properties(
            &context, CL_QUEUE_PROFILING_ENABLE, 0
        ).ok()?;

        let kernel_src = include_str!("kernel.cl");
        let program = opencl3::program::Program::create_and_build_from_source(
            &context, kernel_src, ""
        ).ok()?;

        let d_all_tree_nodes = unsafe {
            opencl3::memory::Buffer::create(
                &context, opencl3::memory::CL_MEM_READ_ONLY, 1, std::ptr::null_mut()
            ).ok()?
        };
        let d_tree_node_offsets = unsafe {
            opencl3::memory::Buffer::create(
                &context, opencl3::memory::CL_MEM_READ_ONLY, 1, std::ptr::null_mut()
            ).ok()?
        };

        Some(ComputeBackend::OpenCl(OpenClBackend {
            context,
            queue,
            program,
            d_all_tree_nodes,
            d_tree_node_offsets,
            total_gpu_nodes: 0,
        }))
    }

    fn prepare_forest(&mut self, flat: &FlattenedForest) {
        match self {
            #[cfg(feature = "cuda")]
            ComputeBackend::Cuda(_) => {
                self.prepare_cuda(flat);
            }
            #[cfg(feature = "opencl")]
            ComputeBackend::OpenCl(_) => {
                self.prepare_opencl(flat);
            }
            ComputeBackend::Hybrid(h) => {
                h.gpu.prepare_forest(flat);
            }
            _ => {}
        }
    }

    #[cfg(feature = "cuda")]
    fn prepare_cuda(&mut self, flat: &FlattenedForest) {
        let device = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to initialize CUDA device: {}", e);
                return;
            }
        };

        let d_all_tree_nodes = match device.htod_sync_copy(&flat.all_nodes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy tree nodes to GPU: {}", e);
                return;
            }
        };

        let d_tree_node_offsets = match device.htod_sync_copy(&flat.tree_offsets) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy offsets to GPU: {}", e);
                return;
            }
        };

        *self = ComputeBackend::Cuda(CudaBackend {
            device,
            d_all_tree_nodes,
            d_tree_node_offsets,
            total_gpu_nodes: flat.total_nodes,
        });
    }

    #[cfg(feature = "opencl")]
    fn prepare_opencl(&mut self, flat: &FlattenedForest) {
        use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_COPY_HOST_PTR};

        if let ComputeBackend::OpenCl(ref mut ocl) = self {
            let node_byte_count = flat.all_nodes.len() * std::mem::size_of::<FlatTreeNode>();

            let d_nodes: Result<Buffer<opencl3::types::cl_uchar>, _> = unsafe {
                Buffer::create(
                    &ocl.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    node_byte_count,
                    flat.all_nodes.as_ptr() as *mut std::ffi::c_void,
                )
            };

            let d_offsets: Result<Buffer<opencl3::types::cl_int>, _> = unsafe {
                Buffer::create(
                    &ocl.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    flat.tree_offsets.len(),
                    flat.tree_offsets.as_ptr() as *mut std::ffi::c_void,
                )
            };

            match (d_nodes, d_offsets) {
                (Ok(nodes), Ok(offsets)) => {
                    ocl.d_all_tree_nodes = nodes;
                    ocl.d_tree_node_offsets = offsets;
                    ocl.total_gpu_nodes = flat.total_nodes;
                }
                _ => {
                    eprintln!("Warning: Failed to upload forest to OpenCL device");
                }
            }
        }
    }

    fn predict_batch(
        &self,
        forest: &TRandomForest,
        samples: &[f64],
        n_samples: usize,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        match self {
            #[cfg(feature = "cuda")]
            ComputeBackend::Cuda(cuda) => {
                Self::predict_batch_cuda(cuda, forest, samples, n_samples, weights)
            }
            #[cfg(feature = "opencl")]
            ComputeBackend::OpenCl(ocl) => {
                Self::predict_batch_opencl(ocl, forest, samples, n_samples, weights)
            }
            ComputeBackend::Cpu(_) => {
                Self::predict_batch_cpu(forest, samples, n_samples)
            }
            ComputeBackend::Hybrid(h) => {
                Self::predict_batch_hybrid(h, forest, samples, n_samples, weights)
            }
            ComputeBackend::None => {
                Self::predict_batch_cpu(forest, samples, n_samples)
            }
        }
    }

    fn predict_batch_cpu(forest: &TRandomForest, samples: &[f64], n_samples: usize) -> Vec<f64> {
        forest.predict_batch(samples, n_samples)
    }

    fn predict_batch_hybrid(
        hybrid: &HybridBackend,
        forest: &TRandomForest,
        samples: &[f64],
        n_samples: usize,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        let n_gpu = ((n_samples as f64) * hybrid.gpu_fraction) as usize;
        let n_cpu = n_samples - n_gpu;
        let num_features = forest.num_features as usize;

        let gpu_samples = &samples[..n_gpu * num_features];
        let cpu_samples = &samples[n_gpu * num_features..];

        let mut gpu_preds = hybrid.gpu.predict_batch(forest, gpu_samples, n_gpu, weights);
        let cpu_preds = Self::predict_batch_cpu(forest, cpu_samples, n_cpu);

        gpu_preds.extend_from_slice(&cpu_preds);
        gpu_preds
    }

    #[cfg(feature = "cuda")]
    fn predict_batch_cuda(
        cuda: &CudaBackend,
        forest: &TRandomForest,
        samples: &[f64],
        n_samples: usize,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        let kernel_src = include_str!("kernel.cu");
        let ptx = match cudarc::nvrtc::compile_ptx(kernel_src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: Failed to compile CUDA kernel: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let kernel_name = if weights.is_some() {
            "predictBatchWeightedKernel"
        } else {
            "predictBatchKernel"
        };

        if let Err(e) = cuda.device.load_ptx(ptx, "random_forest", &[kernel_name]) {
            eprintln!("Warning: Failed to load PTX: {}", e);
            return Self::predict_batch_cpu(forest, samples, n_samples);
        }

        let func = match cuda.device.get_func("random_forest", kernel_name) {
            Some(f) => f,
            None => {
                eprintln!("Warning: Failed to get kernel function");
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let d_samples = match cuda.device.htod_sync_copy(samples) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to copy samples to GPU: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let d_predictions: CudaSlice<f64> = match cuda.device.alloc_zeros(n_samples) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: Failed to allocate predictions on GPU: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let block_size = 256u32;
        let num_blocks = ((n_samples as u32) + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_features = forest.num_features;
        let num_trees = forest.num_trees as i32;
        let task_type = forest.task_type as i32;

        if let Some(w) = weights {
            let d_weights = match cuda.device.htod_sync_copy(w) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Warning: Failed to copy weights to GPU: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            };
            unsafe {
                if let Err(e) = func.launch(
                    config,
                    (
                        &d_samples,
                        num_features,
                        &cuda.d_all_tree_nodes,
                        &cuda.d_tree_node_offsets,
                        &d_weights,
                        num_trees,
                        n_samples as i32,
                        task_type,
                        &d_predictions,
                    ),
                ) {
                    eprintln!("Warning: Kernel launch failed: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            }
        } else {
            unsafe {
                if let Err(e) = func.launch(
                    config,
                    (
                        &d_samples,
                        num_features,
                        &cuda.d_all_tree_nodes,
                        &cuda.d_tree_node_offsets,
                        num_trees,
                        n_samples as i32,
                        task_type,
                        &d_predictions,
                    ),
                ) {
                    eprintln!("Warning: Kernel launch failed: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            }
        }

        match cuda.device.dtoh_sync_copy(&d_predictions) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: Failed to copy predictions from GPU: {}", e);
                Self::predict_batch_cpu(forest, samples, n_samples)
            }
        }
    }

    #[cfg(feature = "opencl")]
    fn predict_batch_opencl(
        ocl: &OpenClBackend,
        forest: &TRandomForest,
        samples: &[f64],
        n_samples: usize,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        use opencl3::kernel::{ExecuteKernel, Kernel};
        use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_COPY_HOST_PTR};
        use opencl3::types::CL_BLOCKING;

        let d_samples: Buffer<opencl3::types::cl_double> = match unsafe {
            Buffer::create(
                &ocl.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                samples.len(),
                samples.as_ptr() as *mut std::ffi::c_void,
            )
        } {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Warning: OpenCL failed to create sample buffer: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let d_predictions: Buffer<opencl3::types::cl_double> = match unsafe {
            Buffer::create(&ocl.context, CL_MEM_WRITE_ONLY, n_samples, std::ptr::null_mut())
        } {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Warning: OpenCL failed to create prediction buffer: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        };

        let num_features = forest.num_features;
        let num_trees = forest.num_trees as i32;
        let task_type = forest.task_type as i32;

        let local_size = 256usize;
        let global_size = ((n_samples + local_size - 1) / local_size) * local_size;

        if let Some(w) = weights {
            let kernel = match Kernel::create(&ocl.program, "predictBatchWeightedKernel") {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("Warning: OpenCL failed to create weighted kernel: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            };

            let d_weights: Buffer<opencl3::types::cl_double> = match unsafe {
                Buffer::create(
                    &ocl.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    w.len(),
                    w.as_ptr() as *mut std::ffi::c_void,
                )
            } {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Warning: OpenCL failed to create weight buffer: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            };

            let event = unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_samples)
                    .set_arg(&num_features)
                    .set_arg(&ocl.d_all_tree_nodes)
                    .set_arg(&ocl.d_tree_node_offsets)
                    .set_arg(&d_weights)
                    .set_arg(&num_trees)
                    .set_arg(&(n_samples as i32))
                    .set_arg(&task_type)
                    .set_arg(&d_predictions)
                    .set_global_work_size(global_size)
                    .set_local_work_size(local_size)
                    .enqueue_nd_range(&ocl.queue)
            };

            if let Err(e) = event {
                eprintln!("Warning: OpenCL kernel launch failed: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        } else {
            let kernel = match Kernel::create(&ocl.program, "predictBatchKernel") {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("Warning: OpenCL failed to create kernel: {}", e);
                    return Self::predict_batch_cpu(forest, samples, n_samples);
                }
            };

            let event = unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg(&d_samples)
                    .set_arg(&num_features)
                    .set_arg(&ocl.d_all_tree_nodes)
                    .set_arg(&ocl.d_tree_node_offsets)
                    .set_arg(&num_trees)
                    .set_arg(&(n_samples as i32))
                    .set_arg(&task_type)
                    .set_arg(&d_predictions)
                    .set_global_work_size(global_size)
                    .set_local_work_size(local_size)
                    .enqueue_nd_range(&ocl.queue)
            };

            if let Err(e) = event {
                eprintln!("Warning: OpenCL kernel launch failed: {}", e);
                return Self::predict_batch_cpu(forest, samples, n_samples);
            }
        }

        let _ = ocl.queue.finish();

        let mut predictions = vec![0.0f64; n_samples];
        let read_result = unsafe {
            ocl.queue.enqueue_read_buffer(
                &d_predictions,
                CL_BLOCKING,
                0,
                &mut predictions,
                &[],
            )
        };

        match read_result {
            Ok(_) => predictions,
            Err(e) => {
                eprintln!("Warning: OpenCL failed to read predictions: {}", e);
                Self::predict_batch_cpu(forest, samples, n_samples)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRandomForest
// ═══════════════════════════════════════════════════════════════════

pub struct TRandomForest {
    pub trees: Vec<Option<FlatTree>>,
    pub num_trees: usize,
    pub max_depth: i32,
    pub min_samples_leaf: i32,
    pub min_samples_split: i32,
    pub max_features: i32,
    pub num_features: i32,
    pub num_samples: usize,
    pub task_type: TaskType,
    pub criterion: SplitCriterion,
    pub feature_importances: Vec<f64>,
    pub random_seed: u64,
    pub rng_state: u64,

    pub data: Vec<f64>,
    pub targets: Vec<f64>,

    pub backend: ComputeBackend,
    pub backend_kind: BackendKind,
}

impl TRandomForest {
    #[cfg(not(kani))]
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self::new_with_seed(seed)
    }

    #[cfg(kani)]
    pub fn new() -> Self {
        Self::new_with_seed(42)
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            trees: vec![None; MAX_TREES],
            num_trees: 100,
            max_depth: MAX_DEPTH_DEFAULT,
            min_samples_leaf: MIN_SAMPLES_LEAF_DEFAULT,
            min_samples_split: MIN_SAMPLES_SPLIT_DEFAULT,
            max_features: 0,
            num_features: 0,
            num_samples: 0,
            task_type: TaskType::Classification,
            criterion: SplitCriterion::Gini,
            feature_importances: vec![0.0; MAX_FEATURES],
            random_seed: 42,
            rng_state: seed,
            data: vec![0.0; MAX_SAMPLES * MAX_FEATURES],
            targets: vec![0.0; MAX_SAMPLES],
            backend: ComputeBackend::None,
            backend_kind: BackendKind::Auto,
        }
    }

    pub fn set_backend_kind(&mut self, kind: BackendKind) {
        self.backend_kind = kind;
    }

    pub fn set_num_trees(&mut self, n: usize) {
        self.num_trees = n.clamp(1, MAX_TREES);
    }

    pub fn set_max_depth(&mut self, d: i32) {
        self.max_depth = d.max(1);
    }

    pub fn set_min_samples_leaf(&mut self, m: i32) {
        self.min_samples_leaf = m.max(1);
    }

    pub fn set_min_samples_split(&mut self, m: i32) {
        self.min_samples_split = m.max(2);
    }

    pub fn set_max_features(&mut self, m: i32) {
        self.max_features = m;
    }

    pub fn set_task_type(&mut self, t: TaskType) {
        self.task_type = t;
        self.criterion = if t == TaskType::Classification {
            SplitCriterion::Gini
        } else {
            SplitCriterion::MSE
        };
    }

    pub fn set_criterion(&mut self, c: SplitCriterion) {
        self.criterion = c;
    }

    pub fn set_random_seed(&mut self, seed: u64) {
        self.random_seed = seed;
        self.rng_state = seed;
    }

    fn random_int(&mut self, max_val: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.rng_state >> 33) as usize) % max_val
    }

    #[allow(dead_code)]
    fn random_double(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn load_data(&mut self, input_data: &[f64], input_targets: &[f64], n_samples: usize, n_features: usize) {
        self.num_samples = n_samples;
        self.num_features = n_features as i32;

        if self.max_features == 0 {
            if self.task_type == TaskType::Classification {
                self.max_features = (n_features as f64).sqrt().round() as i32;
            } else {
                self.max_features = (n_features / 3) as i32;
            }
            if self.max_features < 1 {
                self.max_features = 1;
            }
        }

        for i in 0..n_samples {
            for j in 0..n_features {
                self.data[i * MAX_FEATURES + j] = input_data[i * n_features + j];
            }
            self.targets[i] = input_targets[i];
        }
    }

    fn bootstrap(&mut self, sample_indices: &mut Vec<usize>, oob_mask: &mut Vec<bool>) {
        sample_indices.clear();
        sample_indices.resize(self.num_samples, 0);
        oob_mask.clear();
        oob_mask.resize(self.num_samples, true);

        for i in 0..self.num_samples {
            let idx = self.random_int(self.num_samples);
            sample_indices[i] = idx;
            oob_mask[idx] = false;
        }
    }

    fn select_feature_subset(&mut self) -> Vec<usize> {
        let mut available: Vec<usize> = (0..self.num_features as usize).collect();

        for i in (1..self.num_features as usize).rev() {
            let j = self.random_int(i + 1);
            available.swap(i, j);
        }

        let num_selected = (self.max_features as usize).min(self.num_features as usize);
        available.truncate(num_selected);
        available
    }

    fn calculate_gini(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        let n = indices.len() as f64;
        let mut gini = 1.0;
        for count in &class_count {
            let prob = *count as f64 / n;
            gini -= prob * prob;
        }
        gini
    }

    fn calculate_entropy(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        let n = indices.len() as f64;
        let mut entropy = 0.0;
        for count in &class_count {
            if *count > 0 {
                let prob = *count as f64 / n;
                entropy -= prob * prob.log2();
            }
        }
        entropy
    }

    fn calculate_mse(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let mean: f64 = indices.iter().map(|&i| self.targets[i]).sum::<f64>() / indices.len() as f64;

        indices.iter().map(|&i| {
            let diff = self.targets[i] - mean;
            diff * diff
        }).sum::<f64>() / indices.len() as f64
    }

    fn calculate_impurity(&self, indices: &[usize]) -> f64 {
        match self.criterion {
            SplitCriterion::Gini => self.calculate_gini(indices),
            SplitCriterion::Entropy => self.calculate_entropy(indices),
            SplitCriterion::MSE | SplitCriterion::VarianceReduction => self.calculate_mse(indices),
        }
    }

    fn find_best_split(
        &self,
        indices: &[usize],
        feature_indices: &[usize],
    ) -> Option<(usize, f64, f64)> {
        if indices.len() < self.min_samples_split as usize {
            return None;
        }

        let parent_impurity = self.calculate_impurity(indices);
        let mut best_gain = 0.0;
        let mut best_feature = None;
        let mut best_threshold = 0.0;

        for &feat in feature_indices {
            let mut indexed_values: Vec<(usize, f64)> = indices
                .iter()
                .map(|&i| (i, self.data[i * MAX_FEATURES + feat]))
                .collect();
            indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for i in 0..indexed_values.len() - 1 {
                if indexed_values[i].1 == indexed_values[i + 1].1 {
                    continue;
                }

                let threshold = (indexed_values[i].1 + indexed_values[i + 1].1) / 2.0;

                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .partition(|&&idx| self.data[idx * MAX_FEATURES + feat] <= threshold);

                if left_indices.len() < self.min_samples_leaf as usize
                    || right_indices.len() < self.min_samples_leaf as usize
                {
                    continue;
                }

                let left_impurity = self.calculate_impurity(&left_indices);
                let right_impurity = self.calculate_impurity(&right_indices);

                let gain = parent_impurity
                    - (left_indices.len() as f64 / indices.len() as f64) * left_impurity
                    - (right_indices.len() as f64 / indices.len() as f64) * right_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = Some(feat);
                    best_threshold = threshold;
                }
            }
        }

        best_feature.map(|f| (f, best_threshold, best_gain))
    }

    fn get_majority_class(&self, indices: &[usize]) -> i32 {
        let mut class_count = [0i32; 100];

        for &idx in indices {
            let class_label = self.targets[idx].round() as i32;
            if class_label >= 0 && class_label < 100 {
                class_count[class_label as usize] += 1;
            }
        }

        class_count
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }

    fn get_mean_target(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        indices.iter().map(|&i| self.targets[i]).sum::<f64>() / indices.len() as f64
    }

    fn create_leaf_node(&self, indices: &[usize]) -> TreeNode {
        let impurity = self.calculate_impurity(indices);

        if self.task_type == TaskType::Classification {
            let class_label = self.get_majority_class(indices);
            TreeNode::new_leaf(class_label as f64, class_label, impurity, indices.len())
        } else {
            let prediction = self.get_mean_target(indices);
            TreeNode::new_leaf(prediction, prediction.round() as i32, impurity, indices.len())
        }
    }

    fn should_stop(&self, depth: i32, num_indices: usize, impurity: f64) -> bool {
        depth >= self.max_depth
            || (num_indices as i32) < self.min_samples_split
            || num_indices <= self.min_samples_leaf as usize
            || impurity < 1e-10
    }

    fn build_tree(&mut self, indices: &[usize], depth: i32) -> TreeNode {
        let current_impurity = self.calculate_impurity(indices);

        if self.should_stop(depth, indices.len(), current_impurity) {
            return self.create_leaf_node(indices);
        }

        let feature_indices = self.select_feature_subset();

        let split = self.find_best_split(indices, &feature_indices);
        if split.is_none() {
            return self.create_leaf_node(indices);
        }

        let (best_feature, best_threshold, _) = split.unwrap();

        let (prediction, class_label) = if self.task_type == TaskType::Classification {
            let cl = self.get_majority_class(indices);
            (cl as f64, cl)
        } else {
            let pred = self.get_mean_target(indices);
            (pred, pred.round() as i32)
        };

        let mut node = TreeNode::new_split(
            best_feature as i32,
            best_threshold,
            prediction,
            class_label,
            current_impurity,
            indices.len(),
        );

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&idx| self.data[idx * MAX_FEATURES + best_feature] <= best_threshold);

        let left_impurity = self.calculate_impurity(&left_indices);
        let right_impurity = self.calculate_impurity(&right_indices);

        self.feature_importances[best_feature] += indices.len() as f64 * current_impurity
            - left_indices.len() as f64 * left_impurity
            - right_indices.len() as f64 * right_impurity;

        node.left = Some(Box::new(self.build_tree(&left_indices, depth + 1)));
        node.right = Some(Box::new(self.build_tree(&right_indices, depth + 1)));

        node
    }

    fn flatten_tree(node: &TreeNode, flat: &mut FlatTree, node_idx: &mut usize) {
        if *node_idx >= MAX_NODES {
            return;
        }

        let current_idx = *node_idx;
        *node_idx += 1;

        flat.nodes[current_idx] = FlatTreeNode {
            is_leaf: if node.is_leaf { 1 } else { 0 },
            feature_index: node.feature_index,
            threshold: node.threshold,
            prediction: node.prediction,
            class_label: node.class_label,
            left_child: -1,
            right_child: -1,
        };

        if !node.is_leaf {
            if let Some(ref left) = node.left {
                flat.nodes[current_idx].left_child = *node_idx as i32;
                Self::flatten_tree(left, flat, node_idx);
            }
            if let Some(ref right) = node.right {
                flat.nodes[current_idx].right_child = *node_idx as i32;
                Self::flatten_tree(right, flat, node_idx);
            }
        }
    }

    fn flatten_forest(&self) -> FlattenedForest {
        let mut total_nodes = 0;
        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                total_nodes += tree.num_nodes;
            }
        }

        let mut all_nodes = Vec::with_capacity(total_nodes);
        let mut tree_offsets = Vec::with_capacity(self.num_trees);

        let mut offset = 0i32;
        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                tree_offsets.push(offset);
                for n in 0..tree.num_nodes {
                    all_nodes.push(tree.nodes[n]);
                }
                offset += tree.num_nodes as i32;
            }
        }

        FlattenedForest {
            all_nodes,
            tree_offsets,
            total_nodes,
        }
    }

    pub fn fit(&mut self) {
        for i in 0..MAX_FEATURES {
            self.feature_importances[i] = 0.0;
        }

        for i in 0..self.num_trees {
            self.fit_tree(i);
        }

        self.calculate_feature_importance();
        self.init_backend();
    }

    fn fit_tree(&mut self, tree_index: usize) {
        let mut flat = FlatTree::new();

        let mut sample_indices = Vec::new();
        let mut oob_mask = Vec::new();
        self.bootstrap(&mut sample_indices, &mut oob_mask);

        flat.oob_indices = oob_mask.clone();
        flat.num_oob_indices = oob_mask.iter().filter(|&&x| x).count();

        let root = self.build_tree(&sample_indices, 0);

        let mut node_idx = 0;
        Self::flatten_tree(&root, &mut flat, &mut node_idx);
        flat.num_nodes = node_idx;

        self.trees[tree_index] = Some(flat);
    }

    fn init_backend(&mut self) {
        self.free_backend();
        self.backend = ComputeBackend::detect_and_create(self.backend_kind);
        let flat = self.flatten_forest();
        self.backend.prepare_forest(&flat);
    }

    fn free_backend(&mut self) {
        self.backend = ComputeBackend::None;
    }

    pub fn predict(&self, sample: &[f64]) -> f64 {
        if self.task_type == TaskType::Regression {
            let mut sum = 0.0;
            for t in 0..self.num_trees {
                if let Some(ref tree) = self.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    sum += tree.nodes[node_idx].prediction;
                }
            }
            sum / self.num_trees as f64
        } else {
            let mut votes = [0i32; 100];
            for t in 0..self.num_trees {
                if let Some(ref tree) = self.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    let class_label = tree.nodes[node_idx].class_label;
                    if class_label >= 0 && class_label < 100 {
                        votes[class_label as usize] += 1;
                    }
                }
            }

            votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i as f64)
                .unwrap_or(0.0)
        }
    }

    pub fn predict_class(&self, sample: &[f64]) -> i32 {
        self.predict(sample).round() as i32
    }

    pub fn predict_batch(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let mut predictions = vec![0.0; n_samples];
        for i in 0..n_samples {
            let start = i * self.num_features as usize;
            let end = start + self.num_features as usize;
            predictions[i] = self.predict(&samples[start..end]);
        }
        predictions
    }

    pub fn predict_batch_gpu(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        self.backend.predict_batch(self, samples, n_samples, None)
    }

    pub fn predict_batch_gpu_weighted(
        &self,
        samples: &[f64],
        n_samples: usize,
        weights: &[f64],
    ) -> Vec<f64> {
        self.backend.predict_batch(self, samples, n_samples, Some(weights))
    }

    pub fn calculate_oob_error(&self) -> f64 {
        let mut predictions = vec![0.0; MAX_SAMPLES];
        let mut pred_counts = vec![0i32; MAX_SAMPLES];
        let mut votes = vec![[0i32; 100]; MAX_SAMPLES];

        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                for i in 0..self.num_samples {
                    if tree.oob_indices[i] {
                        let mut sample = vec![0.0; self.num_features as usize];
                        for j in 0..self.num_features as usize {
                            sample[j] = self.data[i * MAX_FEATURES + j];
                        }

                        let pred = self.predict(&sample);
                        if self.task_type == TaskType::Regression {
                            predictions[i] += pred;
                        } else {
                            let j = pred.round() as i32;
                            if j >= 0 && j < 100 {
                                votes[i][j as usize] += 1;
                            }
                        }
                        pred_counts[i] += 1;
                    }
                }
            }
        }

        let mut error = 0.0;
        let mut count = 0;

        for i in 0..self.num_samples {
            if pred_counts[i] > 0 {
                if self.task_type == TaskType::Regression {
                    let pred = predictions[i] / pred_counts[i] as f64;
                    let diff = pred - self.targets[i];
                    error += diff * diff;
                } else {
                    let max_class = votes[i]
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, &v)| v)
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0);
                    if max_class != self.targets[i].round() as i32 {
                        error += 1.0;
                    }
                }
                count += 1;
            }
        }

        if count > 0 { error / count as f64 } else { 0.0 }
    }

    fn calculate_feature_importance(&mut self) {
        let total: f64 = self.feature_importances[..self.num_features as usize].iter().sum();

        if total > 0.0 {
            for i in 0..self.num_features as usize {
                self.feature_importances[i] /= total;
            }
        }
    }

    pub fn get_feature_importance(&self, feature_index: usize) -> f64 {
        if feature_index < self.num_features as usize {
            self.feature_importances[feature_index]
        } else {
            0.0
        }
    }

    pub fn print_feature_importances(&self) {
        println!("Feature Importances:");
        for i in 0..self.num_features as usize {
            println!("  Feature {}: {:.4}", i, self.feature_importances[i]);
        }
    }

    pub fn accuracy(predictions: &[f64], actual: &[f64]) -> f64 {
        let correct = predictions
            .iter()
            .zip(actual.iter())
            .filter(|(&p, &a)| p.round() as i32 == a.round() as i32)
            .count();
        correct as f64 / predictions.len() as f64
    }

    pub fn precision(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let (tp, fp) = predictions.iter().zip(actual.iter()).fold((0, 0), |(tp, fp), (&p, &a)| {
            if p.round() as i32 == positive_class {
                if a.round() as i32 == positive_class { (tp + 1, fp) } else { (tp, fp + 1) }
            } else {
                (tp, fp)
            }
        });
        if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 }
    }

    pub fn recall(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let (tp, fn_count) = predictions.iter().zip(actual.iter()).fold((0, 0), |(tp, fn_c), (&p, &a)| {
            if a.round() as i32 == positive_class {
                if p.round() as i32 == positive_class { (tp + 1, fn_c) } else { (tp, fn_c + 1) }
            } else {
                (tp, fn_c)
            }
        });
        if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 }
    }

    pub fn f1_score(predictions: &[f64], actual: &[f64], positive_class: i32) -> f64 {
        let p = Self::precision(predictions, actual, positive_class);
        let r = Self::recall(predictions, actual, positive_class);
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    }

    pub fn mean_squared_error(predictions: &[f64], actual: &[f64]) -> f64 {
        predictions
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64
    }

    pub fn r_squared(predictions: &[f64], actual: &[f64]) -> f64 {
        let mean: f64 = actual.iter().sum::<f64>() / actual.len() as f64;

        let ss_res: f64 = predictions.iter().zip(actual.iter()).map(|(&p, &a)| (p - a).powi(2)).sum();
        let ss_tot: f64 = actual.iter().map(|&a| (a - mean).powi(2)).sum();

        if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 }
    }

    pub fn print_forest_info(&self) {
        println!("Random Forest Configuration:");
        println!("  Number of Trees: {}", self.num_trees);
        println!("  Max Depth: {}", self.max_depth);
        println!("  Min Samples Leaf: {}", self.min_samples_leaf);
        println!("  Min Samples Split: {}", self.min_samples_split);
        println!("  Max Features: {}", self.max_features);
        println!("  Number of Features: {}", self.num_features);
        println!("  Number of Samples: {}", self.num_samples);
        println!(
            "  Task Type: {}",
            if self.task_type == TaskType::Classification { "Classification" } else { "Regression" }
        );
        println!(
            "  Criterion: {}",
            match self.criterion {
                SplitCriterion::Gini => "Gini",
                SplitCriterion::Entropy => "Entropy",
                SplitCriterion::MSE => "MSE",
                SplitCriterion::VarianceReduction => "Variance Reduction",
            }
        );
        println!("  Compute Backend: {}", self.backend.backend_name());
    }

    pub fn free_forest(&mut self) {
        for i in 0..MAX_TREES {
            self.trees[i] = None;
        }
    }

    pub fn get_num_trees(&self) -> usize { self.num_trees }
    pub fn get_num_features(&self) -> i32 { self.num_features }
    pub fn get_num_samples(&self) -> usize { self.num_samples }
    pub fn get_max_depth_val(&self) -> i32 { self.max_depth }
    pub fn get_task_type(&self) -> TaskType { self.task_type }
    pub fn get_criterion(&self) -> SplitCriterion { self.criterion }

    pub fn load_csv(&mut self, filename: &str, target_column: i32, has_header: bool) -> bool {
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {}: {}", filename, e);
                return false;
            }
        };

        let reader = BufReader::new(file);
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut num_cols = 0;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            if has_header && line_num == 0 {
                continue;
            }
            if line.is_empty() {
                continue;
            }

            let row: Vec<f64> = line
                .split(',')
                .map(|cell| cell.trim().parse::<f64>().unwrap_or(0.0))
                .collect();

            if num_cols == 0 {
                num_cols = row.len();
            }
            if row.len() == num_cols {
                rows.push(row);
            }
        }

        if rows.is_empty() {
            eprintln!("Error: No data loaded from {}", filename);
            return false;
        }

        let n_samples = rows.len().min(MAX_SAMPLES);
        let n_features = (num_cols - 1).min(MAX_FEATURES);
        let target_col = if target_column < 0 { num_cols as i32 - 1 } else { target_column };

        self.num_samples = n_samples;
        self.num_features = n_features as i32;

        if self.max_features == 0 {
            if self.task_type == TaskType::Classification {
                self.max_features = (n_features as f64).sqrt().round() as i32;
            } else {
                self.max_features = (n_features / 3) as i32;
            }
            if self.max_features < 1 {
                self.max_features = 1;
            }
        }

        for i in 0..n_samples {
            let mut feat_idx = 0;
            for j in 0..num_cols {
                if j == target_col as usize {
                    self.targets[i] = rows[i][j];
                } else if feat_idx < n_features {
                    self.data[i * MAX_FEATURES + feat_idx] = rows[i][j];
                    feat_idx += 1;
                }
            }
        }

        println!(
            "Loaded {} samples with {} features from {}",
            n_samples, n_features, filename
        );
        true
    }

    pub fn save_model(&self, filename: &str) -> bool {
        let mut file = match File::create(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for writing: {}", filename, e);
                return false;
            }
        };

        let magic = b"RFRS";
        let version: i32 = 1;

        macro_rules! write_val {
            ($val:expr) => {
                if file.write_all(&$val.to_le_bytes()).is_err() {
                    return false;
                }
            };
        }

        if file.write_all(magic).is_err() { return false; }
        write_val!(version);
        write_val!(self.num_trees as i32);
        write_val!(self.max_depth);
        write_val!(self.min_samples_leaf);
        write_val!(self.min_samples_split);
        write_val!(self.max_features);
        write_val!(self.num_features);
        write_val!(self.num_samples as i32);
        write_val!(self.task_type as i32);
        write_val!(self.criterion as i32);

        for i in 0..MAX_FEATURES {
            write_val!(self.feature_importances[i]);
        }

        for t in 0..self.num_trees {
            if let Some(ref tree) = self.trees[t] {
                write_val!(tree.num_nodes as i32);
                write_val!(tree.num_oob_indices as i32);

                for n in 0..tree.num_nodes {
                    let node = &tree.nodes[n];
                    write_val!(node.is_leaf);
                    write_val!(node.feature_index);
                    write_val!(node.threshold);
                    write_val!(node.prediction);
                    write_val!(node.class_label);
                    write_val!(node.left_child);
                    write_val!(node.right_child);
                }

                for i in 0..MAX_SAMPLES {
                    let b: u8 = if tree.oob_indices[i] { 1 } else { 0 };
                    if file.write_all(&[b]).is_err() { return false; }
                }
            }
        }

        println!("Model saved to {}", filename);
        true
    }

    pub fn load_model(&mut self, filename: &str) -> bool {
        let mut file = match File::open(filename) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for reading: {}", filename, e);
                return false;
            }
        };

        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_err() { return false; }
        if &magic != b"RFRS" {
            eprintln!("Error: Invalid model file format");
            return false;
        }

        self.free_forest();
        self.free_backend();

        macro_rules! read_i32 {
            () => {{
                let mut buf = [0u8; 4];
                if file.read_exact(&mut buf).is_err() { return false; }
                i32::from_le_bytes(buf)
            }};
        }

        macro_rules! read_f64 {
            () => {{
                let mut buf = [0u8; 8];
                if file.read_exact(&mut buf).is_err() { return false; }
                f64::from_le_bytes(buf)
            }};
        }

        let _version = read_i32!();
        self.num_trees = read_i32!() as usize;
        self.max_depth = read_i32!();
        self.min_samples_leaf = read_i32!();
        self.min_samples_split = read_i32!();
        self.max_features = read_i32!();
        self.num_features = read_i32!();
        self.num_samples = read_i32!() as usize;
        self.task_type = match read_i32!() {
            1 => TaskType::Regression,
            _ => TaskType::Classification,
        };
        self.criterion = match read_i32!() {
            1 => SplitCriterion::Entropy,
            2 => SplitCriterion::MSE,
            3 => SplitCriterion::VarianceReduction,
            _ => SplitCriterion::Gini,
        };

        for i in 0..MAX_FEATURES {
            self.feature_importances[i] = read_f64!();
        }

        for t in 0..self.num_trees {
            let mut tree = FlatTree::new();
            tree.num_nodes = read_i32!() as usize;
            tree.num_oob_indices = read_i32!() as usize;

            for n in 0..tree.num_nodes {
                tree.nodes[n].is_leaf = read_i32!();
                tree.nodes[n].feature_index = read_i32!();
                tree.nodes[n].threshold = read_f64!();
                tree.nodes[n].prediction = read_f64!();
                tree.nodes[n].class_label = read_i32!();
                tree.nodes[n].left_child = read_i32!();
                tree.nodes[n].right_child = read_i32!();
            }

            for i in 0..MAX_SAMPLES {
                let mut b = [0u8; 1];
                if file.read_exact(&mut b).is_err() { return false; }
                tree.oob_indices[i] = b[0] != 0;
            }

            self.trees[t] = Some(tree);
        }

        println!("Model loaded from {}", filename);
        self.init_backend();
        true
    }

    pub fn predict_csv(&self, input_file: &str, output_file: &str, has_header: bool) -> bool {
        let in_file = match File::open(input_file) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {}: {}", input_file, e);
                return false;
            }
        };

        let mut out_file = match File::create(output_file) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Cannot open file {} for writing: {}", output_file, e);
                return false;
            }
        };

        let reader = BufReader::new(in_file);

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            if has_header && line_num == 0 {
                writeln!(out_file, "{},prediction", line).ok();
                continue;
            }
            if line.is_empty() {
                continue;
            }

            let row: Vec<f64> = line
                .split(',')
                .map(|cell| cell.trim().parse::<f64>().unwrap_or(0.0))
                .collect();

            let mut sample = vec![0.0; self.num_features as usize];
            for j in 0..self.num_features as usize {
                if j < row.len() {
                    sample[j] = row[j];
                }
            }

            let pred = self.predict(&sample);
            writeln!(out_file, "{},{:.4}", line, pred).ok();
        }

        println!("Predictions saved to {}", output_file);
        true
    }

    pub fn add_new_tree(&mut self) {
        if self.num_trees < MAX_TREES {
            self.fit_tree(self.num_trees);
            self.num_trees += 1;
            self.init_backend();
        }
    }

    pub fn remove_tree_at(&mut self, tree_id: usize) {
        if tree_id < self.num_trees && self.num_trees > 1 {
            for i in tree_id..self.num_trees - 1 {
                self.trees[i] = self.trees[i + 1].take();
            }
            self.trees[self.num_trees - 1] = None;
            self.num_trees -= 1;
            self.init_backend();
        }
    }

    pub fn retrain_tree_at(&mut self, tree_id: usize) {
        if tree_id < self.num_trees {
            self.fit_tree(tree_id);
            self.init_backend();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRandomForestFacade
// ═══════════════════════════════════════════════════════════════════

pub struct TRandomForestFacade {
    pub forest: TRandomForest,
    pub forest_initialized: bool,
    pub current_aggregation: AggregationMethod,
    pub tree_weights: Vec<f64>,
    pub feature_enabled: Vec<bool>,
}

impl TRandomForestFacade {
    pub fn new() -> Self {
        Self {
            forest: TRandomForest::new(),
            forest_initialized: false,
            current_aggregation: AggregationMethod::MajorityVote,
            tree_weights: vec![1.0; MAX_TREES],
            feature_enabled: vec![true; MAX_FEATURES],
        }
    }

    pub fn init_forest(&mut self) {
        self.forest_initialized = true;
    }

    pub fn set_backend(&mut self, kind: BackendKind) {
        self.forest.set_backend_kind(kind);
    }

    pub fn set_hyperparameter(&mut self, param_name: &str, value: i32) {
        match param_name {
            "n_estimators" => self.forest.set_num_trees(value as usize),
            "max_depth" => self.forest.set_max_depth(value),
            "min_samples_leaf" => self.forest.set_min_samples_leaf(value),
            "min_samples_split" => self.forest.set_min_samples_split(value),
            "max_features" => self.forest.set_max_features(value),
            _ => {}
        }
    }

    #[allow(dead_code)]
    pub fn set_hyperparameter_float(&mut self, _param_name: &str, _value: f64) {}

    #[allow(dead_code)]
    pub fn get_hyperparameter(&self, _param_name: &str) -> i32 { 0 }

    pub fn set_task_type(&mut self, t: TaskType) { self.forest.set_task_type(t); }
    pub fn set_criterion(&mut self, c: SplitCriterion) { self.forest.set_criterion(c); }
    pub fn print_hyperparameters(&self) { self.forest.print_forest_info(); }

    pub fn load_csv(&mut self, filename: &str) -> bool { self.forest.load_csv(filename, -1, true) }
    pub fn train(&mut self) { self.forest.fit(); }

    pub fn inspect_tree(&self, tree_id: usize) -> TreeInfo {
        let mut info = TreeInfo {
            tree_id: tree_id as i32,
            features_used: vec![false; MAX_FEATURES],
            ..Default::default()
        };

        if let Some(ref tree) = self.forest.trees.get(tree_id).and_then(|t| t.as_ref()) {
            info.num_nodes = tree.num_nodes as i32;

            let mut num_leaves = 0;

            for n in 0..tree.num_nodes {
                let node = &tree.nodes[n];
                if node.is_leaf != 0 {
                    num_leaves += 1;
                } else {
                    if node.feature_index >= 0 && (node.feature_index as usize) < MAX_FEATURES {
                        info.features_used[node.feature_index as usize] = true;
                    }
                }
            }

            fn calc_depth(nodes: &[FlatTreeNode], idx: usize, depth: i32) -> i32 {
                if idx >= nodes.len() || nodes[idx].is_leaf != 0 {
                    return depth;
                }
                let left_depth = if nodes[idx].left_child >= 0 {
                    calc_depth(nodes, nodes[idx].left_child as usize, depth + 1)
                } else { depth };
                let right_depth = if nodes[idx].right_child >= 0 {
                    calc_depth(nodes, nodes[idx].right_child as usize, depth + 1)
                } else { depth };
                left_depth.max(right_depth)
            }

            let max_depth = calc_depth(&tree.nodes, 0, 0);

            info.max_depth = max_depth;
            info.num_leaves = num_leaves;
            info.num_features_used = info.features_used.iter().filter(|&&x| x).count() as i32;
        }

        info
    }

    pub fn print_tree_info(&self, tree_id: usize) {
        let info = self.inspect_tree(tree_id);
        println!("Tree {}: {} nodes, max depth: {}, leaves: {}",
            tree_id, info.num_nodes, info.max_depth, info.num_leaves);
    }

    pub fn print_tree_structure(&self, tree_id: usize) {
        println!("Tree {} structure:", tree_id);
        if let Some(ref tree) = self.forest.trees.get(tree_id).and_then(|t| t.as_ref()) {
            fn print_node(nodes: &[FlatTreeNode], idx: usize, indent: usize) {
                if idx >= nodes.len() { return; }
                let node = &nodes[idx];
                let prefix = "  ".repeat(indent);
                if node.is_leaf != 0 {
                    println!("{}[Leaf] prediction={:.4}, class={}", prefix, node.prediction, node.class_label);
                } else {
                    println!("{}[Node] feature={}, threshold={:.4}", prefix, node.feature_index, node.threshold);
                    if node.left_child >= 0 {
                        println!("{}  Left:", prefix);
                        print_node(nodes, node.left_child as usize, indent + 2);
                    }
                    if node.right_child >= 0 {
                        println!("{}  Right:", prefix);
                        print_node(nodes, node.right_child as usize, indent + 2);
                    }
                }
            }
            print_node(&tree.nodes, 0, 0);
        }
    }

    pub fn add_tree(&mut self) { self.forest.add_new_tree(); }
    pub fn remove_tree(&mut self, tree_id: usize) { self.forest.remove_tree_at(tree_id); }
    pub fn replace_tree(&mut self, tree_id: usize) { self.forest.retrain_tree_at(tree_id); }
    pub fn retrain_tree(&mut self, tree_id: usize) { self.forest.retrain_tree_at(tree_id); }
    pub fn get_num_trees(&self) -> usize { self.forest.get_num_trees() }

    pub fn enable_feature(&mut self, feature_index: usize) {
        if feature_index < MAX_FEATURES {
            self.feature_enabled[feature_index] = true;
        }
    }

    pub fn disable_feature(&mut self, feature_index: usize) {
        if feature_index < MAX_FEATURES {
            self.feature_enabled[feature_index] = false;
        }
    }

    pub fn reset_features(&mut self) {
        for i in 0..MAX_FEATURES {
            self.feature_enabled[i] = true;
        }
    }

    pub fn print_feature_usage(&self) {
        println!("Feature Usage Summary:");
        let mut usage = vec![0i32; MAX_FEATURES];

        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                for n in 0..tree.num_nodes {
                    let node = &tree.nodes[n];
                    if node.is_leaf == 0 && node.feature_index >= 0 && (node.feature_index as usize) < MAX_FEATURES {
                        usage[node.feature_index as usize] += 1;
                    }
                }
            }
        }

        for i in 0..self.forest.num_features as usize {
            let enabled = if self.feature_enabled[i] { "enabled" } else { "disabled" };
            println!("  Feature {}: {} splits ({})", i, usage[i], enabled);
        }
    }

    pub fn print_feature_importances(&self) { self.forest.print_feature_importances(); }

    pub fn set_aggregation_method(&mut self, method: AggregationMethod) {
        self.current_aggregation = method;
    }

    pub fn get_aggregation_method(&self) -> AggregationMethod { self.current_aggregation }

    pub fn set_tree_weight(&mut self, tree_id: usize, weight: f64) {
        if tree_id < MAX_TREES {
            self.tree_weights[tree_id] = weight;
        }
    }

    pub fn get_tree_weight(&self, tree_id: usize) -> f64 {
        if tree_id < MAX_TREES { self.tree_weights[tree_id] } else { 1.0 }
    }

    pub fn reset_tree_weights(&mut self) {
        for i in 0..MAX_TREES {
            self.tree_weights[i] = 1.0;
        }
    }

    pub fn predict(&self, sample: &[f64]) -> f64 {
        match self.current_aggregation {
            AggregationMethod::MajorityVote | AggregationMethod::Mean => {
                self.forest.predict(sample)
            }
            AggregationMethod::WeightedVote | AggregationMethod::WeightedMean => {
                self.predict_weighted(sample)
            }
        }
    }

    fn predict_weighted(&self, sample: &[f64]) -> f64 {
        if self.forest.task_type == TaskType::Regression {
            let mut sum = 0.0;
            let mut total_weight = 0.0;
            for t in 0..self.forest.num_trees {
                if let Some(ref tree) = self.forest.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    sum += tree.nodes[node_idx].prediction * self.tree_weights[t];
                    total_weight += self.tree_weights[t];
                }
            }
            if total_weight > 0.0 { sum / total_weight } else { 0.0 }
        } else {
            let mut votes = [0.0f64; 100];
            for t in 0..self.forest.num_trees {
                if let Some(ref tree) = self.forest.trees[t] {
                    let mut node_idx = 0;
                    while tree.nodes[node_idx].is_leaf == 0 {
                        if sample[tree.nodes[node_idx].feature_index as usize]
                            <= tree.nodes[node_idx].threshold
                        {
                            node_idx = tree.nodes[node_idx].left_child as usize;
                        } else {
                            node_idx = tree.nodes[node_idx].right_child as usize;
                        }
                    }
                    let class_label = tree.nodes[node_idx].class_label;
                    if class_label >= 0 && class_label < 100 {
                        votes[class_label as usize] += self.tree_weights[t];
                    }
                }
            }

            votes
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as f64)
                .unwrap_or(0.0)
        }
    }

    pub fn predict_class(&self, sample: &[f64]) -> i32 {
        self.predict(sample).round() as i32
    }

    pub fn predict_batch(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        let mut predictions = vec![0.0; n_samples];
        for i in 0..n_samples {
            let start = i * self.forest.num_features as usize;
            let end = start + self.forest.num_features as usize;
            predictions[i] = self.predict(&samples[start..end]);
        }
        predictions
    }

    pub fn predict_batch_gpu(&self, samples: &[f64], n_samples: usize) -> Vec<f64> {
        self.forest.predict_batch_gpu(samples, n_samples)
    }

    pub fn track_sample(&self, sample_index: usize) -> SampleTrackInfo {
        let mut info = SampleTrackInfo {
            sample_index: sample_index as i32,
            trees_influenced: vec![false; MAX_TREES],
            oob_trees: vec![false; MAX_TREES],
            predictions: vec![0.0; MAX_TREES],
            ..Default::default()
        };

        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                if sample_index < tree.oob_indices.len() {
                    if tree.oob_indices[sample_index] {
                        info.oob_trees[t] = true;
                        info.num_oob_trees += 1;
                    } else {
                        info.trees_influenced[t] = true;
                        info.num_trees_influenced += 1;
                    }
                }
            }
        }

        info
    }

    pub fn print_sample_tracking(&self, sample_index: usize) {
        let info = self.track_sample(sample_index);
        println!("Sample {} tracking:", sample_index);
        println!("  Trees influenced: {}", info.num_trees_influenced);
        println!("  OOB trees: {}", info.num_oob_trees);
    }

    pub fn oob_error_summary(&self) -> Vec<OOBTreeInfo> {
        let mut summary = Vec::new();
        for t in 0..self.forest.num_trees {
            if let Some(ref tree) = self.forest.trees[t] {
                let num_oob = tree.oob_indices.iter().take(self.forest.num_samples).filter(|&&x| x).count();
                summary.push(OOBTreeInfo {
                    tree_id: t as i32,
                    num_oob_samples: num_oob as i32,
                    oob_error: 0.0,
                    oob_accuracy: 0.0,
                });
            }
        }
        summary
    }

    pub fn print_oob_summary(&self) {
        println!("OOB Error Summary:");
        let summary = self.oob_error_summary();
        for info in &summary {
            println!("  Tree {}: {} OOB samples", info.tree_id, info.num_oob_samples);
        }
        println!("Global OOB Error: {:.4}", self.forest.calculate_oob_error());
    }

    pub fn get_global_oob_error(&self) -> f64 { self.forest.calculate_oob_error() }

    pub fn accuracy(&self, predictions: &[f64], actual: &[f64]) -> f64 {
        TRandomForest::accuracy(predictions, actual)
    }

    pub fn mean_squared_error(&self, predictions: &[f64], actual: &[f64]) -> f64 {
        TRandomForest::mean_squared_error(predictions, actual)
    }

    pub fn highlight_misclassified(&self, predictions: &[f64], actual: &[f64]) {
        println!("Misclassified Samples:");
        for (i, (&p, &a)) in predictions.iter().zip(actual.iter()).enumerate() {
            if p.round() as i32 != a.round() as i32 {
                println!("  Sample {}: predicted={}, actual={}", i, p.round() as i32, a.round() as i32);
            }
        }
    }

    pub fn save_model(&self, filename: &str) -> bool { self.forest.save_model(filename) }
    pub fn load_model(&mut self, filename: &str) -> bool { self.forest.load_model(filename) }
    pub fn print_forest_info(&self) { self.forest.print_forest_info(); }
}

// ═══════════════════════════════════════════════════════════════════
// PyO3 Python Bindings
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "pyo3")]
mod python {
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use super::*;

    #[pyclass(name = "RandomForest")]
    pub struct PyRandomForest {
        inner: TRandomForest,
    }

    #[pymethods]
    impl PyRandomForest {
        #[new]
        #[pyo3(signature = (seed=None))]
        fn new(seed: Option<u64>) -> Self {
            let inner = match seed {
                Some(s) => TRandomForest::new_with_seed(s),
                None => TRandomForest::new(),
            };
            Self { inner }
        }

        fn set_num_trees(&mut self, n: usize) {
            self.inner.set_num_trees(n);
        }

        fn set_max_depth(&mut self, d: i32) {
            self.inner.set_max_depth(d);
        }

        fn set_min_samples_leaf(&mut self, m: i32) {
            self.inner.set_min_samples_leaf(m);
        }

        fn set_min_samples_split(&mut self, m: i32) {
            self.inner.set_min_samples_split(m);
        }

        fn set_max_features(&mut self, m: i32) {
            self.inner.set_max_features(m);
        }

        fn set_task_type(&mut self, task: &str) -> PyResult<()> {
            match task.to_lowercase().as_str() {
                "classification" | "class" => self.inner.set_task_type(TaskType::Classification),
                "regression" | "reg" => self.inner.set_task_type(TaskType::Regression),
                _ => return Err(PyValueError::new_err("task must be 'classification' or 'regression'")),
            }
            Ok(())
        }

        fn set_criterion(&mut self, criterion: &str) -> PyResult<()> {
            match criterion.to_lowercase().as_str() {
                "gini" => self.inner.set_criterion(SplitCriterion::Gini),
                "entropy" => self.inner.set_criterion(SplitCriterion::Entropy),
                "mse" => self.inner.set_criterion(SplitCriterion::MSE),
                "variance" | "var" => self.inner.set_criterion(SplitCriterion::VarianceReduction),
                _ => return Err(PyValueError::new_err("criterion must be 'gini', 'entropy', 'mse', or 'variance'")),
            }
            Ok(())
        }

        fn set_random_seed(&mut self, seed: u64) {
            self.inner.set_random_seed(seed);
        }

        fn set_backend(&mut self, backend: &str) {
            self.inner.set_backend_kind(BackendKind::from_str(backend));
        }

        fn load_data(&mut self, data: Vec<f64>, targets: Vec<f64>, n_samples: usize, n_features: usize) {
            self.inner.load_data(&data, &targets, n_samples, n_features);
        }

        fn load_csv(&mut self, filename: &str, target_column: i32, has_header: bool) -> bool {
            self.inner.load_csv(filename, target_column, has_header)
        }

        fn fit(&mut self) {
            self.inner.fit();
        }

        fn predict(&self, sample: Vec<f64>) -> f64 {
            self.inner.predict(&sample)
        }

        fn predict_class(&self, sample: Vec<f64>) -> i32 {
            self.inner.predict_class(&sample)
        }

        fn predict_batch(&self, samples: Vec<f64>, n_samples: usize) -> Vec<f64> {
            self.inner.predict_batch(&samples, n_samples)
        }

        fn predict_batch_gpu(&self, samples: Vec<f64>, n_samples: usize) -> Vec<f64> {
            self.inner.predict_batch_gpu(&samples, n_samples)
        }

        fn predict_batch_gpu_weighted(&self, samples: Vec<f64>, n_samples: usize, weights: Vec<f64>) -> Vec<f64> {
            self.inner.predict_batch_gpu_weighted(&samples, n_samples, &weights)
        }

        fn calculate_oob_error(&self) -> f64 {
            self.inner.calculate_oob_error()
        }

        fn get_feature_importance(&self, feature_index: usize) -> f64 {
            self.inner.get_feature_importance(feature_index)
        }

        fn get_feature_importances(&self) -> Vec<f64> {
            (0..self.inner.num_features as usize)
                .map(|i| self.inner.get_feature_importance(i))
                .collect()
        }

        fn print_feature_importances(&self) {
            self.inner.print_feature_importances();
        }

        fn print_forest_info(&self) {
            self.inner.print_forest_info();
        }

        fn save_model(&self, filename: &str) -> bool {
            self.inner.save_model(filename)
        }

        fn load_model(&mut self, filename: &str) -> bool {
            self.inner.load_model(filename)
        }

        fn predict_csv(&self, input_file: &str, output_file: &str, has_header: bool) -> bool {
            self.inner.predict_csv(input_file, output_file, has_header)
        }

        fn add_new_tree(&mut self) {
            self.inner.add_new_tree();
        }

        fn remove_tree_at(&mut self, tree_id: usize) {
            self.inner.remove_tree_at(tree_id);
        }

        fn retrain_tree_at(&mut self, tree_id: usize) {
            self.inner.retrain_tree_at(tree_id);
        }

        fn get_num_trees(&self) -> usize {
            self.inner.get_num_trees()
        }

        fn get_num_features(&self) -> i32 {
            self.inner.get_num_features()
        }

        fn get_num_samples(&self) -> usize {
            self.inner.get_num_samples()
        }

        fn get_max_depth(&self) -> i32 {
            self.inner.get_max_depth_val()
        }

        #[staticmethod]
        fn accuracy(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::accuracy(&predictions, &actual)
        }

        #[staticmethod]
        fn precision(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::precision(&predictions, &actual, positive_class)
        }

        #[staticmethod]
        fn recall(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::recall(&predictions, &actual, positive_class)
        }

        #[staticmethod]
        fn f1_score(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::f1_score(&predictions, &actual, positive_class)
        }

        #[staticmethod]
        fn mean_squared_error(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::mean_squared_error(&predictions, &actual)
        }

        #[staticmethod]
        fn r_squared(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::r_squared(&predictions, &actual)
        }
    }

    #[pyclass(name = "RandomForestFacade")]
    pub struct PyRandomForestFacade {
        inner: TRandomForestFacade,
    }

    #[pymethods]
    impl PyRandomForestFacade {
        #[new]
        fn new() -> Self {
            Self { inner: TRandomForestFacade::new() }
        }

        fn init_forest(&mut self) {
            self.inner.init_forest();
        }

        fn set_backend(&mut self, backend: &str) {
            self.inner.set_backend(BackendKind::from_str(backend));
        }

        fn set_hyperparameter(&mut self, param_name: &str, value: i32) {
            self.inner.set_hyperparameter(param_name, value);
        }

        fn set_task_type(&mut self, task: &str) -> PyResult<()> {
            match task.to_lowercase().as_str() {
                "classification" | "class" => self.inner.set_task_type(TaskType::Classification),
                "regression" | "reg" => self.inner.set_task_type(TaskType::Regression),
                _ => return Err(PyValueError::new_err("task must be 'classification' or 'regression'")),
            }
            Ok(())
        }

        fn set_criterion(&mut self, criterion: &str) -> PyResult<()> {
            match criterion.to_lowercase().as_str() {
                "gini" => self.inner.set_criterion(SplitCriterion::Gini),
                "entropy" => self.inner.set_criterion(SplitCriterion::Entropy),
                "mse" => self.inner.set_criterion(SplitCriterion::MSE),
                "variance" | "var" => self.inner.set_criterion(SplitCriterion::VarianceReduction),
                _ => return Err(PyValueError::new_err("criterion must be 'gini', 'entropy', 'mse', or 'variance'")),
            }
            Ok(())
        }

        fn print_hyperparameters(&self) {
            self.inner.print_hyperparameters();
        }

        fn load_csv(&mut self, filename: &str) -> bool {
            self.inner.load_csv(filename)
        }

        fn train(&mut self) {
            self.inner.train();
        }

        fn predict(&self, sample: Vec<f64>) -> f64 {
            self.inner.predict(&sample)
        }

        fn predict_class(&self, sample: Vec<f64>) -> i32 {
            self.inner.predict_class(&sample)
        }

        fn predict_batch(&self, samples: Vec<f64>, n_samples: usize) -> Vec<f64> {
            self.inner.predict_batch(&samples, n_samples)
        }

        fn predict_batch_gpu(&self, samples: Vec<f64>, n_samples: usize) -> Vec<f64> {
            self.inner.predict_batch_gpu(&samples, n_samples)
        }

        fn inspect_tree(&self, tree_id: usize) -> PyResult<(i32, i32, i32, i32)> {
            let info = self.inner.inspect_tree(tree_id);
            Ok((info.num_nodes, info.max_depth, info.num_leaves, info.num_features_used))
        }

        fn print_tree_info(&self, tree_id: usize) {
            self.inner.print_tree_info(tree_id);
        }

        fn print_tree_structure(&self, tree_id: usize) {
            self.inner.print_tree_structure(tree_id);
        }

        fn add_tree(&mut self) {
            self.inner.add_tree();
        }

        fn remove_tree(&mut self, tree_id: usize) {
            self.inner.remove_tree(tree_id);
        }

        fn replace_tree(&mut self, tree_id: usize) {
            self.inner.replace_tree(tree_id);
        }

        fn retrain_tree(&mut self, tree_id: usize) {
            self.inner.retrain_tree(tree_id);
        }

        fn get_num_trees(&self) -> usize {
            self.inner.get_num_trees()
        }

        fn enable_feature(&mut self, feature_index: usize) {
            self.inner.enable_feature(feature_index);
        }

        fn disable_feature(&mut self, feature_index: usize) {
            self.inner.disable_feature(feature_index);
        }

        fn reset_features(&mut self) {
            self.inner.reset_features();
        }

        fn print_feature_usage(&self) {
            self.inner.print_feature_usage();
        }

        fn print_feature_importances(&self) {
            self.inner.print_feature_importances();
        }

        fn set_aggregation_method(&mut self, method: &str) -> PyResult<()> {
            match method.to_lowercase().as_str() {
                "majority" | "majority-vote" => self.inner.set_aggregation_method(AggregationMethod::MajorityVote),
                "weighted" | "weighted-vote" => self.inner.set_aggregation_method(AggregationMethod::WeightedVote),
                "mean" => self.inner.set_aggregation_method(AggregationMethod::Mean),
                "weighted-mean" => self.inner.set_aggregation_method(AggregationMethod::WeightedMean),
                _ => return Err(PyValueError::new_err("method must be 'majority', 'weighted', 'mean', or 'weighted-mean'")),
            }
            Ok(())
        }

        fn set_tree_weight(&mut self, tree_id: usize, weight: f64) {
            self.inner.set_tree_weight(tree_id, weight);
        }

        fn get_tree_weight(&self, tree_id: usize) -> f64 {
            self.inner.get_tree_weight(tree_id)
        }

        fn reset_tree_weights(&mut self) {
            self.inner.reset_tree_weights();
        }

        fn track_sample(&self, sample_index: usize) -> (i32, i32) {
            let info = self.inner.track_sample(sample_index);
            (info.num_trees_influenced, info.num_oob_trees)
        }

        fn print_sample_tracking(&self, sample_index: usize) {
            self.inner.print_sample_tracking(sample_index);
        }

        fn print_oob_summary(&self) {
            self.inner.print_oob_summary();
        }

        fn get_global_oob_error(&self) -> f64 {
            self.inner.get_global_oob_error()
        }

        fn highlight_misclassified(&self, predictions: Vec<f64>, actual: Vec<f64>) {
            self.inner.highlight_misclassified(&predictions, &actual);
        }

        fn save_model(&self, filename: &str) -> bool {
            self.inner.save_model(filename)
        }

        fn load_model(&mut self, filename: &str) -> bool {
            self.inner.load_model(filename)
        }

        fn print_forest_info(&self) {
            self.inner.print_forest_info();
        }
    }

    #[pymodule]
    fn facaded_random_forest(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyRandomForest>()?;
        m.add_class::<PyRandomForestFacade>()?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Node.js (napi-rs) Bindings
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "napi")]
mod node {
    use napi_derive::napi;
    use super::*;

    #[napi(js_name = "RandomForest")]
    pub struct JsRandomForest {
        inner: TRandomForest,
    }

    #[napi]
    impl JsRandomForest {
        #[napi(constructor)]
        pub fn new(seed: Option<u32>) -> Self {
            let inner = match seed {
                Some(s) => TRandomForest::new_with_seed(s as u64),
                None => TRandomForest::new(),
            };
            Self { inner }
        }

        #[napi]
        pub fn set_num_trees(&mut self, n: u32) {
            self.inner.set_num_trees(n as usize);
        }

        #[napi]
        pub fn set_max_depth(&mut self, d: i32) {
            self.inner.set_max_depth(d);
        }

        #[napi]
        pub fn set_min_samples_leaf(&mut self, m: i32) {
            self.inner.set_min_samples_leaf(m);
        }

        #[napi]
        pub fn set_min_samples_split(&mut self, m: i32) {
            self.inner.set_min_samples_split(m);
        }

        #[napi]
        pub fn set_max_features(&mut self, m: i32) {
            self.inner.set_max_features(m);
        }

        #[napi]
        pub fn set_task_type(&mut self, task: String) -> napi::Result<()> {
            match task.to_lowercase().as_str() {
                "classification" | "class" => self.inner.set_task_type(TaskType::Classification),
                "regression" | "reg" => self.inner.set_task_type(TaskType::Regression),
                _ => return Err(napi::Error::from_reason("task must be 'classification' or 'regression'")),
            }
            Ok(())
        }

        #[napi]
        pub fn set_criterion(&mut self, criterion: String) -> napi::Result<()> {
            match criterion.to_lowercase().as_str() {
                "gini" => self.inner.set_criterion(SplitCriterion::Gini),
                "entropy" => self.inner.set_criterion(SplitCriterion::Entropy),
                "mse" => self.inner.set_criterion(SplitCriterion::MSE),
                "variance" | "var" => self.inner.set_criterion(SplitCriterion::VarianceReduction),
                _ => return Err(napi::Error::from_reason("criterion must be 'gini', 'entropy', 'mse', or 'variance'")),
            }
            Ok(())
        }

        #[napi]
        pub fn set_random_seed(&mut self, seed: u32) {
            self.inner.set_random_seed(seed as u64);
        }

        #[napi]
        pub fn set_backend(&mut self, backend: String) {
            self.inner.set_backend_kind(BackendKind::from_str(&backend));
        }

        #[napi]
        pub fn load_data(&mut self, data: Vec<f64>, targets: Vec<f64>, n_samples: u32, n_features: u32) {
            self.inner.load_data(&data, &targets, n_samples as usize, n_features as usize);
        }

        #[napi]
        pub fn load_csv(&mut self, filename: String, target_column: i32, has_header: bool) -> bool {
            self.inner.load_csv(&filename, target_column, has_header)
        }

        #[napi]
        pub fn fit(&mut self) {
            self.inner.fit();
        }

        #[napi]
        pub fn predict(&self, sample: Vec<f64>) -> f64 {
            self.inner.predict(&sample)
        }

        #[napi]
        pub fn predict_class(&self, sample: Vec<f64>) -> i32 {
            self.inner.predict_class(&sample)
        }

        #[napi]
        pub fn predict_batch(&self, samples: Vec<f64>, n_samples: u32) -> Vec<f64> {
            self.inner.predict_batch(&samples, n_samples as usize)
        }

        #[napi]
        pub fn predict_batch_gpu(&self, samples: Vec<f64>, n_samples: u32) -> Vec<f64> {
            self.inner.predict_batch_gpu(&samples, n_samples as usize)
        }

        #[napi]
        pub fn predict_batch_gpu_weighted(&self, samples: Vec<f64>, n_samples: u32, weights: Vec<f64>) -> Vec<f64> {
            self.inner.predict_batch_gpu_weighted(&samples, n_samples as usize, &weights)
        }

        #[napi]
        pub fn calculate_oob_error(&self) -> f64 {
            self.inner.calculate_oob_error()
        }

        #[napi]
        pub fn get_feature_importance(&self, feature_index: u32) -> f64 {
            self.inner.get_feature_importance(feature_index as usize)
        }

        #[napi]
        pub fn get_feature_importances(&self) -> Vec<f64> {
            (0..self.inner.num_features as usize)
                .map(|i| self.inner.get_feature_importance(i))
                .collect()
        }

        #[napi]
        pub fn print_feature_importances(&self) {
            self.inner.print_feature_importances();
        }

        #[napi]
        pub fn print_forest_info(&self) {
            self.inner.print_forest_info();
        }

        #[napi]
        pub fn save_model(&self, filename: String) -> bool {
            self.inner.save_model(&filename)
        }

        #[napi]
        pub fn load_model(&mut self, filename: String) -> bool {
            self.inner.load_model(&filename)
        }

        #[napi]
        pub fn predict_csv(&self, input_file: String, output_file: String, has_header: bool) -> bool {
            self.inner.predict_csv(&input_file, &output_file, has_header)
        }

        #[napi]
        pub fn add_new_tree(&mut self) {
            self.inner.add_new_tree();
        }

        #[napi]
        pub fn remove_tree_at(&mut self, tree_id: u32) {
            self.inner.remove_tree_at(tree_id as usize);
        }

        #[napi]
        pub fn retrain_tree_at(&mut self, tree_id: u32) {
            self.inner.retrain_tree_at(tree_id as usize);
        }

        #[napi]
        pub fn get_num_trees(&self) -> u32 {
            self.inner.get_num_trees() as u32
        }

        #[napi]
        pub fn get_num_features(&self) -> i32 {
            self.inner.get_num_features()
        }

        #[napi]
        pub fn get_num_samples(&self) -> u32 {
            self.inner.get_num_samples() as u32
        }

        #[napi]
        pub fn get_max_depth(&self) -> i32 {
            self.inner.get_max_depth_val()
        }

        #[napi]
        pub fn accuracy(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::accuracy(&predictions, &actual)
        }

        #[napi]
        pub fn precision(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::precision(&predictions, &actual, positive_class)
        }

        #[napi]
        pub fn recall(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::recall(&predictions, &actual, positive_class)
        }

        #[napi]
        pub fn f1_score(predictions: Vec<f64>, actual: Vec<f64>, positive_class: i32) -> f64 {
            TRandomForest::f1_score(&predictions, &actual, positive_class)
        }

        #[napi]
        pub fn mean_squared_error(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::mean_squared_error(&predictions, &actual)
        }

        #[napi]
        pub fn r_squared(predictions: Vec<f64>, actual: Vec<f64>) -> f64 {
            TRandomForest::r_squared(&predictions, &actual)
        }
    }

    #[napi(js_name = "RandomForestFacade")]
    pub struct JsRandomForestFacade {
        inner: TRandomForestFacade,
    }

    #[napi]
    impl JsRandomForestFacade {
        #[napi(constructor)]
        pub fn new() -> Self {
            Self { inner: TRandomForestFacade::new() }
        }

        #[napi]
        pub fn init_forest(&mut self) {
            self.inner.init_forest();
        }

        #[napi]
        pub fn set_backend(&mut self, backend: String) {
            self.inner.set_backend(BackendKind::from_str(&backend));
        }

        #[napi]
        pub fn set_hyperparameter(&mut self, param_name: String, value: i32) {
            self.inner.set_hyperparameter(&param_name, value);
        }

        #[napi]
        pub fn set_task_type(&mut self, task: String) -> napi::Result<()> {
            match task.to_lowercase().as_str() {
                "classification" | "class" => self.inner.set_task_type(TaskType::Classification),
                "regression" | "reg" => self.inner.set_task_type(TaskType::Regression),
                _ => return Err(napi::Error::from_reason("task must be 'classification' or 'regression'")),
            }
            Ok(())
        }

        #[napi]
        pub fn set_criterion(&mut self, criterion: String) -> napi::Result<()> {
            match criterion.to_lowercase().as_str() {
                "gini" => self.inner.set_criterion(SplitCriterion::Gini),
                "entropy" => self.inner.set_criterion(SplitCriterion::Entropy),
                "mse" => self.inner.set_criterion(SplitCriterion::MSE),
                "variance" | "var" => self.inner.set_criterion(SplitCriterion::VarianceReduction),
                _ => return Err(napi::Error::from_reason("criterion must be 'gini', 'entropy', 'mse', or 'variance'")),
            }
            Ok(())
        }

        #[napi]
        pub fn print_hyperparameters(&self) {
            self.inner.print_hyperparameters();
        }

        #[napi]
        pub fn load_csv(&mut self, filename: String) -> bool {
            self.inner.load_csv(&filename)
        }

        #[napi]
        pub fn train(&mut self) {
            self.inner.train();
        }

        #[napi]
        pub fn predict(&self, sample: Vec<f64>) -> f64 {
            self.inner.predict(&sample)
        }

        #[napi]
        pub fn predict_class(&self, sample: Vec<f64>) -> i32 {
            self.inner.predict_class(&sample)
        }

        #[napi]
        pub fn predict_batch(&self, samples: Vec<f64>, n_samples: u32) -> Vec<f64> {
            self.inner.predict_batch(&samples, n_samples as usize)
        }

        #[napi]
        pub fn predict_batch_gpu(&self, samples: Vec<f64>, n_samples: u32) -> Vec<f64> {
            self.inner.predict_batch_gpu(&samples, n_samples as usize)
        }

        #[napi]
        pub fn inspect_tree(&self, tree_id: u32) -> Vec<i32> {
            let info = self.inner.inspect_tree(tree_id as usize);
            vec![info.num_nodes, info.max_depth, info.num_leaves, info.num_features_used]
        }

        #[napi]
        pub fn print_tree_info(&self, tree_id: u32) {
            self.inner.print_tree_info(tree_id as usize);
        }

        #[napi]
        pub fn print_tree_structure(&self, tree_id: u32) {
            self.inner.print_tree_structure(tree_id as usize);
        }

        #[napi]
        pub fn add_tree(&mut self) {
            self.inner.add_tree();
        }

        #[napi]
        pub fn remove_tree(&mut self, tree_id: u32) {
            self.inner.remove_tree(tree_id as usize);
        }

        #[napi]
        pub fn replace_tree(&mut self, tree_id: u32) {
            self.inner.replace_tree(tree_id as usize);
        }

        #[napi]
        pub fn retrain_tree(&mut self, tree_id: u32) {
            self.inner.retrain_tree(tree_id as usize);
        }

        #[napi]
        pub fn get_num_trees(&self) -> u32 {
            self.inner.get_num_trees() as u32
        }

        #[napi]
        pub fn enable_feature(&mut self, feature_index: u32) {
            self.inner.enable_feature(feature_index as usize);
        }

        #[napi]
        pub fn disable_feature(&mut self, feature_index: u32) {
            self.inner.disable_feature(feature_index as usize);
        }

        #[napi]
        pub fn reset_features(&mut self) {
            self.inner.reset_features();
        }

        #[napi]
        pub fn print_feature_usage(&self) {
            self.inner.print_feature_usage();
        }

        #[napi]
        pub fn print_feature_importances(&self) {
            self.inner.print_feature_importances();
        }

        #[napi]
        pub fn set_aggregation_method(&mut self, method: String) -> napi::Result<()> {
            match method.to_lowercase().as_str() {
                "majority" | "majority-vote" => self.inner.set_aggregation_method(AggregationMethod::MajorityVote),
                "weighted" | "weighted-vote" => self.inner.set_aggregation_method(AggregationMethod::WeightedVote),
                "mean" => self.inner.set_aggregation_method(AggregationMethod::Mean),
                "weighted-mean" => self.inner.set_aggregation_method(AggregationMethod::WeightedMean),
                _ => return Err(napi::Error::from_reason("method must be 'majority', 'weighted', 'mean', or 'weighted-mean'")),
            }
            Ok(())
        }

        #[napi]
        pub fn set_tree_weight(&mut self, tree_id: u32, weight: f64) {
            self.inner.set_tree_weight(tree_id as usize, weight);
        }

        #[napi]
        pub fn get_tree_weight(&self, tree_id: u32) -> f64 {
            self.inner.get_tree_weight(tree_id as usize)
        }

        #[napi]
        pub fn reset_tree_weights(&mut self) {
            self.inner.reset_tree_weights();
        }

        #[napi]
        pub fn track_sample(&self, sample_index: u32) -> Vec<i32> {
            let info = self.inner.track_sample(sample_index as usize);
            vec![info.num_trees_influenced, info.num_oob_trees]
        }

        #[napi]
        pub fn print_sample_tracking(&self, sample_index: u32) {
            self.inner.print_sample_tracking(sample_index as usize);
        }

        #[napi]
        pub fn print_oob_summary(&self) {
            self.inner.print_oob_summary();
        }

        #[napi]
        pub fn get_global_oob_error(&self) -> f64 {
            self.inner.get_global_oob_error()
        }

        #[napi]
        pub fn highlight_misclassified(&self, predictions: Vec<f64>, actual: Vec<f64>) {
            self.inner.highlight_misclassified(&predictions, &actual);
        }

        #[napi]
        pub fn save_model(&self, filename: String) -> bool {
            self.inner.save_model(&filename)
        }

        #[napi]
        pub fn load_model(&mut self, filename: String) -> bool {
            self.inner.load_model(&filename)
        }

        #[napi]
        pub fn print_forest_info(&self) {
            self.inner.print_forest_info();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// C FFI Bindings
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "cbindings")]
mod cbindings {
    use super::*;
    use std::ffi::CStr;
    use std::os::raw::c_char;

    unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> &'a str {
        unsafe { CStr::from_ptr(ptr).to_str().unwrap_or("") }
    }

    // ── TRandomForest ──

    #[no_mangle]
    pub extern "C" fn rf_create() -> *mut TRandomForest {
        Box::into_raw(Box::new(TRandomForest::new()))
    }

    #[no_mangle]
    pub extern "C" fn rf_create_with_seed(seed: u64) -> *mut TRandomForest {
        Box::into_raw(Box::new(TRandomForest::new_with_seed(seed)))
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_destroy(ptr: *mut TRandomForest) {
        if !ptr.is_null() {
            drop(unsafe { Box::from_raw(ptr) });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_num_trees(ptr: *mut TRandomForest, n: u32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_num_trees(n as usize);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_max_depth(ptr: *mut TRandomForest, d: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_max_depth(d);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_min_samples_leaf(ptr: *mut TRandomForest, m: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_min_samples_leaf(m);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_min_samples_split(ptr: *mut TRandomForest, m: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_min_samples_split(m);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_max_features(ptr: *mut TRandomForest, m: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_max_features(m);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_task_type(ptr: *mut TRandomForest, task_type: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_task_type(if task_type == 1 { TaskType::Regression } else { TaskType::Classification });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_criterion(ptr: *mut TRandomForest, criterion: i32) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_criterion(match criterion {
                1 => SplitCriterion::Entropy,
                2 => SplitCriterion::MSE,
                3 => SplitCriterion::VarianceReduction,
                _ => SplitCriterion::Gini,
            });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_random_seed(ptr: *mut TRandomForest, seed: u64) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_random_seed(seed);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_set_backend(ptr: *mut TRandomForest, backend: *const c_char) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.set_backend_kind(BackendKind::from_str(unsafe { cstr_to_str(backend) }));
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_load_data(
        ptr: *mut TRandomForest,
        data: *const f64,
        targets: *const f64,
        n_samples: u32,
        n_features: u32,
    ) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            let data_slice = unsafe { std::slice::from_raw_parts(data, (n_samples * n_features) as usize) };
            let target_slice = unsafe { std::slice::from_raw_parts(targets, n_samples as usize) };
            rf.load_data(data_slice, target_slice, n_samples as usize, n_features as usize);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_load_csv(
        ptr: *mut TRandomForest,
        filename: *const c_char,
        target_column: i32,
        has_header: i32,
    ) -> i32 {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.load_csv(unsafe { cstr_to_str(filename) }, target_column, has_header != 0) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_fit(ptr: *mut TRandomForest) {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.fit();
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_predict(ptr: *const TRandomForest, sample: *const f64, n_features: u32) -> f64 {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(sample, n_features as usize) };
            rf.predict(slice)
        } else {
            0.0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_predict_class(ptr: *const TRandomForest, sample: *const f64, n_features: u32) -> i32 {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(sample, n_features as usize) };
            rf.predict_class(slice)
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_predict_batch(
        ptr: *const TRandomForest,
        samples: *const f64,
        n_samples: u32,
        out: *mut f64,
    ) {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(samples, (n_samples as usize) * (rf.num_features as usize)) };
            let preds = rf.predict_batch(slice, n_samples as usize);
            unsafe { std::ptr::copy_nonoverlapping(preds.as_ptr(), out, preds.len()) };
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_predict_batch_gpu(
        ptr: *const TRandomForest,
        samples: *const f64,
        n_samples: u32,
        out: *mut f64,
    ) {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(samples, (n_samples as usize) * (rf.num_features as usize)) };
            let preds = rf.predict_batch_gpu(slice, n_samples as usize);
            unsafe { std::ptr::copy_nonoverlapping(preds.as_ptr(), out, preds.len()) };
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_calculate_oob_error(ptr: *const TRandomForest) -> f64 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.calculate_oob_error() } else { 0.0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_get_feature_importance(ptr: *const TRandomForest, feature_index: u32) -> f64 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.get_feature_importance(feature_index as usize) } else { 0.0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_print_feature_importances(ptr: *const TRandomForest) {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.print_feature_importances(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_print_forest_info(ptr: *const TRandomForest) {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.print_forest_info(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_save_model(ptr: *const TRandomForest, filename: *const c_char) -> i32 {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            rf.save_model(unsafe { cstr_to_str(filename) }) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_load_model(ptr: *mut TRandomForest, filename: *const c_char) -> i32 {
        if let Some(rf) = unsafe { ptr.as_mut() } {
            rf.load_model(unsafe { cstr_to_str(filename) }) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_predict_csv(
        ptr: *const TRandomForest,
        input_file: *const c_char,
        output_file: *const c_char,
        has_header: i32,
    ) -> i32 {
        if let Some(rf) = unsafe { ptr.as_ref() } {
            rf.predict_csv(
                unsafe { cstr_to_str(input_file) },
                unsafe { cstr_to_str(output_file) },
                has_header != 0,
            ) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_add_new_tree(ptr: *mut TRandomForest) {
        if let Some(rf) = unsafe { ptr.as_mut() } { rf.add_new_tree(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_remove_tree_at(ptr: *mut TRandomForest, tree_id: u32) {
        if let Some(rf) = unsafe { ptr.as_mut() } { rf.remove_tree_at(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_retrain_tree_at(ptr: *mut TRandomForest, tree_id: u32) {
        if let Some(rf) = unsafe { ptr.as_mut() } { rf.retrain_tree_at(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_get_num_trees(ptr: *const TRandomForest) -> u32 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.get_num_trees() as u32 } else { 0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_get_num_features(ptr: *const TRandomForest) -> i32 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.get_num_features() } else { 0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_get_num_samples(ptr: *const TRandomForest) -> u32 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.get_num_samples() as u32 } else { 0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_get_max_depth(ptr: *const TRandomForest) -> i32 {
        if let Some(rf) = unsafe { ptr.as_ref() } { rf.get_max_depth_val() } else { 0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_accuracy(predictions: *const f64, actual: *const f64, n: u32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::accuracy(p, a)
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_precision(predictions: *const f64, actual: *const f64, n: u32, positive_class: i32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::precision(p, a, positive_class)
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_recall(predictions: *const f64, actual: *const f64, n: u32, positive_class: i32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::recall(p, a, positive_class)
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_f1_score(predictions: *const f64, actual: *const f64, n: u32, positive_class: i32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::f1_score(p, a, positive_class)
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_mean_squared_error(predictions: *const f64, actual: *const f64, n: u32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::mean_squared_error(p, a)
    }

    #[no_mangle]
    pub unsafe extern "C" fn rf_r_squared(predictions: *const f64, actual: *const f64, n: u32) -> f64 {
        let p = unsafe { std::slice::from_raw_parts(predictions, n as usize) };
        let a = unsafe { std::slice::from_raw_parts(actual, n as usize) };
        TRandomForest::r_squared(p, a)
    }

    // ── TRandomForestFacade ──

    #[no_mangle]
    pub extern "C" fn rff_create() -> *mut TRandomForestFacade {
        Box::into_raw(Box::new(TRandomForestFacade::new()))
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_destroy(ptr: *mut TRandomForestFacade) {
        if !ptr.is_null() {
            drop(unsafe { Box::from_raw(ptr) });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_init_forest(ptr: *mut TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.init_forest(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_backend(ptr: *mut TRandomForestFacade, backend: *const c_char) {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.set_backend(BackendKind::from_str(unsafe { cstr_to_str(backend) }));
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_hyperparameter(ptr: *mut TRandomForestFacade, name: *const c_char, value: i32) {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.set_hyperparameter(unsafe { cstr_to_str(name) }, value);
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_task_type(ptr: *mut TRandomForestFacade, task_type: i32) {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.set_task_type(if task_type == 1 { TaskType::Regression } else { TaskType::Classification });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_criterion(ptr: *mut TRandomForestFacade, criterion: i32) {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.set_criterion(match criterion {
                1 => SplitCriterion::Entropy,
                2 => SplitCriterion::MSE,
                3 => SplitCriterion::VarianceReduction,
                _ => SplitCriterion::Gini,
            });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_hyperparameters(ptr: *const TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_hyperparameters(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_load_csv(ptr: *mut TRandomForestFacade, filename: *const c_char) -> i32 {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.load_csv(unsafe { cstr_to_str(filename) }) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_train(ptr: *mut TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.train(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_predict(ptr: *const TRandomForestFacade, sample: *const f64, n_features: u32) -> f64 {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(sample, n_features as usize) };
            f.predict(slice)
        } else {
            0.0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_predict_class(ptr: *const TRandomForestFacade, sample: *const f64, n_features: u32) -> i32 {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(sample, n_features as usize) };
            f.predict_class(slice)
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_predict_batch(
        ptr: *const TRandomForestFacade,
        samples: *const f64,
        n_samples: u32,
        n_features: u32,
        out: *mut f64,
    ) {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(samples, (n_samples * n_features) as usize) };
            let preds = f.predict_batch(slice, n_samples as usize);
            unsafe { std::ptr::copy_nonoverlapping(preds.as_ptr(), out, preds.len()) };
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_predict_batch_gpu(
        ptr: *const TRandomForestFacade,
        samples: *const f64,
        n_samples: u32,
        n_features: u32,
        out: *mut f64,
    ) {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let slice = unsafe { std::slice::from_raw_parts(samples, (n_samples * n_features) as usize) };
            let preds = f.predict_batch_gpu(slice, n_samples as usize);
            unsafe { std::ptr::copy_nonoverlapping(preds.as_ptr(), out, preds.len()) };
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_inspect_tree(
        ptr: *const TRandomForestFacade,
        tree_id: u32,
        out_num_nodes: *mut i32,
        out_max_depth: *mut i32,
        out_num_leaves: *mut i32,
        out_num_features_used: *mut i32,
    ) {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let info = f.inspect_tree(tree_id as usize);
            if !out_num_nodes.is_null() { unsafe { *out_num_nodes = info.num_nodes }; }
            if !out_max_depth.is_null() { unsafe { *out_max_depth = info.max_depth }; }
            if !out_num_leaves.is_null() { unsafe { *out_num_leaves = info.num_leaves }; }
            if !out_num_features_used.is_null() { unsafe { *out_num_features_used = info.num_features_used }; }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_tree_info(ptr: *const TRandomForestFacade, tree_id: u32) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_tree_info(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_tree_structure(ptr: *const TRandomForestFacade, tree_id: u32) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_tree_structure(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_add_tree(ptr: *mut TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.add_tree(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_remove_tree(ptr: *mut TRandomForestFacade, tree_id: u32) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.remove_tree(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_replace_tree(ptr: *mut TRandomForestFacade, tree_id: u32) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.replace_tree(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_retrain_tree(ptr: *mut TRandomForestFacade, tree_id: u32) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.retrain_tree(tree_id as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_get_num_trees(ptr: *const TRandomForestFacade) -> u32 {
        if let Some(f) = unsafe { ptr.as_ref() } { f.get_num_trees() as u32 } else { 0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_enable_feature(ptr: *mut TRandomForestFacade, feature_index: u32) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.enable_feature(feature_index as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_disable_feature(ptr: *mut TRandomForestFacade, feature_index: u32) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.disable_feature(feature_index as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_reset_features(ptr: *mut TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.reset_features(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_feature_usage(ptr: *const TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_feature_usage(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_feature_importances(ptr: *const TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_feature_importances(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_aggregation_method(ptr: *mut TRandomForestFacade, method: i32) {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.set_aggregation_method(match method {
                1 => AggregationMethod::WeightedVote,
                2 => AggregationMethod::Mean,
                3 => AggregationMethod::WeightedMean,
                _ => AggregationMethod::MajorityVote,
            });
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_set_tree_weight(ptr: *mut TRandomForestFacade, tree_id: u32, weight: f64) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.set_tree_weight(tree_id as usize, weight); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_get_tree_weight(ptr: *const TRandomForestFacade, tree_id: u32) -> f64 {
        if let Some(f) = unsafe { ptr.as_ref() } { f.get_tree_weight(tree_id as usize) } else { 1.0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_reset_tree_weights(ptr: *mut TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_mut() } { f.reset_tree_weights(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_track_sample(
        ptr: *const TRandomForestFacade,
        sample_index: u32,
        out_trees_influenced: *mut i32,
        out_oob_trees: *mut i32,
    ) {
        if let Some(f) = unsafe { ptr.as_ref() } {
            let info = f.track_sample(sample_index as usize);
            if !out_trees_influenced.is_null() { unsafe { *out_trees_influenced = info.num_trees_influenced }; }
            if !out_oob_trees.is_null() { unsafe { *out_oob_trees = info.num_oob_trees }; }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_sample_tracking(ptr: *const TRandomForestFacade, sample_index: u32) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_sample_tracking(sample_index as usize); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_oob_summary(ptr: *const TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_oob_summary(); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_get_global_oob_error(ptr: *const TRandomForestFacade) -> f64 {
        if let Some(f) = unsafe { ptr.as_ref() } { f.get_global_oob_error() } else { 0.0 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_save_model(ptr: *const TRandomForestFacade, filename: *const c_char) -> i32 {
        if let Some(f) = unsafe { ptr.as_ref() } {
            f.save_model(unsafe { cstr_to_str(filename) }) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_load_model(ptr: *mut TRandomForestFacade, filename: *const c_char) -> i32 {
        if let Some(f) = unsafe { ptr.as_mut() } {
            f.load_model(unsafe { cstr_to_str(filename) }) as i32
        } else {
            0
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn rff_print_forest_info(ptr: *const TRandomForestFacade) {
        if let Some(f) = unsafe { ptr.as_ref() } { f.print_forest_info(); }
    }
}

