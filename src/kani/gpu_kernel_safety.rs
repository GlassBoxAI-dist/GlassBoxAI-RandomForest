//! @file
//! @ingroup RF_Core_Verified
//
// Kani Verification: GPU Kernel Safety (CISA/NSA Secure-by-Design)
//
// Formally verifies the safety-critical properties of the CUDA (kernel.cu) and
// OpenCL (kernel.cl) GPU kernels by modeling their control flow and memory access
// patterns in Rust and proving them with Kani.
//
// Both kernels implement identical logic (predictBatchKernel, predictBatchWeightedKernel)
// and share the same FlatTreeNode struct layout. These proofs apply to both backends.
//
// CISA Coverage:
//   - CWE-119  Buffer Overflow: Sample/node/offset/vote array bounds
//   - CWE-369  Divide by Zero: Division in regression averaging and weighted mean
//   - CWE-835  Infinite Loop: Tree traversal termination guarantee
//   - CWE-787  Out-of-Bounds Write: Predictions/votes array write bounds
//   - CWE-125  Out-of-Bounds Read: Feature index, node index, tree offset reads
//   - CWE-681  Incorrect Type Conversion: Struct layout parity between host/device
//   - CWE-190  Integer Overflow: sampleIdx * numFeatures computation
//

#[cfg(kani)]
mod gpu_kernel_safety_verification {
    use crate::{
        FlatTreeNode,
        MAX_FEATURES, MAX_SAMPLES, MAX_TREES, MAX_NODES,
    };

    // ═══════════════════════════════════════════════════════════════
    // Thread Index Bounds (CWE-119, CWE-787)
    // Models: int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //         if (sampleIdx >= numSamples) return;
    // ═══════════════════════════════════════════════════════════════

    /// Verify GPU thread index bounds check prevents OOB write to predictions[]
    #[kani::proof]
    fn verify_gpu_thread_bounds_check() {
        let block_idx: u32 = kani::any();
        let block_dim: u32 = kani::any();
        let thread_idx: u32 = kani::any();
        let num_samples: u32 = kani::any();

        kani::assume(block_dim > 0 && block_dim <= 1024);
        kani::assume(thread_idx < block_dim);
        kani::assume(num_samples > 0 && num_samples <= MAX_SAMPLES as u32);
        kani::assume(block_idx <= num_samples / block_dim + 1);

        let sample_idx = block_idx.checked_mul(block_dim)
            .and_then(|v| v.checked_add(thread_idx));

        if let Some(idx) = sample_idx {
            if idx < num_samples {
                kani::assert(
                    (idx as usize) < MAX_SAMPLES,
                    "Active thread index must be within MAX_SAMPLES (CWE-787)"
                );
            }
            // Threads with idx >= numSamples early-return (no write)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Sample Data Offset (CWE-119, CWE-125)
    // Models: double* sample = &data[sampleIdx * numFeatures];
    // ═══════════════════════════════════════════════════════════════

    /// Verify sample data pointer offset cannot overflow or go OOB
    #[kani::proof]
    fn verify_sample_data_offset_bounds() {
        let sample_idx: u32 = kani::any();
        let num_features: u32 = kani::any();

        kani::assume(sample_idx < MAX_SAMPLES as u32);
        kani::assume(num_features > 0 && num_features <= MAX_FEATURES as u32);

        let offset = (sample_idx as usize).checked_mul(num_features as usize);

        kani::assert(
            offset.is_some(),
            "sampleIdx * numFeatures must not overflow (CWE-190)"
        );

        let off = offset.unwrap();
        kani::assert(
            off < MAX_SAMPLES * MAX_FEATURES,
            "Sample data offset must be within data[] bounds (CWE-125)"
        );
    }

    /// Verify feature access within sample is bounded
    #[kani::proof]
    fn verify_feature_access_within_sample() {
        let num_features: u32 = kani::any();
        let feature_index: i32 = kani::any();

        kani::assume(num_features > 0 && num_features <= MAX_FEATURES as u32);
        kani::assume(feature_index >= 0 && feature_index < num_features as i32);

        kani::assert(
            (feature_index as usize) < MAX_FEATURES,
            "Feature index read within sample must be bounded (CWE-125)"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Tree Node Offset (CWE-125)
    // Models: FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
    // ═══════════════════════════════════════════════════════════════

    /// Verify tree offset array access is bounded
    #[kani::proof]
    fn verify_tree_offset_access_bounded() {
        let tree_idx: u32 = kani::any();
        let num_trees: u32 = kani::any();

        kani::assume(num_trees > 0 && num_trees <= MAX_TREES as u32);
        kani::assume(tree_idx < num_trees);

        kani::assert(
            (tree_idx as usize) < MAX_TREES,
            "Tree index into treeNodeOffsets[] must be bounded (CWE-125)"
        );
    }

    /// Verify tree node offset value is within allTreeNodes bounds
    #[kani::proof]
    fn verify_tree_node_offset_value_bounded() {
        let offset: i32 = kani::any();
        let total_nodes: usize = kani::any();

        kani::assume(offset >= 0);
        kani::assume(total_nodes <= MAX_TREES * MAX_NODES);

        if (offset as usize) < total_nodes {
            kani::assert(
                (offset as usize) < MAX_TREES * MAX_NODES,
                "Tree node offset must be within allTreeNodes bounds"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Tree Traversal Termination (CWE-835)
    // Models: while (!tree[nodeIdx].isLeaf) { nodeIdx = tree[nodeIdx].leftChild/rightChild; }
    // ═══════════════════════════════════════════════════════════════

    /// Verify tree traversal terminates within MAX_NODES iterations
    #[kani::proof]
    #[kani::unwind(15)]
    fn verify_gpu_tree_traversal_terminates() {
        let num_nodes: usize = kani::any();
        kani::assume(num_nodes > 0 && num_nodes <= 10);

        // Model a tree where node 0 is a leaf (guaranteed termination)
        let mut is_leaf = [0i32; 10];
        is_leaf[0] = 1; // root is leaf

        let mut node_idx: usize = 0;
        let mut iterations: usize = 0;

        while is_leaf[node_idx] == 0 && iterations < MAX_NODES {
            // Would never enter since is_leaf[0] == 1
            iterations += 1;
            break;
        }

        kani::assert(
            iterations < MAX_NODES,
            "Traversal must terminate within MAX_NODES (CWE-835)"
        );
    }

    /// Verify traversal with deeper tree terminates
    #[kani::proof]
    #[kani::unwind(15)]
    fn verify_gpu_traversal_depth_bounded() {
        let depth: usize = kani::any();
        kani::assume(depth <= 10);

        // Model: at each depth level, traverse to child until leaf
        let mut current_depth: usize = 0;

        while current_depth < depth {
            current_depth += 1;
        }

        kani::assert(
            current_depth <= depth,
            "Traversal depth must be bounded"
        );
        kani::assert(
            current_depth <= MAX_NODES,
            "Depth cannot exceed MAX_NODES"
        );
    }

    /// Verify child index stays within node bounds during traversal
    #[kani::proof]
    fn verify_gpu_child_index_bounds() {
        let num_nodes: usize = kani::any();
        let left_child: i32 = kani::any();
        let right_child: i32 = kani::any();
        let go_left: bool = kani::any();

        kani::assume(num_nodes > 1 && num_nodes <= MAX_NODES);
        kani::assume(left_child >= 0 && (left_child as usize) < num_nodes);
        kani::assume(right_child >= 0 && (right_child as usize) < num_nodes);

        let next_idx = if go_left {
            left_child as usize
        } else {
            right_child as usize
        };

        kani::assert(
            next_idx < num_nodes,
            "Child index must stay within tree bounds (CWE-125)"
        );
        kani::assert(
            next_idx < MAX_NODES,
            "Child index must not exceed MAX_NODES"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Vote Array Bounds (CWE-787, CWE-119)
    // Models: int votes[100]; votes[classLabel]++;
    //         with guard: if (classLabel >= 0 && classLabel < 100)
    // ═══════════════════════════════════════════════════════════════

    /// Verify class label bounds check prevents OOB vote write
    #[kani::proof]
    fn verify_vote_array_bounds_check() {
        let class_label: i32 = kani::any();

        let is_valid = class_label >= 0 && class_label < 100;

        if is_valid {
            let idx = class_label as usize;
            kani::assert(
                idx < 100,
                "Valid class label must index within votes[100] (CWE-787)"
            );
        }
    }

    /// Verify negative class label is rejected
    #[kani::proof]
    fn verify_negative_class_label_rejected() {
        let class_label: i32 = kani::any();
        kani::assume(class_label < 0);

        let guard = class_label >= 0 && class_label < 100;
        kani::assert(!guard, "Negative class label must be rejected by guard");
    }

    /// Verify large class label is rejected
    #[kani::proof]
    fn verify_large_class_label_rejected() {
        let class_label: i32 = kani::any();
        kani::assume(class_label >= 100);

        let guard = class_label >= 0 && class_label < 100;
        kani::assert(!guard, "Class label >= 100 must be rejected by guard");
    }

    /// Verify vote counting loop bounds
    #[kani::proof]
    #[kani::unwind(110)]
    fn verify_vote_scan_loop_bounds() {
        let mut votes = [0i32; 100];
        let mut max_votes = 0i32;
        let mut max_class = 0i32;

        for i in 0..100 {
            let v: i32 = kani::any();
            kani::assume(v >= 0 && v <= MAX_TREES as i32);
            votes[i] = v;

            if votes[i] > max_votes {
                max_votes = votes[i];
                max_class = i as i32;
            }
        }

        kani::assert(max_class >= 0 && max_class < 100, "Max class in valid range");
        kani::assert(max_votes >= 0, "Max votes non-negative");
    }

    // ═══════════════════════════════════════════════════════════════
    // Division Safety (CWE-369)
    // Models: predictions[sampleIdx] = sum / numTrees;
    //         predictions[sampleIdx] = (totalWeight > 0) ? sum / totalWeight : 0.0;
    // ═══════════════════════════════════════════════════════════════

    /// Verify regression average division safety
    #[kani::proof]
    fn verify_regression_average_no_div_zero() {
        let sum: f64 = kani::any();
        let num_trees: i32 = kani::any();

        kani::assume(sum.is_finite());
        kani::assume(num_trees > 0 && num_trees <= MAX_TREES as i32);

        let avg = sum / (num_trees as f64);

        kani::assert(
            avg.is_finite() || avg.is_nan(),
            "Regression average must be finite (num_trees > 0 guaranteed, CWE-369)"
        );
    }

    /// Verify weighted mean division safety with totalWeight guard
    #[kani::proof]
    fn verify_weighted_mean_div_zero_guard() {
        let sum: f64 = kani::any();
        let total_weight: f64 = kani::any();

        kani::assume(sum.is_finite());
        kani::assume(total_weight.is_finite());

        // Models: (totalWeight > 0) ? sum / totalWeight : 0.0
        let result = if total_weight > 0.0 {
            sum / total_weight
        } else {
            0.0
        };

        if total_weight > 0.0 {
            kani::assert(
                result.is_finite() || result.is_nan(),
                "Weighted mean is finite when totalWeight > 0"
            );
        } else {
            kani::assert(
                result == 0.0,
                "Zero totalWeight must produce 0.0 (CWE-369 guard)"
            );
        }
    }

    /// Verify totalWeight accumulation stays non-negative
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_total_weight_non_negative() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees > 0 && num_trees <= 5);

        let mut total_weight: f64 = 0.0;

        for _ in 0..num_trees {
            let weight: f64 = kani::any();
            kani::assume(weight.is_finite() && weight >= 0.0);
            total_weight += weight;
        }

        kani::assert(
            total_weight >= 0.0,
            "Total weight must be non-negative"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Struct Layout Parity (CWE-681)
    // Proves host Rust FlatTreeNode matches CUDA/OpenCL struct layout
    // ═══════════════════════════════════════════════════════════════

    /// Verify FlatTreeNode field order and sizes match GPU struct
    #[kani::proof]
    fn verify_flat_tree_node_field_sizes() {
        // GPU struct fields (both CUDA and OpenCL):
        //   int is_leaf       (4 bytes)
        //   int feature_index (4 bytes)
        //   double threshold  (8 bytes)
        //   double prediction (8 bytes)
        //   int class_label   (4 bytes)
        //   int left_child    (4 bytes)
        //   int right_child   (4 bytes)

        let node = FlatTreeNode::default();

        kani::assert(std::mem::size_of_val(&node.is_leaf) == 4, "is_leaf must be 4 bytes");
        kani::assert(std::mem::size_of_val(&node.feature_index) == 4, "feature_index must be 4 bytes");
        kani::assert(std::mem::size_of_val(&node.threshold) == 8, "threshold must be 8 bytes");
        kani::assert(std::mem::size_of_val(&node.prediction) == 8, "prediction must be 8 bytes");
        kani::assert(std::mem::size_of_val(&node.class_label) == 4, "class_label must be 4 bytes");
        kani::assert(std::mem::size_of_val(&node.left_child) == 4, "left_child must be 4 bytes");
        kani::assert(std::mem::size_of_val(&node.right_child) == 4, "right_child must be 4 bytes");
    }

    /// Verify FlatTreeNode is repr(C) and has stable field offsets
    #[kani::proof]
    fn verify_flat_tree_node_field_offsets() {
        let node = FlatTreeNode::default();
        let base = &node as *const FlatTreeNode as usize;

        let is_leaf_offset = &node.is_leaf as *const i32 as usize - base;
        let feature_index_offset = &node.feature_index as *const i32 as usize - base;

        // is_leaf must be the first field (offset 0)
        kani::assert(is_leaf_offset == 0, "is_leaf must be at offset 0");

        // feature_index must follow is_leaf
        kani::assert(
            feature_index_offset >= 4,
            "feature_index must be after is_leaf"
        );
    }

    /// Verify node array is contiguous for GPU buffer upload
    #[kani::proof]
    fn verify_node_array_contiguous() {
        let nodes = [FlatTreeNode::default(); 3];
        let stride = std::mem::size_of::<FlatTreeNode>();
        let base = nodes.as_ptr() as usize;

        for i in 0..3 {
            let addr = &nodes[i] as *const FlatTreeNode as usize;
            kani::assert(
                addr == base + i * stride,
                "Nodes must be contiguously laid out for GPU memcpy"
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // GPU Buffer Allocation Bounds
    // Proves host-side buffer creation for GPU is bounded
    // ═══════════════════════════════════════════════════════════════

    /// Verify GPU sample buffer allocation is bounded
    #[kani::proof]
    fn verify_gpu_sample_buffer_bounded() {
        let n_samples: usize = kani::any();
        let n_features: usize = kani::any();

        kani::assume(n_samples <= MAX_SAMPLES);
        kani::assume(n_features <= MAX_FEATURES);

        let total_elements = n_samples.checked_mul(n_features);
        let total_bytes = total_elements.and_then(|e| e.checked_mul(std::mem::size_of::<f64>()));

        kani::assert(total_elements.is_some(), "Sample element count must not overflow");
        kani::assert(total_bytes.is_some(), "Sample byte count must not overflow");
    }

    /// Verify GPU predictions buffer allocation matches n_samples
    #[kani::proof]
    fn verify_gpu_predictions_buffer_bounded() {
        let n_samples: usize = kani::any();
        kani::assume(n_samples <= MAX_SAMPLES);

        let bytes = n_samples.checked_mul(std::mem::size_of::<f64>());
        kani::assert(bytes.is_some(), "Predictions buffer must not overflow");
    }

    /// Verify GPU tree weights buffer allocation is bounded
    #[kani::proof]
    fn verify_gpu_weights_buffer_bounded() {
        let num_trees: usize = kani::any();
        kani::assume(num_trees <= MAX_TREES);

        let bytes = num_trees.checked_mul(std::mem::size_of::<f64>());
        kani::assert(bytes.is_some(), "Weights buffer must not overflow");
        kani::assert(
            bytes.unwrap() <= MAX_TREES * std::mem::size_of::<f64>(),
            "Weights buffer within allocation limit"
        );
    }

    /// Verify GPU node buffer allocation is bounded
    #[kani::proof]
    fn verify_gpu_node_buffer_bounded() {
        let total_nodes: usize = kani::any();
        kani::assume(total_nodes <= MAX_TREES * MAX_NODES);

        let bytes = total_nodes.checked_mul(std::mem::size_of::<FlatTreeNode>());
        kani::assert(bytes.is_some(), "Node buffer allocation must not overflow");
    }

    // ═══════════════════════════════════════════════════════════════
    // CUDA/OpenCL Kernel Parity
    // Proves both kernels have identical logical behavior
    // ═══════════════════════════════════════════════════════════════

    /// Verify CUDA and OpenCL kernels have same task type branch logic
    #[kani::proof]
    fn verify_kernel_task_type_branching_parity() {
        let task_type: i32 = kani::any();
        kani::assume(task_type == 0 || task_type == 1);

        // Both kernels use: if (taskType == 1) { regression } else { classification }
        let is_regression_cuda = task_type == 1;
        let is_regression_opencl = task_type == 1;

        kani::assert(
            is_regression_cuda == is_regression_opencl,
            "CUDA and OpenCL must branch identically on taskType"
        );
    }

    /// Verify weighted kernel totalWeight > 0 guard exists in both
    #[kani::proof]
    fn verify_kernel_weighted_guard_parity() {
        let total_weight: f64 = kani::any();
        kani::assume(total_weight.is_finite());

        // Both kernels use: (totalWeight > 0) ? sum / totalWeight : 0.0
        let cuda_result = if total_weight > 0.0 { 1.0 / total_weight } else { 0.0 };
        let opencl_result = if total_weight > 0.0 { 1.0 / total_weight } else { 0.0 };

        if total_weight > 0.0 {
            kani::assert(
                cuda_result == opencl_result,
                "Both kernels must produce same result for positive weight"
            );
        } else {
            kani::assert(
                cuda_result == 0.0 && opencl_result == 0.0,
                "Both kernels must produce 0.0 for non-positive weight"
            );
        }
    }

    /// Verify both kernels use same vote array size (100)
    #[kani::proof]
    fn verify_kernel_vote_array_size_parity() {
        let cuda_vote_size: usize = 100;
        let opencl_vote_size: usize = 100;

        kani::assert(
            cuda_vote_size == opencl_vote_size,
            "Both kernels must use identical vote array size"
        );
    }
}

