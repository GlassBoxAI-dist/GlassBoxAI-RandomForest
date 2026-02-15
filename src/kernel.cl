//
// Matthew Abbott 2025
// OpenCL kernels for Facaded Random Forest
// Ported from CUDA kernel.cu
//

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct {
    int is_leaf;
    int feature_index;
    double threshold;
    double prediction;
    int class_label;
    int left_child;
    int right_child;
} FlatTreeNode;

__kernel void predictBatchKernel(
    __global const double* data,
    int numFeatures,
    __global const FlatTreeNode* allTreeNodes,
    __global const int* treeNodeOffsets,
    int numTrees,
    int numSamples,
    int taskType,
    __global double* predictions
) {
    int sampleIdx = get_global_id(0);
    if (sampleIdx >= numSamples) return;

    __global const double* sample = &data[sampleIdx * numFeatures];

    if (taskType == 1) {
        double sum = 0.0;
        for (int t = 0; t < numTrees; t++) {
            __global const FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (tree[nodeIdx].is_leaf == 0) {
                if (sample[tree[nodeIdx].feature_index] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].left_child;
                else
                    nodeIdx = tree[nodeIdx].right_child;
            }
            sum += tree[nodeIdx].prediction;
        }
        predictions[sampleIdx] = sum / numTrees;
    } else {
        int votes[100];
        for (int i = 0; i < 100; i++) votes[i] = 0;

        for (int t = 0; t < numTrees; t++) {
            __global const FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (tree[nodeIdx].is_leaf == 0) {
                if (sample[tree[nodeIdx].feature_index] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].left_child;
                else
                    nodeIdx = tree[nodeIdx].right_child;
            }
            int classLabel = tree[nodeIdx].class_label;
            if (classLabel >= 0 && classLabel < 100)
                votes[classLabel]++;
        }

        int maxVotes = 0;
        int maxClass = 0;
        for (int i = 0; i < 100; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                maxClass = i;
            }
        }
        predictions[sampleIdx] = maxClass;
    }
}

__kernel void predictBatchWeightedKernel(
    __global const double* data,
    int numFeatures,
    __global const FlatTreeNode* allTreeNodes,
    __global const int* treeNodeOffsets,
    __global const double* treeWeights,
    int numTrees,
    int numSamples,
    int taskType,
    __global double* predictions
) {
    int sampleIdx = get_global_id(0);
    if (sampleIdx >= numSamples) return;

    __global const double* sample = &data[sampleIdx * numFeatures];

    if (taskType == 1) {
        double sum = 0.0;
        double totalWeight = 0.0;
        for (int t = 0; t < numTrees; t++) {
            __global const FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (tree[nodeIdx].is_leaf == 0) {
                if (sample[tree[nodeIdx].feature_index] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].left_child;
                else
                    nodeIdx = tree[nodeIdx].right_child;
            }
            sum += tree[nodeIdx].prediction * treeWeights[t];
            totalWeight += treeWeights[t];
        }
        predictions[sampleIdx] = (totalWeight > 0) ? sum / totalWeight : 0.0;
    } else {
        double votes[100];
        for (int i = 0; i < 100; i++) votes[i] = 0.0;

        for (int t = 0; t < numTrees; t++) {
            __global const FlatTreeNode* tree = &allTreeNodes[treeNodeOffsets[t]];
            int nodeIdx = 0;
            while (tree[nodeIdx].is_leaf == 0) {
                if (sample[tree[nodeIdx].feature_index] <= tree[nodeIdx].threshold)
                    nodeIdx = tree[nodeIdx].left_child;
                else
                    nodeIdx = tree[nodeIdx].right_child;
            }
            int classLabel = tree[nodeIdx].class_label;
            if (classLabel >= 0 && classLabel < 100)
                votes[classLabel] += treeWeights[t];
        }

        double maxVotes = 0.0;
        int maxClass = 0;
        for (int i = 0; i < 100; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                maxClass = i;
            }
        }
        predictions[sampleIdx] = maxClass;
    }
}
