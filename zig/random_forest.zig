//
// Matthew Abbott 2025
// Zig wrapper for Facaded Random Forest
//
// Build the Rust library with: cargo build --release --features cbindings
//
// Usage:
//   const rf = @import("random_forest");
//
//   var forest = rf.RandomForest.create();
//   defer forest.destroy();
//   _ = forest.loadCsv("data.csv", -1, true);
//   forest.fit();
//   const pred = forest.predict(&[_]f64{ 5.1, 3.5, 1.4, 0.2 });
//
// Build with:
//   zig build-exe main.zig -lfacaded_random_forest -L ../target/release -lc
//

const std = @import("std");

// ═══════════════════════════════════════════════════════════════════
// C FFI declarations
// ═══════════════════════════════════════════════════════════════════

const c = @cImport({
    @cInclude("facaded_random_forest.h");
});

// ═══════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════

pub const TaskType = enum(i32) {
    classification = 0,
    regression = 1,
};

pub const Criterion = enum(i32) {
    gini = 0,
    entropy = 1,
    mse = 2,
    variance_reduction = 3,
};

pub const AggregationMethod = enum(i32) {
    majority_vote = 0,
    weighted_vote = 1,
    mean = 2,
    weighted_mean = 3,
};

// ═══════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════

pub const TreeInspection = struct {
    num_nodes: i32,
    max_depth: i32,
    num_leaves: i32,
    num_features_used: i32,
};

pub const SampleTracking = struct {
    trees_influenced: i32,
    oob_trees: i32,
};

// ═══════════════════════════════════════════════════════════════════
// RandomForest (wraps TRandomForest)
// ═══════════════════════════════════════════════════════════════════

pub const RandomForest = struct {
    ptr: *c.TRandomForest,

    pub fn create() RandomForest {
        return .{ .ptr = c.rf_create().? };
    }

    pub fn createWithSeed(seed: u64) RandomForest {
        return .{ .ptr = c.rf_create_with_seed(seed).? };
    }

    pub fn destroy(self: *RandomForest) void {
        c.rf_destroy(self.ptr);
    }

    pub fn setNumTrees(self: *RandomForest, n: u32) void {
        c.rf_set_num_trees(self.ptr, n);
    }

    pub fn setMaxDepth(self: *RandomForest, d: i32) void {
        c.rf_set_max_depth(self.ptr, d);
    }

    pub fn setMinSamplesLeaf(self: *RandomForest, m: i32) void {
        c.rf_set_min_samples_leaf(self.ptr, m);
    }

    pub fn setMinSamplesSplit(self: *RandomForest, m: i32) void {
        c.rf_set_min_samples_split(self.ptr, m);
    }

    pub fn setMaxFeatures(self: *RandomForest, m: i32) void {
        c.rf_set_max_features(self.ptr, m);
    }

    pub fn setTaskType(self: *RandomForest, task_type: TaskType) void {
        c.rf_set_task_type(self.ptr, @intFromEnum(task_type));
    }

    pub fn setCriterion(self: *RandomForest, criterion: Criterion) void {
        c.rf_set_criterion(self.ptr, @intFromEnum(criterion));
    }

    pub fn setRandomSeed(self: *RandomForest, seed: u64) void {
        c.rf_set_random_seed(self.ptr, seed);
    }

    pub fn setBackend(self: *RandomForest, backend: [*:0]const u8) void {
        c.rf_set_backend(self.ptr, backend);
    }

    pub fn loadData(self: *RandomForest, data: []const f64, targets: []const f64, n_samples: u32, n_features: u32) void {
        c.rf_load_data(self.ptr, data.ptr, targets.ptr, n_samples, n_features);
    }

    pub fn loadCsv(self: *RandomForest, filename: [*:0]const u8, target_column: i32, has_header: bool) bool {
        return c.rf_load_csv(self.ptr, filename, target_column, if (has_header) @as(i32, 1) else @as(i32, 0)) != 0;
    }

    pub fn fit(self: *RandomForest) void {
        c.rf_fit(self.ptr);
    }

    pub fn predict(self: *const RandomForest, sample: []const f64) f64 {
        return c.rf_predict(self.ptr, sample.ptr, @intCast(sample.len));
    }

    pub fn predictClass(self: *const RandomForest, sample: []const f64) i32 {
        return c.rf_predict_class(self.ptr, sample.ptr, @intCast(sample.len));
    }

    pub fn predictBatch(self: *const RandomForest, samples: []const f64, n_samples: u32, output: []f64) void {
        c.rf_predict_batch(self.ptr, samples.ptr, n_samples, output.ptr);
    }

    pub fn predictBatchGpu(self: *const RandomForest, samples: []const f64, n_samples: u32, output: []f64) void {
        c.rf_predict_batch_gpu(self.ptr, samples.ptr, n_samples, output.ptr);
    }

    pub fn calculateOobError(self: *const RandomForest) f64 {
        return c.rf_calculate_oob_error(self.ptr);
    }

    pub fn getFeatureImportance(self: *const RandomForest, feature_index: u32) f64 {
        return c.rf_get_feature_importance(self.ptr, feature_index);
    }

    pub fn printFeatureImportances(self: *const RandomForest) void {
        c.rf_print_feature_importances(self.ptr);
    }

    pub fn printForestInfo(self: *const RandomForest) void {
        c.rf_print_forest_info(self.ptr);
    }

    pub fn saveModel(self: *const RandomForest, filename: [*:0]const u8) bool {
        return c.rf_save_model(self.ptr, filename) != 0;
    }

    pub fn loadModel(self: *RandomForest, filename: [*:0]const u8) bool {
        return c.rf_load_model(self.ptr, filename) != 0;
    }

    pub fn predictCsv(self: *const RandomForest, input_file: [*:0]const u8, output_file: [*:0]const u8, has_header: bool) bool {
        return c.rf_predict_csv(self.ptr, input_file, output_file, if (has_header) @as(i32, 1) else @as(i32, 0)) != 0;
    }

    pub fn addNewTree(self: *RandomForest) void {
        c.rf_add_new_tree(self.ptr);
    }

    pub fn removeTreeAt(self: *RandomForest, tree_id: u32) void {
        c.rf_remove_tree_at(self.ptr, tree_id);
    }

    pub fn retrainTreeAt(self: *RandomForest, tree_id: u32) void {
        c.rf_retrain_tree_at(self.ptr, tree_id);
    }

    pub fn getNumTrees(self: *const RandomForest) u32 {
        return c.rf_get_num_trees(self.ptr);
    }

    pub fn getNumFeatures(self: *const RandomForest) i32 {
        return c.rf_get_num_features(self.ptr);
    }

    pub fn getNumSamples(self: *const RandomForest) u32 {
        return c.rf_get_num_samples(self.ptr);
    }

    pub fn getMaxDepth(self: *const RandomForest) i32 {
        return c.rf_get_max_depth(self.ptr);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Static metrics (no forest instance needed)
// ═══════════════════════════════════════════════════════════════════

pub fn accuracy(predictions: []const f64, actual: []const f64) f64 {
    return c.rf_accuracy(predictions.ptr, actual.ptr, @intCast(predictions.len));
}

pub fn precision(predictions: []const f64, actual: []const f64, positive_class: i32) f64 {
    return c.rf_precision(predictions.ptr, actual.ptr, @intCast(predictions.len), positive_class);
}

pub fn recall(predictions: []const f64, actual: []const f64, positive_class: i32) f64 {
    return c.rf_recall(predictions.ptr, actual.ptr, @intCast(predictions.len), positive_class);
}

pub fn f1Score(predictions: []const f64, actual: []const f64, positive_class: i32) f64 {
    return c.rf_f1_score(predictions.ptr, actual.ptr, @intCast(predictions.len), positive_class);
}

pub fn meanSquaredError(predictions: []const f64, actual: []const f64) f64 {
    return c.rf_mean_squared_error(predictions.ptr, actual.ptr, @intCast(predictions.len));
}

pub fn rSquared(predictions: []const f64, actual: []const f64) f64 {
    return c.rf_r_squared(predictions.ptr, actual.ptr, @intCast(predictions.len));
}

// ═══════════════════════════════════════════════════════════════════
// RandomForestFacade (wraps TRandomForestFacade)
// ═══════════════════════════════════════════════════════════════════

pub const RandomForestFacade = struct {
    ptr: *c.TRandomForestFacade,

    pub fn create() RandomForestFacade {
        return .{ .ptr = c.rff_create().? };
    }

    pub fn destroy(self: *RandomForestFacade) void {
        c.rff_destroy(self.ptr);
    }

    pub fn initForest(self: *RandomForestFacade) void {
        c.rff_init_forest(self.ptr);
    }

    pub fn setBackend(self: *RandomForestFacade, backend: [*:0]const u8) void {
        c.rff_set_backend(self.ptr, backend);
    }

    pub fn setHyperparameter(self: *RandomForestFacade, name: [*:0]const u8, value: i32) void {
        c.rff_set_hyperparameter(self.ptr, name, value);
    }

    pub fn setTaskType(self: *RandomForestFacade, task_type: TaskType) void {
        c.rff_set_task_type(self.ptr, @intFromEnum(task_type));
    }

    pub fn setCriterion(self: *RandomForestFacade, criterion: Criterion) void {
        c.rff_set_criterion(self.ptr, @intFromEnum(criterion));
    }

    pub fn printHyperparameters(self: *const RandomForestFacade) void {
        c.rff_print_hyperparameters(self.ptr);
    }

    pub fn loadCsv(self: *RandomForestFacade, filename: [*:0]const u8) bool {
        return c.rff_load_csv(self.ptr, filename) != 0;
    }

    pub fn train(self: *RandomForestFacade) void {
        c.rff_train(self.ptr);
    }

    pub fn predict(self: *const RandomForestFacade, sample: []const f64) f64 {
        return c.rff_predict(self.ptr, sample.ptr, @intCast(sample.len));
    }

    pub fn predictClass(self: *const RandomForestFacade, sample: []const f64) i32 {
        return c.rff_predict_class(self.ptr, sample.ptr, @intCast(sample.len));
    }

    pub fn predictBatch(self: *const RandomForestFacade, samples: []const f64, n_samples: u32, n_features: u32, output: []f64) void {
        c.rff_predict_batch(self.ptr, samples.ptr, n_samples, n_features, output.ptr);
    }

    pub fn predictBatchGpu(self: *const RandomForestFacade, samples: []const f64, n_samples: u32, n_features: u32, output: []f64) void {
        c.rff_predict_batch_gpu(self.ptr, samples.ptr, n_samples, n_features, output.ptr);
    }

    pub fn inspectTree(self: *const RandomForestFacade, tree_id: u32) TreeInspection {
        var info: TreeInspection = undefined;
        c.rff_inspect_tree(self.ptr, tree_id, &info.num_nodes, &info.max_depth, &info.num_leaves, &info.num_features_used);
        return info;
    }

    pub fn printTreeInfo(self: *const RandomForestFacade, tree_id: u32) void {
        c.rff_print_tree_info(self.ptr, tree_id);
    }

    pub fn printTreeStructure(self: *const RandomForestFacade, tree_id: u32) void {
        c.rff_print_tree_structure(self.ptr, tree_id);
    }

    pub fn addTree(self: *RandomForestFacade) void {
        c.rff_add_tree(self.ptr);
    }

    pub fn removeTree(self: *RandomForestFacade, tree_id: u32) void {
        c.rff_remove_tree(self.ptr, tree_id);
    }

    pub fn replaceTree(self: *RandomForestFacade, tree_id: u32) void {
        c.rff_replace_tree(self.ptr, tree_id);
    }

    pub fn retrainTree(self: *RandomForestFacade, tree_id: u32) void {
        c.rff_retrain_tree(self.ptr, tree_id);
    }

    pub fn getNumTrees(self: *const RandomForestFacade) u32 {
        return c.rff_get_num_trees(self.ptr);
    }

    pub fn enableFeature(self: *RandomForestFacade, feature_index: u32) void {
        c.rff_enable_feature(self.ptr, feature_index);
    }

    pub fn disableFeature(self: *RandomForestFacade, feature_index: u32) void {
        c.rff_disable_feature(self.ptr, feature_index);
    }

    pub fn resetFeatures(self: *RandomForestFacade) void {
        c.rff_reset_features(self.ptr);
    }

    pub fn printFeatureUsage(self: *const RandomForestFacade) void {
        c.rff_print_feature_usage(self.ptr);
    }

    pub fn printFeatureImportances(self: *const RandomForestFacade) void {
        c.rff_print_feature_importances(self.ptr);
    }

    pub fn setAggregationMethod(self: *RandomForestFacade, method: AggregationMethod) void {
        c.rff_set_aggregation_method(self.ptr, @intFromEnum(method));
    }

    pub fn setTreeWeight(self: *RandomForestFacade, tree_id: u32, weight: f64) void {
        c.rff_set_tree_weight(self.ptr, tree_id, weight);
    }

    pub fn getTreeWeight(self: *const RandomForestFacade, tree_id: u32) f64 {
        return c.rff_get_tree_weight(self.ptr, tree_id);
    }

    pub fn resetTreeWeights(self: *RandomForestFacade) void {
        c.rff_reset_tree_weights(self.ptr);
    }

    pub fn trackSample(self: *const RandomForestFacade, sample_index: u32) SampleTracking {
        var info: SampleTracking = undefined;
        c.rff_track_sample(self.ptr, sample_index, &info.trees_influenced, &info.oob_trees);
        return info;
    }

    pub fn printSampleTracking(self: *const RandomForestFacade, sample_index: u32) void {
        c.rff_print_sample_tracking(self.ptr, sample_index);
    }

    pub fn printOobSummary(self: *const RandomForestFacade) void {
        c.rff_print_oob_summary(self.ptr);
    }

    pub fn getGlobalOobError(self: *const RandomForestFacade) f64 {
        return c.rff_get_global_oob_error(self.ptr);
    }

    pub fn saveModel(self: *const RandomForestFacade, filename: [*:0]const u8) bool {
        return c.rff_save_model(self.ptr, filename) != 0;
    }

    pub fn loadModel(self: *RandomForestFacade, filename: [*:0]const u8) bool {
        return c.rff_load_model(self.ptr, filename) != 0;
    }

    pub fn printForestInfo(self: *const RandomForestFacade) void {
        c.rff_print_forest_info(self.ptr);
    }
};
