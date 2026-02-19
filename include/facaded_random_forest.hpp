/**
 * @file
 * @ingroup RF_Internal_Logic
 */
//
// Matthew Abbott 2025
// C++ RAII Wrapper for Facaded Random Forest
//
// Build the Rust library with: cargo build --release --features cbindings
// Link against: -lfacaded_random_forest -L target/release
//
// Usage:
//   #include "facaded_random_forest.hpp"
//   rf::RandomForestFacade forest;
//   forest.loadCsv("data.csv");
//   forest.train();
//   double pred = forest.predict(sample.data(), sample.size());
//

#ifndef FACADED_RANDOM_FOREST_HPP
#define FACADED_RANDOM_FOREST_HPP

#include "facaded_random_forest.h"
#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace rf {

enum class TaskType : int32_t {
    Classification = 0,
    Regression = 1
};

enum class Criterion : int32_t {
    Gini = 0,
    Entropy = 1,
    MSE = 2,
    VarianceReduction = 3
};

enum class AggregationMethod : int32_t {
    MajorityVote = 0,
    WeightedVote = 1,
    Mean = 2,
    WeightedMean = 3
};

struct TreeInspection {
    int32_t numNodes;
    int32_t maxDepth;
    int32_t numLeaves;
    int32_t numFeaturesUsed;
};

struct SampleTracking {
    int32_t treesInfluenced;
    int32_t oobTrees;
};

class RandomForest {
    TRandomForest* ptr_;

public:
    RandomForest() : ptr_(rf_create()) {}
    explicit RandomForest(uint64_t seed) : ptr_(rf_create_with_seed(seed)) {}
    ~RandomForest() { if (ptr_) rf_destroy(ptr_); }

    RandomForest(const RandomForest&) = delete;
    RandomForest& operator=(const RandomForest&) = delete;
    RandomForest(RandomForest&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    RandomForest& operator=(RandomForest&& o) noexcept {
        if (this != &o) { if (ptr_) rf_destroy(ptr_); ptr_ = o.ptr_; o.ptr_ = nullptr; }
        return *this;
    }

    void setNumTrees(uint32_t n) { rf_set_num_trees(ptr_, n); }
    void setMaxDepth(int32_t d) { rf_set_max_depth(ptr_, d); }
    void setMinSamplesLeaf(int32_t m) { rf_set_min_samples_leaf(ptr_, m); }
    void setMinSamplesSplit(int32_t m) { rf_set_min_samples_split(ptr_, m); }
    void setMaxFeatures(int32_t m) { rf_set_max_features(ptr_, m); }
    void setTaskType(TaskType t) { rf_set_task_type(ptr_, static_cast<int32_t>(t)); }
    void setCriterion(Criterion c) { rf_set_criterion(ptr_, static_cast<int32_t>(c)); }
    void setRandomSeed(uint64_t seed) { rf_set_random_seed(ptr_, seed); }
    void setBackend(const std::string& backend) { rf_set_backend(ptr_, backend.c_str()); }

    void loadData(const double* data, const double* targets, uint32_t nSamples, uint32_t nFeatures) {
        rf_load_data(ptr_, data, targets, nSamples, nFeatures);
    }
    void loadData(const std::vector<double>& data, const std::vector<double>& targets,
                  uint32_t nSamples, uint32_t nFeatures) {
        rf_load_data(ptr_, data.data(), targets.data(), nSamples, nFeatures);
    }
    bool loadCsv(const std::string& filename, int32_t targetColumn = -1, bool hasHeader = true) {
        return rf_load_csv(ptr_, filename.c_str(), targetColumn, hasHeader ? 1 : 0) != 0;
    }

    void fit() { rf_fit(ptr_); }

    double predict(const double* sample, uint32_t nFeatures) const {
        return rf_predict(ptr_, sample, nFeatures);
    }
    double predict(const std::vector<double>& sample) const {
        return rf_predict(ptr_, sample.data(), static_cast<uint32_t>(sample.size()));
    }
    int32_t predictClass(const double* sample, uint32_t nFeatures) const {
        return rf_predict_class(ptr_, sample, nFeatures);
    }
    int32_t predictClass(const std::vector<double>& sample) const {
        return rf_predict_class(ptr_, sample.data(), static_cast<uint32_t>(sample.size()));
    }

    std::vector<double> predictBatch(const double* samples, uint32_t nSamples) const {
        std::vector<double> out(nSamples);
        rf_predict_batch(ptr_, samples, nSamples, out.data());
        return out;
    }
    std::vector<double> predictBatchGpu(const double* samples, uint32_t nSamples) const {
        std::vector<double> out(nSamples);
        rf_predict_batch_gpu(ptr_, samples, nSamples, out.data());
        return out;
    }

    double calculateOobError() const { return rf_calculate_oob_error(ptr_); }
    double getFeatureImportance(uint32_t idx) const { return rf_get_feature_importance(ptr_, idx); }
    void printFeatureImportances() const { rf_print_feature_importances(ptr_); }
    void printForestInfo() const { rf_print_forest_info(ptr_); }

    bool saveModel(const std::string& filename) const { return rf_save_model(ptr_, filename.c_str()) != 0; }
    bool loadModel(const std::string& filename) { return rf_load_model(ptr_, filename.c_str()) != 0; }
    bool predictCsv(const std::string& in, const std::string& out, bool hasHeader = true) const {
        return rf_predict_csv(ptr_, in.c_str(), out.c_str(), hasHeader ? 1 : 0) != 0;
    }

    void addNewTree() { rf_add_new_tree(ptr_); }
    void removeTreeAt(uint32_t id) { rf_remove_tree_at(ptr_, id); }
    void retrainTreeAt(uint32_t id) { rf_retrain_tree_at(ptr_, id); }

    uint32_t getNumTrees() const { return rf_get_num_trees(ptr_); }
    int32_t getNumFeatures() const { return rf_get_num_features(ptr_); }
    uint32_t getNumSamples() const { return rf_get_num_samples(ptr_); }
    int32_t getMaxDepth() const { return rf_get_max_depth(ptr_); }

    static double accuracy(const double* p, const double* a, uint32_t n) { return rf_accuracy(p, a, n); }
    static double precision(const double* p, const double* a, uint32_t n, int32_t cls) { return rf_precision(p, a, n, cls); }
    static double recall(const double* p, const double* a, uint32_t n, int32_t cls) { return rf_recall(p, a, n, cls); }
    static double f1Score(const double* p, const double* a, uint32_t n, int32_t cls) { return rf_f1_score(p, a, n, cls); }
    static double meanSquaredError(const double* p, const double* a, uint32_t n) { return rf_mean_squared_error(p, a, n); }
    static double rSquared(const double* p, const double* a, uint32_t n) { return rf_r_squared(p, a, n); }
};

class RandomForestFacade {
    TRandomForestFacade* ptr_;

public:
    RandomForestFacade() : ptr_(rff_create()) {}
    ~RandomForestFacade() { if (ptr_) rff_destroy(ptr_); }

    RandomForestFacade(const RandomForestFacade&) = delete;
    RandomForestFacade& operator=(const RandomForestFacade&) = delete;
    RandomForestFacade(RandomForestFacade&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
    RandomForestFacade& operator=(RandomForestFacade&& o) noexcept {
        if (this != &o) { if (ptr_) rff_destroy(ptr_); ptr_ = o.ptr_; o.ptr_ = nullptr; }
        return *this;
    }

    void initForest() { rff_init_forest(ptr_); }
    void setBackend(const std::string& backend) { rff_set_backend(ptr_, backend.c_str()); }
    void setHyperparameter(const std::string& name, int32_t value) {
        rff_set_hyperparameter(ptr_, name.c_str(), value);
    }
    void setTaskType(TaskType t) { rff_set_task_type(ptr_, static_cast<int32_t>(t)); }
    void setCriterion(Criterion c) { rff_set_criterion(ptr_, static_cast<int32_t>(c)); }
    void printHyperparameters() const { rff_print_hyperparameters(ptr_); }

    bool loadCsv(const std::string& filename) { return rff_load_csv(ptr_, filename.c_str()) != 0; }
    void train() { rff_train(ptr_); }

    double predict(const double* sample, uint32_t nFeatures) const {
        return rff_predict(ptr_, sample, nFeatures);
    }
    double predict(const std::vector<double>& sample) const {
        return rff_predict(ptr_, sample.data(), static_cast<uint32_t>(sample.size()));
    }
    int32_t predictClass(const double* sample, uint32_t nFeatures) const {
        return rff_predict_class(ptr_, sample, nFeatures);
    }
    int32_t predictClass(const std::vector<double>& sample) const {
        return rff_predict_class(ptr_, sample.data(), static_cast<uint32_t>(sample.size()));
    }
    std::vector<double> predictBatch(const double* samples, uint32_t nSamples, uint32_t nFeatures) const {
        std::vector<double> out(nSamples);
        rff_predict_batch(ptr_, samples, nSamples, nFeatures, out.data());
        return out;
    }
    std::vector<double> predictBatchGpu(const double* samples, uint32_t nSamples, uint32_t nFeatures) const {
        std::vector<double> out(nSamples);
        rff_predict_batch_gpu(ptr_, samples, nSamples, nFeatures, out.data());
        return out;
    }

    TreeInspection inspectTree(uint32_t treeId) const {
        TreeInspection info{};
        rff_inspect_tree(ptr_, treeId, &info.numNodes, &info.maxDepth, &info.numLeaves, &info.numFeaturesUsed);
        return info;
    }
    void printTreeInfo(uint32_t treeId) const { rff_print_tree_info(ptr_, treeId); }
    void printTreeStructure(uint32_t treeId) const { rff_print_tree_structure(ptr_, treeId); }

    void addTree() { rff_add_tree(ptr_); }
    void removeTree(uint32_t treeId) { rff_remove_tree(ptr_, treeId); }
    void replaceTree(uint32_t treeId) { rff_replace_tree(ptr_, treeId); }
    void retrainTree(uint32_t treeId) { rff_retrain_tree(ptr_, treeId); }
    uint32_t getNumTrees() const { return rff_get_num_trees(ptr_); }

    void enableFeature(uint32_t idx) { rff_enable_feature(ptr_, idx); }
    void disableFeature(uint32_t idx) { rff_disable_feature(ptr_, idx); }
    void resetFeatures() { rff_reset_features(ptr_); }
    void printFeatureUsage() const { rff_print_feature_usage(ptr_); }
    void printFeatureImportances() const { rff_print_feature_importances(ptr_); }

    void setAggregationMethod(AggregationMethod m) { rff_set_aggregation_method(ptr_, static_cast<int32_t>(m)); }
    void setTreeWeight(uint32_t treeId, double weight) { rff_set_tree_weight(ptr_, treeId, weight); }
    double getTreeWeight(uint32_t treeId) const { return rff_get_tree_weight(ptr_, treeId); }
    void resetTreeWeights() { rff_reset_tree_weights(ptr_); }

    SampleTracking trackSample(uint32_t sampleIndex) const {
        SampleTracking info{};
        rff_track_sample(ptr_, sampleIndex, &info.treesInfluenced, &info.oobTrees);
        return info;
    }
    void printSampleTracking(uint32_t sampleIndex) const { rff_print_sample_tracking(ptr_, sampleIndex); }
    void printOobSummary() const { rff_print_oob_summary(ptr_); }
    double getGlobalOobError() const { return rff_get_global_oob_error(ptr_); }

    bool saveModel(const std::string& filename) const { return rff_save_model(ptr_, filename.c_str()) != 0; }
    bool loadModel(const std::string& filename) { return rff_load_model(ptr_, filename.c_str()) != 0; }
    void printForestInfo() const { rff_print_forest_info(ptr_); }
};

} // namespace rf

#endif // FACADED_RANDOM_FOREST_HPP
