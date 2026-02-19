/**
 * @file
 * @ingroup RF_Wrappers
 */
//
// Matthew Abbott 2025
// C# wrapper for Facaded Random Forest
//
// Build the Rust library with: cargo build --release --features cbindings
// Link against: libfacaded_random_forest.so (Linux)
//               libfacaded_random_forest.dylib (macOS)
//               facaded_random_forest.dll (Windows)
//
// Usage:
//   using FacadedRandomForest;
//
//   using var rf = new RandomForest();
//   rf.LoadCsv("data.csv", -1, true);
//   rf.Fit();
//   double pred = rf.Predict(new double[] { 5.1, 3.5, 1.4, 0.2 });
//

using System;
using System.Runtime.InteropServices;

namespace FacadedRandomForest
{
    // ═══════════════════════════════════════════════════════════════════
    // Enums
    // ═══════════════════════════════════════════════════════════════════

    public enum TaskType : int
    {
        Classification = 0,
        Regression = 1,
    }

    public enum Criterion : int
    {
        Gini = 0,
        Entropy = 1,
        MSE = 2,
        VarianceReduction = 3,
    }

    public enum AggregationMethod : int
    {
        MajorityVote = 0,
        WeightedVote = 1,
        Mean = 2,
        WeightedMean = 3,
    }

    // ═══════════════════════════════════════════════════════════════════
    // Result types
    // ═══════════════════════════════════════════════════════════════════

    public struct TreeInspection
    {
        public int NumNodes;
        public int MaxDepth;
        public int NumLeaves;
        public int NumFeaturesUsed;
    }

    public struct SampleTracking
    {
        public int TreesInfluenced;
        public int OobTrees;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Native imports
    // ═══════════════════════════════════════════════════════════════════

    internal static class Native
    {
        private const string Lib = "facaded_random_forest";

        // ── TRandomForest ──

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr rf_create();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr rf_create_with_seed(ulong seed);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_destroy(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_num_trees(IntPtr rf, uint n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_max_depth(IntPtr rf, int d);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_min_samples_leaf(IntPtr rf, int m);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_min_samples_split(IntPtr rf, int m);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_max_features(IntPtr rf, int m);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_task_type(IntPtr rf, int taskType);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_criterion(IntPtr rf, int criterion);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_set_random_seed(IntPtr rf, ulong seed);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern void rf_set_backend(IntPtr rf, string backend);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_load_data(IntPtr rf, double[] data, double[] targets, uint nSamples, uint nFeatures);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rf_load_csv(IntPtr rf, string filename, int targetColumn, int hasHeader);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_fit(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_predict(IntPtr rf, double[] sample, uint nFeatures);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int rf_predict_class(IntPtr rf, double[] sample, uint nFeatures);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_predict_batch(IntPtr rf, double[] samples, uint nSamples, double[] output);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_predict_batch_gpu(IntPtr rf, double[] samples, uint nSamples, double[] output);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_calculate_oob_error(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_get_feature_importance(IntPtr rf, uint featureIndex);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_print_feature_importances(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_print_forest_info(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rf_save_model(IntPtr rf, string filename);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rf_load_model(IntPtr rf, string filename);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rf_predict_csv(IntPtr rf, string inputFile, string outputFile, int hasHeader);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_add_new_tree(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_remove_tree_at(IntPtr rf, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rf_retrain_tree_at(IntPtr rf, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint rf_get_num_trees(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int rf_get_num_features(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint rf_get_num_samples(IntPtr rf);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int rf_get_max_depth(IntPtr rf);

        // Static metrics

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_accuracy(double[] predictions, double[] actual, uint n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_precision(double[] predictions, double[] actual, uint n, int positiveClass);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_recall(double[] predictions, double[] actual, uint n, int positiveClass);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_f1_score(double[] predictions, double[] actual, uint n, int positiveClass);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_mean_squared_error(double[] predictions, double[] actual, uint n);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rf_r_squared(double[] predictions, double[] actual, uint n);

        // ── TRandomForestFacade ──

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr rff_create();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_destroy(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_init_forest(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern void rff_set_backend(IntPtr f, string backend);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern void rff_set_hyperparameter(IntPtr f, string name, int value);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_set_task_type(IntPtr f, int taskType);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_set_criterion(IntPtr f, int criterion);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_hyperparameters(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rff_load_csv(IntPtr f, string filename);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_train(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rff_predict(IntPtr f, double[] sample, uint nFeatures);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int rff_predict_class(IntPtr f, double[] sample, uint nFeatures);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_predict_batch(IntPtr f, double[] samples, uint nSamples, uint nFeatures, double[] output);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_predict_batch_gpu(IntPtr f, double[] samples, uint nSamples, uint nFeatures, double[] output);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_inspect_tree(IntPtr f, uint treeId, out int numNodes, out int maxDepth, out int numLeaves, out int numFeaturesUsed);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_tree_info(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_tree_structure(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_add_tree(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_remove_tree(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_replace_tree(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_retrain_tree(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint rff_get_num_trees(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_enable_feature(IntPtr f, uint featureIndex);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_disable_feature(IntPtr f, uint featureIndex);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_reset_features(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_feature_usage(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_feature_importances(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_set_aggregation_method(IntPtr f, int method);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_set_tree_weight(IntPtr f, uint treeId, double weight);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rff_get_tree_weight(IntPtr f, uint treeId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_reset_tree_weights(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_track_sample(IntPtr f, uint sampleIndex, out int treesInfluenced, out int oobTrees);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_sample_tracking(IntPtr f, uint sampleIndex);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_oob_summary(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern double rff_get_global_oob_error(IntPtr f);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rff_save_model(IntPtr f, string filename);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        internal static extern int rff_load_model(IntPtr f, string filename);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rff_print_forest_info(IntPtr f);
    }

    // ═══════════════════════════════════════════════════════════════════
    // RandomForest (wraps TRandomForest)
    // ═══════════════════════════════════════════════════════════════════

    public class RandomForest : IDisposable
    {
        private IntPtr _ptr;
        private bool _disposed;

        public RandomForest()
        {
            _ptr = Native.rf_create();
        }

        public RandomForest(ulong seed)
        {
            _ptr = Native.rf_create_with_seed(seed);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _ptr != IntPtr.Zero)
            {
                Native.rf_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
            _disposed = true;
        }

        ~RandomForest() => Dispose(false);

        public void SetNumTrees(uint n) => Native.rf_set_num_trees(_ptr, n);
        public void SetMaxDepth(int d) => Native.rf_set_max_depth(_ptr, d);
        public void SetMinSamplesLeaf(int m) => Native.rf_set_min_samples_leaf(_ptr, m);
        public void SetMinSamplesSplit(int m) => Native.rf_set_min_samples_split(_ptr, m);
        public void SetMaxFeatures(int m) => Native.rf_set_max_features(_ptr, m);
        public void SetTaskType(TaskType t) => Native.rf_set_task_type(_ptr, (int)t);
        public void SetCriterion(Criterion c) => Native.rf_set_criterion(_ptr, (int)c);
        public void SetRandomSeed(ulong seed) => Native.rf_set_random_seed(_ptr, seed);
        public void SetBackend(string backend) => Native.rf_set_backend(_ptr, backend);

        public void LoadData(double[] data, double[] targets, uint nSamples, uint nFeatures)
        {
            Native.rf_load_data(_ptr, data, targets, nSamples, nFeatures);
        }

        public bool LoadCsv(string filename, int targetColumn = -1, bool hasHeader = true)
        {
            return Native.rf_load_csv(_ptr, filename, targetColumn, hasHeader ? 1 : 0) != 0;
        }

        public void Fit() => Native.rf_fit(_ptr);

        public double Predict(double[] sample)
        {
            return Native.rf_predict(_ptr, sample, (uint)sample.Length);
        }

        public int PredictClass(double[] sample)
        {
            return Native.rf_predict_class(_ptr, sample, (uint)sample.Length);
        }

        public double[] PredictBatch(double[] samples, uint nSamples)
        {
            var output = new double[nSamples];
            Native.rf_predict_batch(_ptr, samples, nSamples, output);
            return output;
        }

        public double[] PredictBatchGpu(double[] samples, uint nSamples)
        {
            var output = new double[nSamples];
            Native.rf_predict_batch_gpu(_ptr, samples, nSamples, output);
            return output;
        }

        public double CalculateOobError() => Native.rf_calculate_oob_error(_ptr);

        public double GetFeatureImportance(uint featureIndex)
        {
            return Native.rf_get_feature_importance(_ptr, featureIndex);
        }

        public void PrintFeatureImportances() => Native.rf_print_feature_importances(_ptr);
        public void PrintForestInfo() => Native.rf_print_forest_info(_ptr);

        public bool SaveModel(string filename) => Native.rf_save_model(_ptr, filename) != 0;
        public bool LoadModel(string filename) => Native.rf_load_model(_ptr, filename) != 0;

        public bool PredictCsv(string inputFile, string outputFile, bool hasHeader = true)
        {
            return Native.rf_predict_csv(_ptr, inputFile, outputFile, hasHeader ? 1 : 0) != 0;
        }

        public void AddNewTree() => Native.rf_add_new_tree(_ptr);
        public void RemoveTreeAt(uint treeId) => Native.rf_remove_tree_at(_ptr, treeId);
        public void RetrainTreeAt(uint treeId) => Native.rf_retrain_tree_at(_ptr, treeId);

        public uint NumTrees => Native.rf_get_num_trees(_ptr);
        public int NumFeatures => Native.rf_get_num_features(_ptr);
        public uint NumSamples => Native.rf_get_num_samples(_ptr);
        public int MaxDepth => Native.rf_get_max_depth(_ptr);

        // Static metrics

        public static double Accuracy(double[] predictions, double[] actual)
        {
            return Native.rf_accuracy(predictions, actual, (uint)predictions.Length);
        }

        public static double Precision(double[] predictions, double[] actual, int positiveClass)
        {
            return Native.rf_precision(predictions, actual, (uint)predictions.Length, positiveClass);
        }

        public static double Recall(double[] predictions, double[] actual, int positiveClass)
        {
            return Native.rf_recall(predictions, actual, (uint)predictions.Length, positiveClass);
        }

        public static double F1Score(double[] predictions, double[] actual, int positiveClass)
        {
            return Native.rf_f1_score(predictions, actual, (uint)predictions.Length, positiveClass);
        }

        public static double MeanSquaredError(double[] predictions, double[] actual)
        {
            return Native.rf_mean_squared_error(predictions, actual, (uint)predictions.Length);
        }

        public static double RSquared(double[] predictions, double[] actual)
        {
            return Native.rf_r_squared(predictions, actual, (uint)predictions.Length);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // RandomForestFacade (wraps TRandomForestFacade)
    // ═══════════════════════════════════════════════════════════════════

    public class RandomForestFacade : IDisposable
    {
        private IntPtr _ptr;
        private bool _disposed;

        public RandomForestFacade()
        {
            _ptr = Native.rff_create();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _ptr != IntPtr.Zero)
            {
                Native.rff_destroy(_ptr);
                _ptr = IntPtr.Zero;
            }
            _disposed = true;
        }

        ~RandomForestFacade() => Dispose(false);

        public void InitForest() => Native.rff_init_forest(_ptr);
        public void SetBackend(string backend) => Native.rff_set_backend(_ptr, backend);
        public void SetHyperparameter(string name, int value) => Native.rff_set_hyperparameter(_ptr, name, value);
        public void SetTaskType(TaskType t) => Native.rff_set_task_type(_ptr, (int)t);
        public void SetCriterion(Criterion c) => Native.rff_set_criterion(_ptr, (int)c);
        public void PrintHyperparameters() => Native.rff_print_hyperparameters(_ptr);

        public bool LoadCsv(string filename) => Native.rff_load_csv(_ptr, filename) != 0;
        public void Train() => Native.rff_train(_ptr);

        public double Predict(double[] sample)
        {
            return Native.rff_predict(_ptr, sample, (uint)sample.Length);
        }

        public int PredictClass(double[] sample)
        {
            return Native.rff_predict_class(_ptr, sample, (uint)sample.Length);
        }

        public double[] PredictBatch(double[] samples, uint nSamples, uint nFeatures)
        {
            var output = new double[nSamples];
            Native.rff_predict_batch(_ptr, samples, nSamples, nFeatures, output);
            return output;
        }

        public double[] PredictBatchGpu(double[] samples, uint nSamples, uint nFeatures)
        {
            var output = new double[nSamples];
            Native.rff_predict_batch_gpu(_ptr, samples, nSamples, nFeatures, output);
            return output;
        }

        public TreeInspection InspectTree(uint treeId)
        {
            Native.rff_inspect_tree(_ptr, treeId, out int numNodes, out int maxDepth, out int numLeaves, out int numFeaturesUsed);
            return new TreeInspection
            {
                NumNodes = numNodes,
                MaxDepth = maxDepth,
                NumLeaves = numLeaves,
                NumFeaturesUsed = numFeaturesUsed,
            };
        }

        public void PrintTreeInfo(uint treeId) => Native.rff_print_tree_info(_ptr, treeId);
        public void PrintTreeStructure(uint treeId) => Native.rff_print_tree_structure(_ptr, treeId);

        public void AddTree() => Native.rff_add_tree(_ptr);
        public void RemoveTree(uint treeId) => Native.rff_remove_tree(_ptr, treeId);
        public void ReplaceTree(uint treeId) => Native.rff_replace_tree(_ptr, treeId);
        public void RetrainTree(uint treeId) => Native.rff_retrain_tree(_ptr, treeId);
        public uint NumTrees => Native.rff_get_num_trees(_ptr);

        public void EnableFeature(uint featureIndex) => Native.rff_enable_feature(_ptr, featureIndex);
        public void DisableFeature(uint featureIndex) => Native.rff_disable_feature(_ptr, featureIndex);
        public void ResetFeatures() => Native.rff_reset_features(_ptr);
        public void PrintFeatureUsage() => Native.rff_print_feature_usage(_ptr);
        public void PrintFeatureImportances() => Native.rff_print_feature_importances(_ptr);

        public void SetAggregationMethod(AggregationMethod m) => Native.rff_set_aggregation_method(_ptr, (int)m);
        public void SetTreeWeight(uint treeId, double weight) => Native.rff_set_tree_weight(_ptr, treeId, weight);
        public double GetTreeWeight(uint treeId) => Native.rff_get_tree_weight(_ptr, treeId);
        public void ResetTreeWeights() => Native.rff_reset_tree_weights(_ptr);

        public SampleTracking TrackSample(uint sampleIndex)
        {
            Native.rff_track_sample(_ptr, sampleIndex, out int treesInfluenced, out int oobTrees);
            return new SampleTracking
            {
                TreesInfluenced = treesInfluenced,
                OobTrees = oobTrees,
            };
        }

        public void PrintSampleTracking(uint sampleIndex) => Native.rff_print_sample_tracking(_ptr, sampleIndex);
        public void PrintOobSummary() => Native.rff_print_oob_summary(_ptr);
        public double GlobalOobError => Native.rff_get_global_oob_error(_ptr);

        public bool SaveModel(string filename) => Native.rff_save_model(_ptr, filename) != 0;
        public bool LoadModel(string filename) => Native.rff_load_model(_ptr, filename) != 0;
        public void PrintForestInfo() => Native.rff_print_forest_info(_ptr);
    }
}
