/**
 * @file
 * @ingroup RF_Wrappers
 */
//
// Matthew Abbott 2025
// Go wrapper for Facaded Random Forest
//
// Build the Rust library with: cargo build --release --features cbindings
//
// Usage:
//
//	import "github.com/GlassBoxAI/RandomForest/go/randomforest"
//
//	rf := randomforest.NewRandomForest()
//	defer rf.Close()
//	rf.LoadCSV("data.csv", -1, true)
//	rf.Fit()
//	pred := rf.Predict([]float64{5.1, 3.5, 1.4, 0.2})
//
// Set CGO_LDFLAGS and CGO_CFLAGS or use the FACADED_RF_LIB_DIR env var
// to point to the directory containing the compiled shared library.
//

package randomforest

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lfacaded_random_forest -lm -ldl -lpthread

#include "facaded_random_forest.h"
#include <stdlib.h>
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════

type TaskType int32

const (
	Classification TaskType = 0
	Regression     TaskType = 1
)

type CriterionType int32

const (
	Gini              CriterionType = 0
	Entropy           CriterionType = 1
	MSE               CriterionType = 2
	VarianceReduction CriterionType = 3
)

type AggregationMethod int32

const (
	MajorityVote AggregationMethod = 0
	WeightedVote AggregationMethod = 1
	Mean         AggregationMethod = 2
	WeightedMean AggregationMethod = 3
)

// ═══════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════

type TreeInspection struct {
	NumNodes        int32
	MaxDepth        int32
	NumLeaves       int32
	NumFeaturesUsed int32
}

type SampleTracking struct {
	TreesInfluenced int32
	OOBTrees        int32
}

// ═══════════════════════════════════════════════════════════════════
// RandomForest (wraps TRandomForest)
// ═══════════════════════════════════════════════════════════════════

type RandomForest struct {
	ptr *C.TRandomForest
}

func NewRandomForest() *RandomForest {
	rf := &RandomForest{ptr: C.rf_create()}
	runtime.SetFinalizer(rf, (*RandomForest).Close)
	return rf
}

func NewRandomForestWithSeed(seed uint64) *RandomForest {
	rf := &RandomForest{ptr: C.rf_create_with_seed(C.uint64_t(seed))}
	runtime.SetFinalizer(rf, (*RandomForest).Close)
	return rf
}

func (rf *RandomForest) Close() {
	if rf.ptr != nil {
		C.rf_destroy(rf.ptr)
		rf.ptr = nil
	}
}

func (rf *RandomForest) SetNumTrees(n int) {
	C.rf_set_num_trees(rf.ptr, C.uint32_t(n))
}

func (rf *RandomForest) SetMaxDepth(d int) {
	C.rf_set_max_depth(rf.ptr, C.int32_t(d))
}

func (rf *RandomForest) SetMinSamplesLeaf(m int) {
	C.rf_set_min_samples_leaf(rf.ptr, C.int32_t(m))
}

func (rf *RandomForest) SetMinSamplesSplit(m int) {
	C.rf_set_min_samples_split(rf.ptr, C.int32_t(m))
}

func (rf *RandomForest) SetMaxFeatures(m int) {
	C.rf_set_max_features(rf.ptr, C.int32_t(m))
}

func (rf *RandomForest) SetTaskType(t TaskType) {
	C.rf_set_task_type(rf.ptr, C.int32_t(t))
}

func (rf *RandomForest) SetCriterion(c CriterionType) {
	C.rf_set_criterion(rf.ptr, C.int32_t(c))
}

func (rf *RandomForest) SetRandomSeed(seed uint64) {
	C.rf_set_random_seed(rf.ptr, C.uint64_t(seed))
}

func (rf *RandomForest) SetBackend(backend string) {
	cs := C.CString(backend)
	defer C.free(unsafe.Pointer(cs))
	C.rf_set_backend(rf.ptr, cs)
}

func (rf *RandomForest) LoadData(data [][]float64, targets []float64) {
	nSamples := len(data)
	if nSamples == 0 {
		return
	}
	nFeatures := len(data[0])
	flat := make([]float64, nSamples*nFeatures)
	for i, row := range data {
		copy(flat[i*nFeatures:], row)
	}
	C.rf_load_data(
		rf.ptr,
		(*C.double)(unsafe.Pointer(&flat[0])),
		(*C.double)(unsafe.Pointer(&targets[0])),
		C.uint32_t(nSamples),
		C.uint32_t(nFeatures),
	)
}

func (rf *RandomForest) LoadCSV(filename string, targetColumn int, hasHeader bool) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	hdr := C.int32_t(0)
	if hasHeader {
		hdr = 1
	}
	return C.rf_load_csv(rf.ptr, cs, C.int32_t(targetColumn), hdr) != 0
}

func (rf *RandomForest) Fit() {
	C.rf_fit(rf.ptr)
}

func (rf *RandomForest) Predict(sample []float64) float64 {
	return float64(C.rf_predict(
		rf.ptr,
		(*C.double)(unsafe.Pointer(&sample[0])),
		C.uint32_t(len(sample)),
	))
}

func (rf *RandomForest) PredictClass(sample []float64) int {
	return int(C.rf_predict_class(
		rf.ptr,
		(*C.double)(unsafe.Pointer(&sample[0])),
		C.uint32_t(len(sample)),
	))
}

func (rf *RandomForest) PredictBatch(samples [][]float64) []float64 {
	nSamples := len(samples)
	if nSamples == 0 {
		return nil
	}
	nFeatures := len(samples[0])
	flat := make([]float64, nSamples*nFeatures)
	for i, row := range samples {
		copy(flat[i*nFeatures:], row)
	}
	out := make([]float64, nSamples)
	C.rf_predict_batch(
		rf.ptr,
		(*C.double)(unsafe.Pointer(&flat[0])),
		C.uint32_t(nSamples),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	return out
}

func (rf *RandomForest) PredictBatchGPU(samples [][]float64) []float64 {
	nSamples := len(samples)
	if nSamples == 0 {
		return nil
	}
	nFeatures := len(samples[0])
	flat := make([]float64, nSamples*nFeatures)
	for i, row := range samples {
		copy(flat[i*nFeatures:], row)
	}
	out := make([]float64, nSamples)
	C.rf_predict_batch_gpu(
		rf.ptr,
		(*C.double)(unsafe.Pointer(&flat[0])),
		C.uint32_t(nSamples),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	return out
}

func (rf *RandomForest) CalculateOOBError() float64 {
	return float64(C.rf_calculate_oob_error(rf.ptr))
}

func (rf *RandomForest) GetFeatureImportance(featureIndex int) float64 {
	return float64(C.rf_get_feature_importance(rf.ptr, C.uint32_t(featureIndex)))
}

func (rf *RandomForest) PrintFeatureImportances() {
	C.rf_print_feature_importances(rf.ptr)
}

func (rf *RandomForest) PrintForestInfo() {
	C.rf_print_forest_info(rf.ptr)
}

func (rf *RandomForest) SaveModel(filename string) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	return C.rf_save_model(rf.ptr, cs) != 0
}

func (rf *RandomForest) LoadModel(filename string) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	return C.rf_load_model(rf.ptr, cs) != 0
}

func (rf *RandomForest) PredictCSV(inputFile, outputFile string, hasHeader bool) bool {
	ci := C.CString(inputFile)
	defer C.free(unsafe.Pointer(ci))
	co := C.CString(outputFile)
	defer C.free(unsafe.Pointer(co))
	hdr := C.int32_t(0)
	if hasHeader {
		hdr = 1
	}
	return C.rf_predict_csv(rf.ptr, ci, co, hdr) != 0
}

func (rf *RandomForest) AddTree() {
	C.rf_add_new_tree(rf.ptr)
}

func (rf *RandomForest) RemoveTree(treeID int) {
	C.rf_remove_tree_at(rf.ptr, C.uint32_t(treeID))
}

func (rf *RandomForest) RetrainTree(treeID int) {
	C.rf_retrain_tree_at(rf.ptr, C.uint32_t(treeID))
}

func (rf *RandomForest) GetNumTrees() int {
	return int(C.rf_get_num_trees(rf.ptr))
}

func (rf *RandomForest) GetNumFeatures() int {
	return int(C.rf_get_num_features(rf.ptr))
}

func (rf *RandomForest) GetNumSamples() int {
	return int(C.rf_get_num_samples(rf.ptr))
}

func (rf *RandomForest) GetMaxDepth() int {
	return int(C.rf_get_max_depth(rf.ptr))
}

// ═══════════════════════════════════════════════════════════════════
// Static Metrics
// ═══════════════════════════════════════════════════════════════════

func Accuracy(predictions, actual []float64) float64 {
	n := len(predictions)
	return float64(C.rf_accuracy(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
	))
}

func Precision(predictions, actual []float64, positiveClass int) float64 {
	n := len(predictions)
	return float64(C.rf_precision(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
		C.int32_t(positiveClass),
	))
}

func Recall(predictions, actual []float64, positiveClass int) float64 {
	n := len(predictions)
	return float64(C.rf_recall(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
		C.int32_t(positiveClass),
	))
}

func F1Score(predictions, actual []float64, positiveClass int) float64 {
	n := len(predictions)
	return float64(C.rf_f1_score(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
		C.int32_t(positiveClass),
	))
}

func MeanSquaredError(predictions, actual []float64) float64 {
	n := len(predictions)
	return float64(C.rf_mean_squared_error(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
	))
}

func RSquared(predictions, actual []float64) float64 {
	n := len(predictions)
	return float64(C.rf_r_squared(
		(*C.double)(unsafe.Pointer(&predictions[0])),
		(*C.double)(unsafe.Pointer(&actual[0])),
		C.uint32_t(n),
	))
}

// ═══════════════════════════════════════════════════════════════════
// RandomForestFacade (wraps TRandomForestFacade)
// ═══════════════════════════════════════════════════════════════════

type RandomForestFacade struct {
	ptr *C.TRandomForestFacade
}

func NewRandomForestFacade() *RandomForestFacade {
	f := &RandomForestFacade{ptr: C.rff_create()}
	runtime.SetFinalizer(f, (*RandomForestFacade).Close)
	return f
}

func (f *RandomForestFacade) Close() {
	if f.ptr != nil {
		C.rff_destroy(f.ptr)
		f.ptr = nil
	}
}

func (f *RandomForestFacade) InitForest() {
	C.rff_init_forest(f.ptr)
}

func (f *RandomForestFacade) SetBackend(backend string) {
	cs := C.CString(backend)
	defer C.free(unsafe.Pointer(cs))
	C.rff_set_backend(f.ptr, cs)
}

func (f *RandomForestFacade) SetHyperparameter(name string, value int) {
	cs := C.CString(name)
	defer C.free(unsafe.Pointer(cs))
	C.rff_set_hyperparameter(f.ptr, cs, C.int32_t(value))
}

func (f *RandomForestFacade) SetTaskType(t TaskType) {
	C.rff_set_task_type(f.ptr, C.int32_t(t))
}

func (f *RandomForestFacade) SetCriterion(c CriterionType) {
	C.rff_set_criterion(f.ptr, C.int32_t(c))
}

func (f *RandomForestFacade) PrintHyperparameters() {
	C.rff_print_hyperparameters(f.ptr)
}

func (f *RandomForestFacade) LoadCSV(filename string) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	return C.rff_load_csv(f.ptr, cs) != 0
}

func (f *RandomForestFacade) Train() {
	C.rff_train(f.ptr)
}

func (f *RandomForestFacade) Predict(sample []float64) float64 {
	return float64(C.rff_predict(
		f.ptr,
		(*C.double)(unsafe.Pointer(&sample[0])),
		C.uint32_t(len(sample)),
	))
}

func (f *RandomForestFacade) PredictClass(sample []float64) int {
	return int(C.rff_predict_class(
		f.ptr,
		(*C.double)(unsafe.Pointer(&sample[0])),
		C.uint32_t(len(sample)),
	))
}

func (f *RandomForestFacade) PredictBatch(samples [][]float64) []float64 {
	nSamples := len(samples)
	if nSamples == 0 {
		return nil
	}
	nFeatures := len(samples[0])
	flat := make([]float64, nSamples*nFeatures)
	for i, row := range samples {
		copy(flat[i*nFeatures:], row)
	}
	out := make([]float64, nSamples)
	C.rff_predict_batch(
		f.ptr,
		(*C.double)(unsafe.Pointer(&flat[0])),
		C.uint32_t(nSamples),
		C.uint32_t(nFeatures),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	return out
}

func (f *RandomForestFacade) PredictBatchGPU(samples [][]float64) []float64 {
	nSamples := len(samples)
	if nSamples == 0 {
		return nil
	}
	nFeatures := len(samples[0])
	flat := make([]float64, nSamples*nFeatures)
	for i, row := range samples {
		copy(flat[i*nFeatures:], row)
	}
	out := make([]float64, nSamples)
	C.rff_predict_batch_gpu(
		f.ptr,
		(*C.double)(unsafe.Pointer(&flat[0])),
		C.uint32_t(nSamples),
		C.uint32_t(nFeatures),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	return out
}

func (f *RandomForestFacade) InspectTree(treeID int) TreeInspection {
	var numNodes, maxDepth, numLeaves, numFeaturesUsed C.int32_t
	C.rff_inspect_tree(f.ptr, C.uint32_t(treeID), &numNodes, &maxDepth, &numLeaves, &numFeaturesUsed)
	return TreeInspection{
		NumNodes:        int32(numNodes),
		MaxDepth:        int32(maxDepth),
		NumLeaves:       int32(numLeaves),
		NumFeaturesUsed: int32(numFeaturesUsed),
	}
}

func (f *RandomForestFacade) PrintTreeInfo(treeID int) {
	C.rff_print_tree_info(f.ptr, C.uint32_t(treeID))
}

func (f *RandomForestFacade) PrintTreeStructure(treeID int) {
	C.rff_print_tree_structure(f.ptr, C.uint32_t(treeID))
}

func (f *RandomForestFacade) AddTree() {
	C.rff_add_tree(f.ptr)
}

func (f *RandomForestFacade) RemoveTree(treeID int) {
	C.rff_remove_tree(f.ptr, C.uint32_t(treeID))
}

func (f *RandomForestFacade) ReplaceTree(treeID int) {
	C.rff_replace_tree(f.ptr, C.uint32_t(treeID))
}

func (f *RandomForestFacade) RetrainTree(treeID int) {
	C.rff_retrain_tree(f.ptr, C.uint32_t(treeID))
}

func (f *RandomForestFacade) GetNumTrees() int {
	return int(C.rff_get_num_trees(f.ptr))
}

func (f *RandomForestFacade) EnableFeature(featureIndex int) {
	C.rff_enable_feature(f.ptr, C.uint32_t(featureIndex))
}

func (f *RandomForestFacade) DisableFeature(featureIndex int) {
	C.rff_disable_feature(f.ptr, C.uint32_t(featureIndex))
}

func (f *RandomForestFacade) ResetFeatures() {
	C.rff_reset_features(f.ptr)
}

func (f *RandomForestFacade) PrintFeatureUsage() {
	C.rff_print_feature_usage(f.ptr)
}

func (f *RandomForestFacade) PrintFeatureImportances() {
	C.rff_print_feature_importances(f.ptr)
}

func (f *RandomForestFacade) SetAggregationMethod(m AggregationMethod) {
	C.rff_set_aggregation_method(f.ptr, C.int32_t(m))
}

func (f *RandomForestFacade) SetTreeWeight(treeID int, weight float64) {
	C.rff_set_tree_weight(f.ptr, C.uint32_t(treeID), C.double(weight))
}

func (f *RandomForestFacade) GetTreeWeight(treeID int) float64 {
	return float64(C.rff_get_tree_weight(f.ptr, C.uint32_t(treeID)))
}

func (f *RandomForestFacade) ResetTreeWeights() {
	C.rff_reset_tree_weights(f.ptr)
}

func (f *RandomForestFacade) TrackSample(sampleIndex int) SampleTracking {
	var treesInfluenced, oobTrees C.int32_t
	C.rff_track_sample(f.ptr, C.uint32_t(sampleIndex), &treesInfluenced, &oobTrees)
	return SampleTracking{
		TreesInfluenced: int32(treesInfluenced),
		OOBTrees:        int32(oobTrees),
	}
}

func (f *RandomForestFacade) PrintSampleTracking(sampleIndex int) {
	C.rff_print_sample_tracking(f.ptr, C.uint32_t(sampleIndex))
}

func (f *RandomForestFacade) PrintOOBSummary() {
	C.rff_print_oob_summary(f.ptr)
}

func (f *RandomForestFacade) GetGlobalOOBError() float64 {
	return float64(C.rff_get_global_oob_error(f.ptr))
}

func (f *RandomForestFacade) SaveModel(filename string) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	return C.rff_save_model(f.ptr, cs) != 0
}

func (f *RandomForestFacade) LoadModel(filename string) bool {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	return C.rff_load_model(f.ptr, cs) != 0
}

func (f *RandomForestFacade) PrintForestInfo() {
	C.rff_print_forest_info(f.ptr)
}
