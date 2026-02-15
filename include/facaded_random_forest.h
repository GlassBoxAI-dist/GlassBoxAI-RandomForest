//
// Matthew Abbott 2025
// C API for Facaded Random Forest
//
// Build the Rust library with: cargo build --release --features cbindings
// Link against: target/release/libfacaded_random_forest.so (Linux)
//               target/release/libfacaded_random_forest.dylib (macOS)
//

#ifndef FACADED_RANDOM_FOREST_H
#define FACADED_RANDOM_FOREST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct TRandomForest TRandomForest;
typedef struct TRandomForestFacade TRandomForestFacade;

// Task type: 0 = Classification, 1 = Regression
// Criterion: 0 = Gini, 1 = Entropy, 2 = MSE, 3 = VarianceReduction
// Aggregation: 0 = MajorityVote, 1 = WeightedVote, 2 = Mean, 3 = WeightedMean

// ── TRandomForest ──

TRandomForest* rf_create(void);
TRandomForest* rf_create_with_seed(uint64_t seed);
void rf_destroy(TRandomForest* rf);

void rf_set_num_trees(TRandomForest* rf, uint32_t n);
void rf_set_max_depth(TRandomForest* rf, int32_t d);
void rf_set_min_samples_leaf(TRandomForest* rf, int32_t m);
void rf_set_min_samples_split(TRandomForest* rf, int32_t m);
void rf_set_max_features(TRandomForest* rf, int32_t m);
void rf_set_task_type(TRandomForest* rf, int32_t task_type);
void rf_set_criterion(TRandomForest* rf, int32_t criterion);
void rf_set_random_seed(TRandomForest* rf, uint64_t seed);
void rf_set_backend(TRandomForest* rf, const char* backend);

void rf_load_data(TRandomForest* rf, const double* data, const double* targets,
                  uint32_t n_samples, uint32_t n_features);
int32_t rf_load_csv(TRandomForest* rf, const char* filename,
                    int32_t target_column, int32_t has_header);
void rf_fit(TRandomForest* rf);

double rf_predict(const TRandomForest* rf, const double* sample, uint32_t n_features);
int32_t rf_predict_class(const TRandomForest* rf, const double* sample, uint32_t n_features);
void rf_predict_batch(const TRandomForest* rf, const double* samples,
                      uint32_t n_samples, double* out);
void rf_predict_batch_gpu(const TRandomForest* rf, const double* samples,
                          uint32_t n_samples, double* out);

double rf_calculate_oob_error(const TRandomForest* rf);
double rf_get_feature_importance(const TRandomForest* rf, uint32_t feature_index);
void rf_print_feature_importances(const TRandomForest* rf);
void rf_print_forest_info(const TRandomForest* rf);

int32_t rf_save_model(const TRandomForest* rf, const char* filename);
int32_t rf_load_model(TRandomForest* rf, const char* filename);
int32_t rf_predict_csv(const TRandomForest* rf, const char* input_file,
                       const char* output_file, int32_t has_header);

void rf_add_new_tree(TRandomForest* rf);
void rf_remove_tree_at(TRandomForest* rf, uint32_t tree_id);
void rf_retrain_tree_at(TRandomForest* rf, uint32_t tree_id);

uint32_t rf_get_num_trees(const TRandomForest* rf);
int32_t rf_get_num_features(const TRandomForest* rf);
uint32_t rf_get_num_samples(const TRandomForest* rf);
int32_t rf_get_max_depth(const TRandomForest* rf);

// Static metrics (no forest instance needed)
double rf_accuracy(const double* predictions, const double* actual, uint32_t n);
double rf_precision(const double* predictions, const double* actual, uint32_t n, int32_t positive_class);
double rf_recall(const double* predictions, const double* actual, uint32_t n, int32_t positive_class);
double rf_f1_score(const double* predictions, const double* actual, uint32_t n, int32_t positive_class);
double rf_mean_squared_error(const double* predictions, const double* actual, uint32_t n);
double rf_r_squared(const double* predictions, const double* actual, uint32_t n);

// ── TRandomForestFacade ──

TRandomForestFacade* rff_create(void);
void rff_destroy(TRandomForestFacade* f);

void rff_init_forest(TRandomForestFacade* f);
void rff_set_backend(TRandomForestFacade* f, const char* backend);
void rff_set_hyperparameter(TRandomForestFacade* f, const char* name, int32_t value);
void rff_set_task_type(TRandomForestFacade* f, int32_t task_type);
void rff_set_criterion(TRandomForestFacade* f, int32_t criterion);
void rff_print_hyperparameters(const TRandomForestFacade* f);

int32_t rff_load_csv(TRandomForestFacade* f, const char* filename);
void rff_train(TRandomForestFacade* f);

double rff_predict(const TRandomForestFacade* f, const double* sample, uint32_t n_features);
int32_t rff_predict_class(const TRandomForestFacade* f, const double* sample, uint32_t n_features);
void rff_predict_batch(const TRandomForestFacade* f, const double* samples,
                       uint32_t n_samples, uint32_t n_features, double* out);
void rff_predict_batch_gpu(const TRandomForestFacade* f, const double* samples,
                           uint32_t n_samples, uint32_t n_features, double* out);

void rff_inspect_tree(const TRandomForestFacade* f, uint32_t tree_id,
                      int32_t* out_num_nodes, int32_t* out_max_depth,
                      int32_t* out_num_leaves, int32_t* out_num_features_used);
void rff_print_tree_info(const TRandomForestFacade* f, uint32_t tree_id);
void rff_print_tree_structure(const TRandomForestFacade* f, uint32_t tree_id);

void rff_add_tree(TRandomForestFacade* f);
void rff_remove_tree(TRandomForestFacade* f, uint32_t tree_id);
void rff_replace_tree(TRandomForestFacade* f, uint32_t tree_id);
void rff_retrain_tree(TRandomForestFacade* f, uint32_t tree_id);
uint32_t rff_get_num_trees(const TRandomForestFacade* f);

void rff_enable_feature(TRandomForestFacade* f, uint32_t feature_index);
void rff_disable_feature(TRandomForestFacade* f, uint32_t feature_index);
void rff_reset_features(TRandomForestFacade* f);
void rff_print_feature_usage(const TRandomForestFacade* f);
void rff_print_feature_importances(const TRandomForestFacade* f);

void rff_set_aggregation_method(TRandomForestFacade* f, int32_t method);
void rff_set_tree_weight(TRandomForestFacade* f, uint32_t tree_id, double weight);
double rff_get_tree_weight(const TRandomForestFacade* f, uint32_t tree_id);
void rff_reset_tree_weights(TRandomForestFacade* f);

void rff_track_sample(const TRandomForestFacade* f, uint32_t sample_index,
                      int32_t* out_trees_influenced, int32_t* out_oob_trees);
void rff_print_sample_tracking(const TRandomForestFacade* f, uint32_t sample_index);
void rff_print_oob_summary(const TRandomForestFacade* f);
double rff_get_global_oob_error(const TRandomForestFacade* f);

int32_t rff_save_model(const TRandomForestFacade* f, const char* filename);
int32_t rff_load_model(TRandomForestFacade* f, const char* filename);
void rff_print_forest_info(const TRandomForestFacade* f);

#ifdef __cplusplus
}
#endif

#endif // FACADED_RANDOM_FOREST_H
