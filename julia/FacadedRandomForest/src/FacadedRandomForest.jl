## @file
## @ingroup RF_Wrappers
#
# Matthew Abbott 2025
# Julia wrapper for Facaded Random Forest
#
# Build the Rust library with: cargo build --release --features cbindings
# Point FACADED_RF_LIB to the shared library path, or place it on the system library path.
#
# Usage:
#   using FacadedRandomForest
#   rf = RandomForest()
#   load_csv!(rf, "data.csv")
#   fit!(rf)
#   pred = predict(rf, [5.1, 3.5, 1.4, 0.2])
#

module FacadedRandomForest

export RandomForest, RandomForestFacade,
       TaskType, Criterion, AggregationMethod,
       fit!, predict, predict_class, predict_batch, predict_batch_gpu,
       load_data!, load_csv!, predict_csv,
       save_model, load_model,
       set_num_trees!, set_max_depth!, set_min_samples_leaf!, set_min_samples_split!,
       set_max_features!, set_task_type!, set_criterion!, set_random_seed!, set_backend!,
       get_num_trees, get_num_features, get_num_samples, get_max_depth,
       calculate_oob_error, get_feature_importance,
       print_feature_importances, print_forest_info,
       add_tree!, remove_tree!, retrain_tree!,
       accuracy, precision, recall, f1_score, mean_squared_error, r_squared,
       init_forest!, set_hyperparameter!, print_hyperparameters,
       train!, inspect_tree, print_tree_info, print_tree_structure,
       replace_tree!,
       enable_feature!, disable_feature!, reset_features!, print_feature_usage,
       set_aggregation_method!, set_tree_weight!, get_tree_weight, reset_tree_weights!,
       track_sample, print_sample_tracking, print_oob_summary, get_global_oob_error

const _libpath = Ref{String}("")

function _libname()
    if !isempty(_libpath[])
        return _libpath[]
    end
    if haskey(ENV, "FACADED_RF_LIB")
        return ENV["FACADED_RF_LIB"]
    end
    root = joinpath(@__DIR__, "..", "..", "..")
    if Sys.islinux()
        candidate = joinpath(root, "target", "release", "libfacaded_random_forest.so")
    elseif Sys.isapple()
        candidate = joinpath(root, "target", "release", "libfacaded_random_forest.dylib")
    elseif Sys.iswindows()
        candidate = joinpath(root, "target", "release", "facaded_random_forest.dll")
    else
        candidate = ""
    end
    if isfile(candidate)
        return candidate
    end
    return "facaded_random_forest"
end

function set_library_path!(path::AbstractString)
    _libpath[] = path
end

# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

@enum TaskType begin
    Classification = 0
    Regression = 1
end

@enum Criterion begin
    Gini = 0
    Entropy = 1
    MSE = 2
    VarianceReduction = 3
end

@enum AggregationMethod begin
    MajorityVote = 0
    WeightedVote = 1
    Mean = 2
    WeightedMean = 3
end

# ═══════════════════════════════════════════════════════════════════
# TreeInspection / SampleTracking
# ═══════════════════════════════════════════════════════════════════

struct TreeInspection
    num_nodes::Int32
    max_depth::Int32
    num_leaves::Int32
    num_features_used::Int32
end

struct SampleTracking
    trees_influenced::Int32
    oob_trees::Int32
end

# ═══════════════════════════════════════════════════════════════════
# RandomForest (wraps TRandomForest)
# ═══════════════════════════════════════════════════════════════════

mutable struct RandomForest
    ptr::Ptr{Cvoid}

    function RandomForest()
        lib = _libname()
        ptr = ccall((:rf_create, lib), Ptr{Cvoid}, ())
        obj = new(ptr)
        finalizer(_rf_destroy, obj)
        return obj
    end

    function RandomForest(seed::UInt64)
        lib = _libname()
        ptr = ccall((:rf_create_with_seed, lib), Ptr{Cvoid}, (UInt64,), seed)
        obj = new(ptr)
        finalizer(_rf_destroy, obj)
        return obj
    end

    RandomForest(seed::Integer) = RandomForest(UInt64(seed))
end

function _rf_destroy(rf::RandomForest)
    if rf.ptr != C_NULL
        ccall((:rf_destroy, _libname()), Cvoid, (Ptr{Cvoid},), rf.ptr)
        rf.ptr = C_NULL
    end
end

function set_num_trees!(rf::RandomForest, n::Integer)
    ccall((:rf_set_num_trees, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), rf.ptr, UInt32(n))
end

function set_max_depth!(rf::RandomForest, d::Integer)
    ccall((:rf_set_max_depth, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(d))
end

function set_min_samples_leaf!(rf::RandomForest, m::Integer)
    ccall((:rf_set_min_samples_leaf, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(m))
end

function set_min_samples_split!(rf::RandomForest, m::Integer)
    ccall((:rf_set_min_samples_split, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(m))
end

function set_max_features!(rf::RandomForest, m::Integer)
    ccall((:rf_set_max_features, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(m))
end

function set_task_type!(rf::RandomForest, t::TaskType)
    ccall((:rf_set_task_type, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(t))
end

function set_criterion!(rf::RandomForest, c::Criterion)
    ccall((:rf_set_criterion, _libname()), Cvoid, (Ptr{Cvoid}, Int32), rf.ptr, Int32(c))
end

function set_random_seed!(rf::RandomForest, seed::Integer)
    ccall((:rf_set_random_seed, _libname()), Cvoid, (Ptr{Cvoid}, UInt64), rf.ptr, UInt64(seed))
end

function set_backend!(rf::RandomForest, backend::AbstractString)
    ccall((:rf_set_backend, _libname()), Cvoid, (Ptr{Cvoid}, Cstring), rf.ptr, backend)
end

function load_data!(rf::RandomForest, data::Matrix{Float64}, targets::Vector{Float64})
    n_samples, n_features = size(data)
    flat = Vector{Float64}(undef, n_samples * n_features)
    idx = 1
    for i in 1:n_samples
        for j in 1:n_features
            flat[idx] = data[i, j]
            idx += 1
        end
    end
    ccall((:rf_load_data, _libname()), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, UInt32, UInt32),
          rf.ptr, flat, targets, UInt32(n_samples), UInt32(n_features))
end

function load_csv!(rf::RandomForest, filename::AbstractString;
                   target_column::Integer=-1, has_header::Bool=true)
    result = ccall((:rf_load_csv, _libname()), Int32,
                   (Ptr{Cvoid}, Cstring, Int32, Int32),
                   rf.ptr, filename, Int32(target_column), Int32(has_header))
    return result != 0
end

function fit!(rf::RandomForest)
    ccall((:rf_fit, _libname()), Cvoid, (Ptr{Cvoid},), rf.ptr)
end

function predict(rf::RandomForest, sample::Vector{Float64})
    ccall((:rf_predict, _libname()), Float64,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32),
          rf.ptr, sample, UInt32(length(sample)))
end

function predict_class(rf::RandomForest, sample::Vector{Float64})
    ccall((:rf_predict_class, _libname()), Int32,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32),
          rf.ptr, sample, UInt32(length(sample)))
end

function predict_batch(rf::RandomForest, samples::Matrix{Float64})
    n_samples = size(samples, 1)
    n_features = size(samples, 2)
    flat = Vector{Float64}(undef, n_samples * n_features)
    idx = 1
    for i in 1:n_samples
        for j in 1:n_features
            flat[idx] = samples[i, j]
            idx += 1
        end
    end
    out = Vector{Float64}(undef, n_samples)
    ccall((:rf_predict_batch, _libname()), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32, Ptr{Float64}),
          rf.ptr, flat, UInt32(n_samples), out)
    return out
end

function predict_batch_gpu(rf::RandomForest, samples::Matrix{Float64})
    n_samples = size(samples, 1)
    n_features = size(samples, 2)
    flat = Vector{Float64}(undef, n_samples * n_features)
    idx = 1
    for i in 1:n_samples
        for j in 1:n_features
            flat[idx] = samples[i, j]
            idx += 1
        end
    end
    out = Vector{Float64}(undef, n_samples)
    ccall((:rf_predict_batch_gpu, _libname()), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32, Ptr{Float64}),
          rf.ptr, flat, UInt32(n_samples), out)
    return out
end

function calculate_oob_error(rf::RandomForest)
    ccall((:rf_calculate_oob_error, _libname()), Float64, (Ptr{Cvoid},), rf.ptr)
end

function get_feature_importance(rf::RandomForest, feature_index::Integer)
    ccall((:rf_get_feature_importance, _libname()), Float64,
          (Ptr{Cvoid}, UInt32), rf.ptr, UInt32(feature_index))
end

function print_feature_importances(rf::RandomForest)
    ccall((:rf_print_feature_importances, _libname()), Cvoid, (Ptr{Cvoid},), rf.ptr)
end

function print_forest_info(rf::RandomForest)
    ccall((:rf_print_forest_info, _libname()), Cvoid, (Ptr{Cvoid},), rf.ptr)
end

function save_model(rf::RandomForest, filename::AbstractString)
    result = ccall((:rf_save_model, _libname()), Int32, (Ptr{Cvoid}, Cstring), rf.ptr, filename)
    return result != 0
end

function load_model(rf::RandomForest, filename::AbstractString)
    result = ccall((:rf_load_model, _libname()), Int32, (Ptr{Cvoid}, Cstring), rf.ptr, filename)
    return result != 0
end

function predict_csv(rf::RandomForest, input_file::AbstractString, output_file::AbstractString;
                     has_header::Bool=true)
    result = ccall((:rf_predict_csv, _libname()), Int32,
                   (Ptr{Cvoid}, Cstring, Cstring, Int32),
                   rf.ptr, input_file, output_file, Int32(has_header))
    return result != 0
end

function add_tree!(rf::RandomForest)
    ccall((:rf_add_new_tree, _libname()), Cvoid, (Ptr{Cvoid},), rf.ptr)
end

function remove_tree!(rf::RandomForest, tree_id::Integer)
    ccall((:rf_remove_tree_at, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), rf.ptr, UInt32(tree_id))
end

function retrain_tree!(rf::RandomForest, tree_id::Integer)
    ccall((:rf_retrain_tree_at, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), rf.ptr, UInt32(tree_id))
end

function get_num_trees(rf::RandomForest)
    Int(ccall((:rf_get_num_trees, _libname()), UInt32, (Ptr{Cvoid},), rf.ptr))
end

function get_num_features(rf::RandomForest)
    Int(ccall((:rf_get_num_features, _libname()), Int32, (Ptr{Cvoid},), rf.ptr))
end

function get_num_samples(rf::RandomForest)
    Int(ccall((:rf_get_num_samples, _libname()), UInt32, (Ptr{Cvoid},), rf.ptr))
end

function get_max_depth(rf::RandomForest)
    Int(ccall((:rf_get_max_depth, _libname()), Int32, (Ptr{Cvoid},), rf.ptr))
end

# ═══════════════════════════════════════════════════════════════════
# Static Metrics
# ═══════════════════════════════════════════════════════════════════

function accuracy(predictions::Vector{Float64}, actual::Vector{Float64})
    @assert length(predictions) == length(actual)
    ccall((:rf_accuracy, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32),
          predictions, actual, UInt32(length(predictions)))
end

function precision(predictions::Vector{Float64}, actual::Vector{Float64}, positive_class::Integer)
    @assert length(predictions) == length(actual)
    ccall((:rf_precision, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32, Int32),
          predictions, actual, UInt32(length(predictions)), Int32(positive_class))
end

function recall(predictions::Vector{Float64}, actual::Vector{Float64}, positive_class::Integer)
    @assert length(predictions) == length(actual)
    ccall((:rf_recall, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32, Int32),
          predictions, actual, UInt32(length(predictions)), Int32(positive_class))
end

function f1_score(predictions::Vector{Float64}, actual::Vector{Float64}, positive_class::Integer)
    @assert length(predictions) == length(actual)
    ccall((:rf_f1_score, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32, Int32),
          predictions, actual, UInt32(length(predictions)), Int32(positive_class))
end

function mean_squared_error(predictions::Vector{Float64}, actual::Vector{Float64})
    @assert length(predictions) == length(actual)
    ccall((:rf_mean_squared_error, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32),
          predictions, actual, UInt32(length(predictions)))
end

function r_squared(predictions::Vector{Float64}, actual::Vector{Float64})
    @assert length(predictions) == length(actual)
    ccall((:rf_r_squared, _libname()), Float64,
          (Ptr{Float64}, Ptr{Float64}, UInt32),
          predictions, actual, UInt32(length(predictions)))
end

# ═══════════════════════════════════════════════════════════════════
# RandomForestFacade (wraps TRandomForestFacade)
# ═══════════════════════════════════════════════════════════════════

mutable struct RandomForestFacade
    ptr::Ptr{Cvoid}

    function RandomForestFacade()
        lib = _libname()
        ptr = ccall((:rff_create, lib), Ptr{Cvoid}, ())
        obj = new(ptr)
        finalizer(_rff_destroy, obj)
        return obj
    end
end

function _rff_destroy(f::RandomForestFacade)
    if f.ptr != C_NULL
        ccall((:rff_destroy, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
        f.ptr = C_NULL
    end
end

function init_forest!(f::RandomForestFacade)
    ccall((:rff_init_forest, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function set_backend!(f::RandomForestFacade, backend::AbstractString)
    ccall((:rff_set_backend, _libname()), Cvoid, (Ptr{Cvoid}, Cstring), f.ptr, backend)
end

function set_hyperparameter!(f::RandomForestFacade, name::AbstractString, value::Integer)
    ccall((:rff_set_hyperparameter, _libname()), Cvoid,
          (Ptr{Cvoid}, Cstring, Int32), f.ptr, name, Int32(value))
end

function set_task_type!(f::RandomForestFacade, t::TaskType)
    ccall((:rff_set_task_type, _libname()), Cvoid, (Ptr{Cvoid}, Int32), f.ptr, Int32(t))
end

function set_criterion!(f::RandomForestFacade, c::Criterion)
    ccall((:rff_set_criterion, _libname()), Cvoid, (Ptr{Cvoid}, Int32), f.ptr, Int32(c))
end

function print_hyperparameters(f::RandomForestFacade)
    ccall((:rff_print_hyperparameters, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function load_csv!(f::RandomForestFacade, filename::AbstractString)
    result = ccall((:rff_load_csv, _libname()), Int32, (Ptr{Cvoid}, Cstring), f.ptr, filename)
    return result != 0
end

function train!(f::RandomForestFacade)
    ccall((:rff_train, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function predict(f::RandomForestFacade, sample::Vector{Float64})
    ccall((:rff_predict, _libname()), Float64,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32),
          f.ptr, sample, UInt32(length(sample)))
end

function predict_class(f::RandomForestFacade, sample::Vector{Float64})
    ccall((:rff_predict_class, _libname()), Int32,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32),
          f.ptr, sample, UInt32(length(sample)))
end

function predict_batch(f::RandomForestFacade, samples::Matrix{Float64})
    n_samples = size(samples, 1)
    n_features = size(samples, 2)
    flat = Vector{Float64}(undef, n_samples * n_features)
    idx = 1
    for i in 1:n_samples
        for j in 1:n_features
            flat[idx] = samples[i, j]
            idx += 1
        end
    end
    out = Vector{Float64}(undef, n_samples)
    ccall((:rff_predict_batch, _libname()), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}),
          f.ptr, flat, UInt32(n_samples), UInt32(n_features), out)
    return out
end

function predict_batch_gpu(f::RandomForestFacade, samples::Matrix{Float64})
    n_samples = size(samples, 1)
    n_features = size(samples, 2)
    flat = Vector{Float64}(undef, n_samples * n_features)
    idx = 1
    for i in 1:n_samples
        for j in 1:n_features
            flat[idx] = samples[i, j]
            idx += 1
        end
    end
    out = Vector{Float64}(undef, n_samples)
    ccall((:rff_predict_batch_gpu, _libname()), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, UInt32, UInt32, Ptr{Float64}),
          f.ptr, flat, UInt32(n_samples), UInt32(n_features), out)
    return out
end

function inspect_tree(f::RandomForestFacade, tree_id::Integer)
    num_nodes = Ref{Int32}(0)
    max_depth = Ref{Int32}(0)
    num_leaves = Ref{Int32}(0)
    num_features_used = Ref{Int32}(0)
    ccall((:rff_inspect_tree, _libname()), Cvoid,
          (Ptr{Cvoid}, UInt32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
          f.ptr, UInt32(tree_id), num_nodes, max_depth, num_leaves, num_features_used)
    return TreeInspection(num_nodes[], max_depth[], num_leaves[], num_features_used[])
end

function print_tree_info(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_print_tree_info, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function print_tree_structure(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_print_tree_structure, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function add_tree!(f::RandomForestFacade)
    ccall((:rff_add_tree, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function remove_tree!(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_remove_tree, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function replace_tree!(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_replace_tree, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function retrain_tree!(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_retrain_tree, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function get_num_trees(f::RandomForestFacade)
    Int(ccall((:rff_get_num_trees, _libname()), UInt32, (Ptr{Cvoid},), f.ptr))
end

function enable_feature!(f::RandomForestFacade, feature_index::Integer)
    ccall((:rff_enable_feature, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(feature_index))
end

function disable_feature!(f::RandomForestFacade, feature_index::Integer)
    ccall((:rff_disable_feature, _libname()), Cvoid, (Ptr{Cvoid}, UInt32), f.ptr, UInt32(feature_index))
end

function reset_features!(f::RandomForestFacade)
    ccall((:rff_reset_features, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function print_feature_usage(f::RandomForestFacade)
    ccall((:rff_print_feature_usage, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function print_feature_importances(f::RandomForestFacade)
    ccall((:rff_print_feature_importances, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function set_aggregation_method!(f::RandomForestFacade, m::AggregationMethod)
    ccall((:rff_set_aggregation_method, _libname()), Cvoid, (Ptr{Cvoid}, Int32), f.ptr, Int32(m))
end

function set_tree_weight!(f::RandomForestFacade, tree_id::Integer, weight::Float64)
    ccall((:rff_set_tree_weight, _libname()), Cvoid,
          (Ptr{Cvoid}, UInt32, Float64), f.ptr, UInt32(tree_id), weight)
end

function get_tree_weight(f::RandomForestFacade, tree_id::Integer)
    ccall((:rff_get_tree_weight, _libname()), Float64,
          (Ptr{Cvoid}, UInt32), f.ptr, UInt32(tree_id))
end

function reset_tree_weights!(f::RandomForestFacade)
    ccall((:rff_reset_tree_weights, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function track_sample(f::RandomForestFacade, sample_index::Integer)
    trees_influenced = Ref{Int32}(0)
    oob_trees = Ref{Int32}(0)
    ccall((:rff_track_sample, _libname()), Cvoid,
          (Ptr{Cvoid}, UInt32, Ptr{Int32}, Ptr{Int32}),
          f.ptr, UInt32(sample_index), trees_influenced, oob_trees)
    return SampleTracking(trees_influenced[], oob_trees[])
end

function print_sample_tracking(f::RandomForestFacade, sample_index::Integer)
    ccall((:rff_print_sample_tracking, _libname()), Cvoid,
          (Ptr{Cvoid}, UInt32), f.ptr, UInt32(sample_index))
end

function print_oob_summary(f::RandomForestFacade)
    ccall((:rff_print_oob_summary, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

function get_global_oob_error(f::RandomForestFacade)
    ccall((:rff_get_global_oob_error, _libname()), Float64, (Ptr{Cvoid},), f.ptr)
end

function save_model(f::RandomForestFacade, filename::AbstractString)
    result = ccall((:rff_save_model, _libname()), Int32, (Ptr{Cvoid}, Cstring), f.ptr, filename)
    return result != 0
end

function load_model(f::RandomForestFacade, filename::AbstractString)
    result = ccall((:rff_load_model, _libname()), Int32, (Ptr{Cvoid}, Cstring), f.ptr, filename)
    return result != 0
end

function print_forest_info(f::RandomForestFacade)
    ccall((:rff_print_forest_info, _libname()), Cvoid, (Ptr{Cvoid},), f.ptr)
end

end # module
