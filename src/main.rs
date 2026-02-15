//
// Matthew Abbott 2025
// Random Forest CLI - Multi-Backend (CUDA, OpenCL, CPU, Hybrid)
//

use facaded_random_forest::*;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

fn print_help() {
    println!("Random Forest Facade CLI (Multi-Backend: CUDA/OpenCL/CPU/Hybrid)");
    println!("Matthew Abbott 2025");
    println!("Advanced Random Forest with Introspection, Tree Manipulation, and Feature Control");
    println!();
    println!("Usage: forest_facade <command> [options]");
    println!();
    println!("=== Core Commands ===");
    println!("  create              Create a new empty forest model");
    println!("  train               Train a random forest model");
    println!("  predict             Make predictions using a trained model");
    println!("  evaluate            Evaluate model on test data");
    println!("  save                Save model to file");
    println!("  load                Load model from file");
    println!("  info                Show forest hyperparameters");
    println!("  gpu-info            Show GPU/backend device information");
    println!("  help                Show this help message");
    println!();
    println!("=== Tree Inspection & Manipulation ===");
    println!("  inspect-tree        Inspect tree structure and nodes");
    println!("  tree-depth          Get depth of a specific tree");
    println!("  tree-nodes          Get node count of a specific tree");
    println!("  tree-leaves         Get leaf count of a specific tree");
    println!("  node-details        Get details of a specific node");
    println!("  prune-tree          Prune subtree at specified node");
    println!("  modify-split        Modify split threshold at node");
    println!("  modify-leaf         Modify leaf prediction value");
    println!("  convert-to-leaf     Convert node to leaf");
    println!();
    println!("=== Tree Management ===");
    println!("  add-tree            Add a new tree to the forest");
    println!("  remove-tree         Remove a tree from the forest");
    println!("  replace-tree        Replace a tree with new bootstrap sample");
    println!("  retrain-tree        Retrain a specific tree");
    println!();
    println!("=== Feature Control ===");
    println!("  enable-feature      Enable a feature for predictions");
    println!("  disable-feature     Disable a feature for predictions");
    println!("  reset-features      Reset all feature filters");
    println!("  feature-usage       Show feature usage summary");
    println!("  importance          Show feature importances");
    println!();
    println!("=== Aggregation Control ===");
    println!("  set-aggregation     Set prediction aggregation method");
    println!("  get-aggregation     Get current aggregation method");
    println!("  set-weight          Set weight for specific tree");
    println!("  get-weight          Get weight of specific tree");
    println!("  reset-weights       Reset all tree weights to 1.0");
    println!();
    println!("=== Performance Analysis ===");
    println!("  oob-summary         Show OOB error summary per tree");
    println!("  track-sample        Track which trees influence a sample");
    println!("  metrics             Calculate accuracy/MSE/F1 etc.");
    println!("  misclassified       Highlight misclassified samples");
    println!("  worst-trees         Find trees with highest error");
    println!();
    println!("=== Options ===");
    println!();
    println!("Compute Backend:");
    println!("  --backend=<type>        auto|cuda|opencl|cpu|hybrid (default: auto)");
    println!("                          auto: detect NVIDIA GPU->CUDA, AMD/Intel GPU->OpenCL, else CPU");
    println!("                          cuda: force CUDA (NVIDIA GPUs)");
    println!("                          opencl: force OpenCL (AMD, Intel, NVIDIA GPUs)");
    println!("                          cpu: force CPU-only computation");
    println!("                          hybrid: split work between GPU and CPU");
    println!();
    println!("Data & Model:");
    println!("  --input=<file>          Training input data (CSV)");
    println!("  --target=<file>         Training targets (CSV)");
    println!("  --data=<file>           Test/prediction data (CSV)");
    println!("  --model=<file>          Model file (default: forest.bin)");
    println!("  --output=<file>         Output predictions file");
    println!();
    println!("Hyperparameters:");
    println!("  --trees=<n>             Number of trees (default: 100)");
    println!("  --depth=<n>             Max tree depth (default: 10)");
    println!("  --min-leaf=<n>          Min samples per leaf (default: 1)");
    println!("  --min-split=<n>         Min samples to split node (default: 2)");
    println!("  --max-features=<n>      Max features per split (0=auto)");
    println!("  --task=<class|reg>      Task type (default: class)");
    println!("  --criterion=<c>         Split criterion: gini/entropy/mse/var");
    println!();
    println!("Tree Manipulation:");
    println!("  --tree=<id>             Tree ID for operations");
    println!("  --node=<id>             Node ID for operations");
    println!("  --threshold=<val>       New split threshold");
    println!("  --value=<val>           New leaf value");
    println!();
    println!("Feature/Weight Control:");
    println!("  --feature=<id>          Feature ID for operations");
    println!("  --weight=<val>          Tree weight (0.0-1.0)");
    println!("  --aggregation=<method>  majority|weighted|mean|weighted-mean");
    println!("  --sample=<id>           Sample ID for tracking");
    println!();
    println!("=== Examples ===");
    println!("  # Create and train forest (auto-detect GPU)");
    println!("  forest_facade create --trees=100 --depth=10 --model=rf.bin");
    println!("  forest_facade train --input=data.csv --model=rf.bin");
    println!();
    println!("  # Force OpenCL backend");
    println!("  forest_facade train --input=data.csv --model=rf.bin --backend=opencl");
    println!();
    println!("  # Use hybrid GPU+CPU mode");
    println!("  forest_facade predict --data=test.csv --model=rf.bin --backend=hybrid");
    println!();
    println!("  # CPU-only mode");
    println!("  forest_facade train --input=data.csv --model=rf.bin --backend=cpu");
}

fn get_arg(args: &[String], name: &str) -> Option<String> {
    let prefix = format!("{}=", name);
    for arg in args {
        if arg.starts_with(&prefix) {
            return Some(arg[prefix.len()..].to_string());
        }
    }
    None
}

fn get_arg_int(args: &[String], name: &str, default: i32) -> i32 {
    get_arg(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_arg_float(args: &[String], name: &str, default: f64) -> f64 {
    get_arg(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn get_backend_kind(args: &[String]) -> BackendKind {
    get_arg(args, "--backend")
        .map(|s| BackendKind::from_str(&s))
        .unwrap_or(BackendKind::Auto)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    let command = args[1].to_lowercase();
    let mut facade = TRandomForestFacade::new();
    let backend_kind = get_backend_kind(&args);
    facade.set_backend(backend_kind);

    match command.as_str() {
        "help" | "--help" | "-h" => {
            print_help();
        }

        "create" => {
            facade.init_forest();
            let trees = get_arg_int(&args, "--trees", 100);
            let depth = get_arg_int(&args, "--depth", 10);
            facade.set_hyperparameter("n_estimators", trees);
            facade.set_hyperparameter("max_depth", depth);
            println!("Created Random Forest: {} trees, depth {}", trees, depth);

            if let Some(model_file) = get_arg(&args, "--model") {
                facade.save_model(&model_file);
            }
        }

        "train" => {
            facade.init_forest();
            let input_file = get_arg(&args, "--input");
            let model_file = get_arg(&args, "--model");

            let trees = get_arg_int(&args, "--trees", 100);
            let depth = get_arg_int(&args, "--depth", 10);
            let min_leaf = get_arg_int(&args, "--min-leaf", 1);
            let min_split = get_arg_int(&args, "--min-split", 2);
            let max_features = get_arg_int(&args, "--max-features", 0);
            facade.set_hyperparameter("n_estimators", trees);
            facade.set_hyperparameter("max_depth", depth);
            facade.set_hyperparameter("min_samples_leaf", min_leaf);
            facade.set_hyperparameter("min_samples_split", min_split);
            facade.set_hyperparameter("max_features", max_features);

            if let Some(task) = get_arg(&args, "--task") {
                match task.to_lowercase().as_str() {
                    "reg" | "regression" => facade.set_task_type(TaskType::Regression),
                    _ => facade.set_task_type(TaskType::Classification),
                }
            }

            if let Some(crit) = get_arg(&args, "--criterion") {
                match crit.to_lowercase().as_str() {
                    "entropy" => facade.set_criterion(SplitCriterion::Entropy),
                    "mse" => facade.set_criterion(SplitCriterion::MSE),
                    "var" | "variance" => facade.set_criterion(SplitCriterion::VarianceReduction),
                    _ => facade.set_criterion(SplitCriterion::Gini),
                }
            }

            if input_file.is_none() {
                eprintln!("Error: --input is required");
                std::process::exit(1);
            }

            println!("Training forest from: {}", input_file.as_ref().unwrap());
            if facade.load_csv(input_file.as_ref().unwrap()) {
                facade.train();
                println!("Training complete");

                if let Some(model) = model_file {
                    facade.save_model(&model);
                }
            }
        }

        "predict" => {
            let model_file = get_arg(&args, "--model");
            let data_file = get_arg(&args, "--data");
            let output_file = get_arg(&args, "--output").unwrap_or_else(|| "predictions.csv".to_string());

            if model_file.is_none() || data_file.is_none() {
                eprintln!("Error: --model and --data are required");
                std::process::exit(1);
            }

            facade.load_model(model_file.as_ref().unwrap());
            println!("Making predictions on: {}", data_file.as_ref().unwrap());
            facade.forest.predict_csv(data_file.as_ref().unwrap(), &output_file, true);
        }

        "info" => {
            let model_file = get_arg(&args, "--model");
            if model_file.is_none() {
                eprintln!("Error: --model is required");
                std::process::exit(1);
            }
            facade.load_model(model_file.as_ref().unwrap());
            facade.print_forest_info();
        }

        "gpu-info" => {
            println!("=== Compute Backend Information ===");
            println!();

            #[cfg(feature = "cuda")]
            {
                print!("CUDA: ");
                match CudaDevice::new(0) {
                    Ok(_device) => println!("Available (NVIDIA GPU detected)"),
                    Err(e) => println!("Not available ({})", e),
                }
            }
            #[cfg(not(feature = "cuda"))]
            println!("CUDA: Not compiled (enable 'cuda' feature)");

            #[cfg(feature = "opencl")]
            {
                use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_ALL};
                print!("OpenCL: ");
                match get_all_devices(CL_DEVICE_TYPE_ALL) {
                    Ok(devices) if !devices.is_empty() => {
                        println!("Available ({} device(s))", devices.len());
                        for (i, dev_id) in devices.iter().enumerate() {
                            let dev = opencl3::device::Device::new(*dev_id);
                            let name = dev.name().unwrap_or_else(|_| "Unknown".to_string());
                            let vendor = dev.vendor().unwrap_or_else(|_| "Unknown".to_string());
                            println!("  Device {}: {} ({})", i, name.trim(), vendor.trim());
                        }
                    }
                    _ => println!("Not available"),
                }
            }
            #[cfg(not(feature = "opencl"))]
            println!("OpenCL: Not compiled (enable 'opencl' feature)");

            println!("CPU: Always available");
            println!();
            println!("Current selection: --backend={}", match backend_kind {
                BackendKind::Auto => "auto",
                BackendKind::Cuda => "cuda",
                BackendKind::OpenCl => "opencl",
                BackendKind::Cpu => "cpu",
                BackendKind::Hybrid => "hybrid",
            });
        }

        "add-tree" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.add_tree();
                println!("Added tree. Total trees: {}", facade.get_num_trees());
                facade.save_model(&model_file);
            }
        }

        "remove-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.remove_tree(tree_id);
                println!("Removed tree {}", tree_id);
                facade.save_model(&model_file);
            }
        }

        "retrain-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.retrain_tree(tree_id);
                println!("Retrained tree {}", tree_id);
                facade.save_model(&model_file);
            }
        }

        "inspect-tree" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_tree_info(tree_id);
                facade.print_tree_structure(tree_id);
            }
        }

        "feature-usage" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_feature_usage();
            }
        }

        "importance" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_feature_importances();
            }
        }

        "set-aggregation" => {
            let method = get_arg(&args, "--aggregation").unwrap_or_default();
            let agg = match method.to_lowercase().as_str() {
                "weighted" | "weighted-vote" => AggregationMethod::WeightedVote,
                "mean" => AggregationMethod::Mean,
                "weighted-mean" => AggregationMethod::WeightedMean,
                _ => AggregationMethod::MajorityVote,
            };
            facade.set_aggregation_method(agg);
            println!("Set aggregation to: {:?}", agg);
        }

        "set-weight" => {
            let tree_id = get_arg_int(&args, "--tree", 0) as usize;
            let weight = get_arg_float(&args, "--weight", 1.0);
            facade.set_tree_weight(tree_id, weight);
            println!("Set weight for tree {} to {:.2}", tree_id, weight);
        }

        "oob-summary" => {
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_oob_summary();
            }
        }

        "track-sample" => {
            let sample_id = get_arg_int(&args, "--sample", 0) as usize;
            if let Some(model_file) = get_arg(&args, "--model") {
                facade.load_model(&model_file);
                facade.print_sample_tracking(sample_id);
            }
        }

        _ => {
            eprintln!("Unknown command: {}", command);
            println!();
            print_help();
            std::process::exit(1);
        }
    }
}
