# GlassBoxAI-RandomForest

## **GPU-Accelerated Random Forest**

### *Multi-Language Random Forest with CUDA/OpenCL Support and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-red.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2.svg)](https://julialang.org/)
[![C#](https://img.shields.io/badge/C%23-.NET%208.0-512BD4.svg)](https://dotnet.microsoft.com/)
[![Zig](https://img.shields.io/badge/Zig-0.13+-F7A41D.svg)](https://ziglang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-RandomForest is a production-ready, GPU-accelerated Random Forest implementation featuring:

- **Dual GPU backends**: CUDA for NVIDIA GPUs, OpenCL for AMD/Intel/cross-platform GPU acceleration
- **Multi-language bindings**: Native support for Rust, Python, Node.js, C, C++, Julia, Go, C#, and Zig
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: Kani-verified implementation for memory safety guarantees
- **Classification & Regression**: Full support for both task types with multiple split criteria
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [Language Bindings](#language-bindings)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
   - [Node.js API](#nodejs-api)
   - [C API](#c-api)
   - [C++ API](#c-api-1)
   - [Julia API](#julia-api)
   - [Go API](#go-api)
   - [C# API](#c-api-2)
   - [Zig API](#zig-api)
   7. [CLI Reference](#cli-reference)
   8. [Formal Verification with Kani](#formal-verification-with-kani)
   9. [CISA/NSA Compliance](#cisansa-compliance)
   10. [License](#license)
   11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Classification** | Majority vote and weighted vote aggregation with Gini/Entropy criteria |
| **Regression** | Mean and weighted mean aggregation with MSE/Variance Reduction criteria |
| **GPU Batch Prediction** | Parallel inference on CUDA and OpenCL with per-tree weighted variants |
| **Out-of-Bag Error** | Automatic OOB error estimation from bootstrap sampling |
| **Feature Importance** | Permutation-based feature importance scores |
| **Tree Inspection** | Per-tree node counts, depth, leaf counts, and structure printing |
| **Sample Tracking** | Trace which trees each training sample influenced |
| **Model Persistence** | Binary serialization for model save/load |
| **CSV I/O** | Load training data and batch-predict from CSV files |
| **Dynamic Forest** | Add, remove, replace, and retrain individual trees at runtime |

### GPU Backends

| Backend | Platform | Features |
|---------|----------|----------|
| **CUDA** | NVIDIA GPUs | Full batch prediction and weighted prediction kernels |
| **OpenCL** | AMD, Intel, NVIDIA, Apple | Cross-platform GPU acceleration via OpenCL 1.2+ |
| **CPU** | All platforms | Fallback CPU implementation |
| **Hybrid** | GPU + CPU | Configurable GPU/CPU split (default 80/20) |

### Multi-Language Support

| Language | Binding Technology | Status |
|----------|-------------------|--------|
| **Rust** | Native | ✓ Full API |
| **Python** | PyO3 | ✓ Full API |
| **Node.js** | napi-rs | ✓ Full API |
| **C** | FFI | ✓ Full API |
| **C++** | FFI + RAII Wrapper | ✓ Full API |
| **Julia** | ccall | ✓ Full API |
| **Go** | cgo | ✓ Full API |
| **C#** | P/Invoke | ✓ Full API |
| **Zig** | C FFI | ✓ Full API |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses (175+ proofs across 18 categories) |
| **Bounds Checking** | Verified array access |
| **FFI Safety** | Verified type widths, null guards, lifecycle ordering |
| **GPU Kernel Safety** | Verified traversal termination, bounds, division guards |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GlassBoxAI-RandomForest                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Rust Core Library                            │   │
│  │                     (src/lib.rs)                                │   │
│  │  • CUDA/OpenCL Kernels  • Bootstrap Sampling  • OOB Error      │   │
│  │  • Decision Trees       • Feature Importance  • Model I/O      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────┼────────────────────────────────────┐   │
│  │                    Language Bindings                             │   │
│  ├────────┬────────┬──────┴──┬────────┬──────┬──────┬──────┬─────────┤│
│  │ Python │Node.js │  C/C++ │ Julia  │  Go  │  C#  │ Zig  │   CLI   ││
│  │ (PyO3) │(napi)  │  (FFI) │(ccall) │(cgo) │(P/I) │(FFI) │  (Rust) ││
│  └────────┴────────┴────────┴────────┴──────┴──────┴──────┴─────────┘│
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Security Features                            │   │
│  │  • Kani Formal Verification  • CISA Secure by Design            │   │
│  │  • Memory Safe Rust  • Comprehensive Error Handling             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-RandomForest/
│
├── src/                        # Rust source code
│   ├── lib.rs                  # Core library (forest, facade, pyo3, napi, cbindings)
│   ├── main.rs                 # CLI binary
│   ├── kernel.cu               # CUDA GPU kernels
│   ├── kernel.cl               # OpenCL GPU kernels
│   └── kani/                   # Formal verification harnesses
│       ├── mod.rs              # Module registry
│       ├── bounds_checks.rs    # Array bounds verification
│       ├── pointer_validity.rs # Pointer alignment and validity
│       ├── no_panic.rs         # No-panic guarantee
│       ├── integer_overflow.rs # Arithmetic overflow prevention
│       ├── division_by_zero.rs # Division safety
│       ├── state_consistency.rs# State invariant verification
│       ├── deadlock_free.rs    # Deadlock-free logic
│       ├── input_sanitization.rs# Input bounds and DoS prevention
│       ├── result_coverage.rs  # Result/Option handling
│       ├── memory_leaks.rs     # Memory leak prevention
│       ├── constant_time.rs    # Constant-time execution
│       ├── state_machine.rs    # State machine integrity
│       ├── enum_exhaustion.rs  # Enum exhaustive matching
│       ├── floating_point.rs   # Floating-point sanity
│       ├── resource_limits.rs  # Resource allocation limits
│       ├── ffi_safety.rs       # FFI boundary safety (all wrappers)
│       └── gpu_kernel_safety.rs# GPU kernel safety (CUDA + OpenCL)
│
├── include/                    # C/C++ headers
│   ├── facaded_random_forest.h # C API header
│   └── facaded_random_forest.hpp # C++ RAII wrapper header
│
├── go/                         # Go package
│   └── randomforest/
│       ├── go.mod              # Go module
│       └── randomforest.go     # Go bindings (cgo)
│
├── julia/                      # Julia package
│   └── FacadedRandomForest/
│       ├── Project.toml        # Julia manifest
│       └── src/
│           └── FacadedRandomForest.jl # Julia module
│
├── csharp/                     # C# wrapper
│   └── FacadedRandomForest.cs  # .NET bindings (P/Invoke)
│
├── zig/                        # Zig wrapper
│   ├── random_forest.zig       # Zig bindings (C FFI)
│   └── build.zig               # Build configuration
│
├── Cargo.toml                  # Rust manifest
├── pyproject.toml              # Python build config (maturin)
├── package.json                # Node.js package config
├── index.js                    # Node.js entry point
├── index.d.ts                  # TypeScript definitions
├── facaded_random_forest.pyi   # Python type stubs
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.75+ | Core library compilation |

### GPU Backend (at least one)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 12.0+ | NVIDIA GPU acceleration |
| **OpenCL SDK** | 1.2+ | Cross-platform GPU acceleration (AMD, Intel, NVIDIA, Apple) |

### Optional (Language Bindings)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **maturin** | 1.0+ | Python package build |
| **Node.js** | 18+ | Node.js bindings |
| **@napi-rs/cli** | 3.0+ | Node.js native module build |
| **GCC/Clang** | 11+ | C/C++ compilation |
| **Julia** | 1.6+ | Julia bindings |
| **Go** | 1.21+ | Go bindings |
| **C# / .NET SDK** | 8.0+ | C# bindings |
| **Zig** | 0.13+ | Zig bindings |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **Rust Library & CLI (CUDA + OpenCL)**

```bash
# Build release binary and library (default: CUDA + OpenCL)
cargo build --release

# Run CLI
./target/release/facaded_random_forest help
```

### **Rust Library with CPU Only**

```bash
# Build without GPU backends
cargo build --release --no-default-features
```

### **Python Bindings**

```bash
# Install maturin
pip install maturin

# Build and install
maturin develop --features pyo3

# Or build wheel
maturin build --features pyo3 --release
pip install target/wheels/facaded_random_forest-*.whl
```

### **Node.js Bindings**

```bash
# Install dependencies
npm install

# Build native module
npm run build

# Development build
npm run build:debug
```

### **C/C++ Library**

```bash
# Build shared library with C API
cargo build --release --features cbindings

# Library located at: target/release/libfacaded_random_forest.so
```

### **Julia Package**

```bash
# Build shared library first
cargo build --release --features cbindings

# Use in Julia
cd julia/FacadedRandomForest
julia --project=.
# julia> using Pkg; Pkg.instantiate()
# julia> using FacadedRandomForest
```

### **Go Package**

```bash
# Build shared library first
cargo build --release --features cbindings

# Set library path
export LD_LIBRARY_PATH=$PWD/target/release:$LD_LIBRARY_PATH

# Build Go program
cd go/randomforest
go build ./...
```

### **Build All**

```bash
# Build everything
cargo build --release
cargo build --release --features pyo3
cargo build --release --features napi
cargo build --release --features cbindings
```

---

## **Language Bindings**

### **Rust API**

```rust
use facaded_random_forest::{TRandomForest, TRandomForestFacade, TaskType, SplitCriterion, BackendKind};

fn main() {
    // ── Direct API (TRandomForest) ──
    let mut rf = TRandomForest::new();
    rf.set_num_trees(100);
    rf.set_max_depth(10);
    rf.set_task_type(TaskType::Classification);
    rf.set_criterion(SplitCriterion::Gini);
    rf.set_backend_kind(BackendKind::Auto);

    rf.load_csv("data.csv", -1, true);
    rf.fit();

    let pred = rf.predict(&[5.1, 3.5, 1.4, 0.2]);
    let class = rf.predict_class(&[5.1, 3.5, 1.4, 0.2]);
    let oob = rf.calculate_oob_error();
    rf.print_feature_importances();

    rf.save_model("model.bin");
    rf.load_model("model.bin");

    // ── Facade API (TRandomForestFacade) ──
    let mut facade = TRandomForestFacade::new();
    facade.set_hyperparameter("n_estimators", 100);
    facade.set_hyperparameter("max_depth", 10);
    facade.load_csv("data.csv");
    facade.train();

    facade.inspect_tree(0);
    facade.print_tree_structure(0);
    facade.set_tree_weight(0, 2.0);
    facade.disable_feature(3);
    facade.print_oob_summary();
}
```

#### TRandomForest Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `new()` | — | `TRandomForest` | Create with random seed |
| `new_with_seed(seed)` | `seed: u64` | `TRandomForest` | Create with fixed seed |
| `set_num_trees(n)` | `n: usize` | — | Set number of trees (clamped to MAX_TREES) |
| `set_max_depth(d)` | `d: i32` | — | Set max tree depth |
| `set_min_samples_leaf(m)` | `m: i32` | — | Set min samples per leaf |
| `set_min_samples_split(m)` | `m: i32` | — | Set min samples to split |
| `set_max_features(m)` | `m: i32` | — | Set max features per split |
| `set_task_type(t)` | `t: TaskType` | — | Classification or Regression |
| `set_criterion(c)` | `c: SplitCriterion` | — | Gini, Entropy, MSE, VarianceReduction |
| `set_random_seed(seed)` | `seed: u64` | — | Set RNG seed |
| `set_backend_kind(kind)` | `kind: BackendKind` | — | Auto, Cuda, OpenCl, Cpu, Hybrid |
| `load_data(data, targets, n, f)` | `data: &[f64], targets: &[f64], n: usize, f: usize` | — | Load flat row-major data |
| `load_csv(file, col, header)` | `file: &str, col: i32, header: bool` | `bool` | Load CSV (-1 = last column) |
| `fit()` | — | — | Train forest |
| `predict(sample)` | `sample: &[f64]` | `f64` | Predict single sample |
| `predict_class(sample)` | `sample: &[f64]` | `i32` | Predict class label |
| `predict_batch(samples, n)` | `samples: &[f64], n: usize` | `Vec<f64>` | CPU batch prediction |
| `predict_batch_gpu(samples, n)` | `samples: &[f64], n: usize` | `Vec<f64>` | GPU batch prediction |
| `predict_batch_gpu_weighted(s, n, w)` | `s: &[f64], n: usize, w: &[f64]` | `Vec<f64>` | Weighted GPU batch |
| `calculate_oob_error()` | — | `f64` | Out-of-bag error |
| `get_feature_importance(i)` | `i: usize` | `f64` | Single feature importance |
| `print_feature_importances()` | — | — | Print all importances |
| `print_forest_info()` | — | — | Print forest summary |
| `save_model(file)` | `file: &str` | `bool` | Serialize to binary |
| `load_model(file)` | `file: &str` | `bool` | Deserialize from binary |
| `predict_csv(in, out, header)` | `in: &str, out: &str, header: bool` | `bool` | CSV batch predict |
| `add_new_tree()` | — | — | Fit and append one tree |
| `remove_tree_at(id)` | `id: usize` | — | Remove tree by index |
| `retrain_tree_at(id)` | `id: usize` | — | Re-fit tree at index |
| `get_num_trees()` | — | `usize` | Number of trees |
| `get_num_features()` | — | `i32` | Number of features |
| `get_num_samples()` | — | `usize` | Number of samples |
| `get_max_depth_val()` | — | `i32` | Max depth setting |

#### TRandomForestFacade Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `new()` | — | `TRandomForestFacade` | Create facade |
| `init_forest()` | — | — | Mark forest initialized |
| `set_backend(kind)` | `kind: BackendKind` | — | Set compute backend |
| `set_hyperparameter(name, val)` | `name: &str, val: i32` | — | Set by name (see below) |
| `set_task_type(t)` | `t: TaskType` | — | Classification or Regression |
| `set_criterion(c)` | `c: SplitCriterion` | — | Set split criterion |
| `print_hyperparameters()` | — | — | Print all settings |
| `load_csv(file)` | `file: &str` | `bool` | Load CSV (last col = target) |
| `train()` | — | — | Train forest |
| `predict(sample)` | `sample: &[f64]` | `f64` | Predict (uses aggregation method) |
| `predict_class(sample)` | `sample: &[f64]` | `i32` | Predict class label |
| `predict_batch(samples, n)` | `samples: &[f64], n: usize` | `Vec<f64>` | Batch predict |
| `predict_batch_gpu(samples, n)` | `samples: &[f64], n: usize` | `Vec<f64>` | GPU batch predict |
| `inspect_tree(id)` | `id: usize` | `TreeInfo` | Get tree statistics |
| `print_tree_info(id)` | `id: usize` | — | Print tree summary |
| `print_tree_structure(id)` | `id: usize` | — | Print full tree |
| `add_tree()` | — | — | Add one tree |
| `remove_tree(id)` | `id: usize` | — | Remove tree |
| `replace_tree(id)` | `id: usize` | — | Replace tree |
| `retrain_tree(id)` | `id: usize` | — | Retrain tree |
| `get_num_trees()` | — | `usize` | Number of trees |
| `enable_feature(i)` | `i: usize` | — | Enable feature |
| `disable_feature(i)` | `i: usize` | — | Disable feature |
| `reset_features()` | — | — | Enable all features |
| `print_feature_usage()` | — | — | Print split counts |
| `print_feature_importances()` | — | — | Print importances |
| `set_aggregation_method(m)` | `m: AggregationMethod` | — | MajorityVote, WeightedVote, Mean, WeightedMean |
| `set_tree_weight(id, w)` | `id: usize, w: f64` | — | Set tree weight |
| `get_tree_weight(id)` | `id: usize` | `f64` | Get tree weight |
| `reset_tree_weights()` | — | — | Reset all weights to 1.0 |
| `track_sample(i)` | `i: usize` | `SampleTrackInfo` | Track training sample |
| `print_sample_tracking(i)` | `i: usize` | — | Print sample tracking |
| `print_oob_summary()` | — | — | Print OOB summary |
| `get_global_oob_error()` | — | `f64` | Global OOB error |
| `highlight_misclassified(p, a)` | `p: &[f64], a: &[f64]` | — | Print misclassified |
| `save_model(file)` | `file: &str` | `bool` | Save model |
| `load_model(file)` | `file: &str` | `bool` | Load model |
| `print_forest_info()` | — | — | Print forest info |

#### Hyperparameter Names

| Name | Description |
|------|-------------|
| `n_estimators` | Number of trees |
| `max_depth` | Maximum tree depth |
| `min_samples_leaf` | Minimum samples per leaf node |
| `min_samples_split` | Minimum samples to split a node |
| `max_features` | Maximum features considered per split |

#### Static Metric Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `accuracy(p, a)` | `p: &[f64], a: &[f64]` | `f64` | Classification accuracy |
| `precision(p, a, cls)` | `p: &[f64], a: &[f64], cls: i32` | `f64` | Precision for class |
| `recall(p, a, cls)` | `p: &[f64], a: &[f64], cls: i32` | `f64` | Recall for class |
| `f1_score(p, a, cls)` | `p: &[f64], a: &[f64], cls: i32` | `f64` | F1 score for class |
| `mean_squared_error(p, a)` | `p: &[f64], a: &[f64]` | `f64` | MSE |
| `r_squared(p, a)` | `p: &[f64], a: &[f64]` | `f64` | R² score |

### **Python API**

```python
from facaded_random_forest import RandomForest, RandomForestFacade

# ── Direct API ──
rf = RandomForest(seed=42)
rf.set_num_trees(100)
rf.set_max_depth(10)
rf.set_task_type("classification")
rf.set_criterion("gini")
rf.set_backend("auto")

rf.load_csv("data.csv", -1, True)
rf.fit()

pred = rf.predict([5.1, 3.5, 1.4, 0.2])
cls = rf.predict_class([5.1, 3.5, 1.4, 0.2])
importances = rf.get_feature_importances()
oob = rf.calculate_oob_error()

rf.save_model("model.bin")
rf.load_model("model.bin")

# Static metrics
acc = RandomForest.accuracy(predictions, actual)
f1 = RandomForest.f1_score(predictions, actual, positive_class=1)

# ── Facade API ──
facade = RandomForestFacade()
facade.set_hyperparameter("n_estimators", 100)
facade.set_task_type("classification")
facade.load_csv("data.csv")
facade.train()

info = facade.inspect_tree(0)  # (num_nodes, max_depth, num_leaves, num_features_used)
facade.set_aggregation_method("weighted-vote")
facade.set_tree_weight(0, 2.0)
facade.disable_feature(3)
facade.print_oob_summary()
facade.save_model("model.bin")
```

### **Node.js API**

```javascript
const { RandomForest, RandomForestFacade } = require('facaded-random-forest');

// ── Direct API ──
const rf = new RandomForest(42);
rf.setNumTrees(100);
rf.setMaxDepth(10);
rf.setTaskType('classification');
rf.setCriterion('gini');
rf.setBackend('auto');

rf.loadCsv('data.csv', -1, true);
rf.fit();

const pred = rf.predict([5.1, 3.5, 1.4, 0.2]);
const cls = rf.predictClass([5.1, 3.5, 1.4, 0.2]);
const importances = rf.getFeatureImportances();
const oob = rf.calculateOobError();

rf.saveModel('model.bin');
rf.loadModel('model.bin');

// Static metrics
const acc = RandomForest.accuracy(predictions, actual);
const f1 = RandomForest.f1Score(predictions, actual, 1);

// ── Facade API ──
const facade = new RandomForestFacade();
facade.setHyperparameter('n_estimators', 100);
facade.setTaskType('classification');
facade.loadCsv('data.csv');
facade.train();

const info = facade.inspectTree(0); // [numNodes, maxDepth, numLeaves, numFeaturesUsed]
facade.setAggregationMethod('weighted-vote');
facade.setTreeWeight(0, 2.0);
facade.disableFeature(3);
facade.printOobSummary();
facade.saveModel('model.bin');
```

### **C API**

```c
#include "facaded_random_forest.h"

int main() {
    // ── Direct API ──
    TRandomForest* rf = rf_create();
    rf_set_num_trees(rf, 100);
    rf_set_max_depth(rf, 10);
    rf_set_task_type(rf, 0);       // 0 = Classification
    rf_set_criterion(rf, 0);       // 0 = Gini
    rf_set_backend(rf, "auto");

    rf_load_csv(rf, "data.csv", -1, 1);
    rf_fit(rf);

    double sample[] = {5.1, 3.5, 1.4, 0.2};
    double pred = rf_predict(rf, sample, 4);
    int cls = rf_predict_class(rf, sample, 4);
    double oob = rf_calculate_oob_error(rf);

    rf_save_model(rf, "model.bin");
    rf_destroy(rf);

    // ── Facade API ──
    TRandomForestFacade* f = rff_create();
    rff_set_hyperparameter(f, "n_estimators", 100);
    rff_set_task_type(f, 0);
    rff_load_csv(f, "data.csv");
    rff_train(f);

    int num_nodes, max_depth, num_leaves, num_features_used;
    rff_inspect_tree(f, 0, &num_nodes, &max_depth, &num_leaves, &num_features_used);

    rff_set_aggregation_method(f, 1);  // 1 = WeightedVote
    rff_set_tree_weight(f, 0, 2.0);
    rff_disable_feature(f, 3);
    rff_save_model(f, "model.bin");
    rff_destroy(f);

    return 0;
}
```

### **C++ API**

```cpp
#include "facaded_random_forest.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace rf;

    // ── Direct API ──
    RandomForest forest(42);
    forest.setNumTrees(100);
    forest.setMaxDepth(10);
    forest.setTaskType(TaskType::Classification);
    forest.setCriterion(Criterion::Gini);
    forest.setBackend("auto");

    forest.loadCsv("data.csv", -1, true);
    forest.fit();

    std::vector<double> sample = {5.1, 3.5, 1.4, 0.2};
    double pred = forest.predict(sample);
    int cls = forest.predictClass(sample);
    double oob = forest.calculateOobError();

    forest.saveModel("model.bin");

    // ── Facade API ──
    RandomForestFacade facade;
    facade.setHyperparameter("n_estimators", 100);
    facade.setTaskType(TaskType::Classification);
    facade.loadCsv("data.csv");
    facade.train();

    auto info = facade.inspectTree(0);
    std::cout << "Nodes: " << info.numNodes << " Depth: " << info.maxDepth << std::endl;

    facade.setAggregationMethod(AggregationMethod::WeightedVote);
    facade.setTreeWeight(0, 2.0);
    facade.disableFeature(3);
    facade.saveModel("model.bin");

    return 0;
}
```

### **Julia API**

```julia
using FacadedRandomForest

# ── Direct API ──
rf = RandomForest()
set_num_trees!(rf, 100)
set_max_depth!(rf, 10)
set_task_type!(rf, Classification)
set_criterion!(rf, Gini)
set_backend!(rf, "auto")

load_csv!(rf, "data.csv", -1, true)
fit!(rf)

pred = predict(rf, [5.1, 3.5, 1.4, 0.2])
cls = predict_class(rf, [5.1, 3.5, 1.4, 0.2])
oob = calculate_oob_error(rf)
save_model(rf, "model.bin")

# ── Facade API ──
facade = RandomForestFacade()
set_hyperparameter!(facade, "n_estimators", 100)
set_task_type!(facade, Classification)
load_csv!(facade, "data.csv")
train!(facade)

info = inspect_tree(facade, 0)
set_tree_weight!(facade, 0, 2.0)
disable_feature!(facade, 3)
save_model(facade, "model.bin")
```

### **Go API**

```go
package main

import (
    "fmt"
    "github.com/GlassBoxAI/RandomForest/go/randomforest"
)

func main() {
    // ── Direct API ──
    rf := randomforest.NewRandomForestWithSeed(42)
    defer rf.Close()

    rf.SetNumTrees(100)
    rf.SetMaxDepth(10)
    rf.SetTaskType(randomforest.Classification)
    rf.SetCriterion(randomforest.Gini)
    rf.SetBackend("auto")

    rf.LoadCSV("data.csv", -1, true)
    rf.Fit()

    pred := rf.Predict([]float64{5.1, 3.5, 1.4, 0.2})
    cls := rf.PredictClass([]float64{5.1, 3.5, 1.4, 0.2})
    oob := rf.CalculateOOBError()
    fmt.Println("Prediction:", pred, "Class:", cls, "OOB:", oob)

    rf.SaveModel("model.bin")

    // ── Facade API ──
    facade := randomforest.NewRandomForestFacade()
    defer facade.Close()

    facade.SetHyperparameter("n_estimators", 100)
    facade.SetTaskType(randomforest.Classification)
    facade.LoadCSV("data.csv")
    facade.Train()

    info := facade.InspectTree(0)
    fmt.Println("Nodes:", info.NumNodes, "Depth:", info.MaxDepth)

    facade.SetTreeWeight(0, 2.0)
    facade.DisableFeature(3)
    facade.SaveModel("model.bin")
}
```

### **C# API**

```csharp
using FacadedRandomForest;

// ── Direct API ──
using var rf = new RandomForest(42);
rf.SetNumTrees(100);
rf.SetMaxDepth(10);
rf.SetTaskType(TaskType.Classification);
rf.SetCriterion(Criterion.Gini);
rf.SetBackend("auto");

rf.LoadCsv("data.csv", -1, true);
rf.Fit();

double pred = rf.Predict(new double[] { 5.1, 3.5, 1.4, 0.2 });
int cls = rf.PredictClass(new double[] { 5.1, 3.5, 1.4, 0.2 });
double oob = rf.CalculateOobError();
Console.WriteLine($"Prediction: {pred}, Class: {cls}, OOB: {oob}");

rf.SaveModel("model.bin");

// ── Facade API ──
using var facade = new RandomForestFacade();
facade.SetHyperparameter("n_estimators", 100);
facade.SetTaskType(TaskType.Classification);
facade.LoadCsv("data.csv");
facade.Train();

var info = facade.InspectTree(0);
Console.WriteLine($"Nodes: {info.NumNodes}, Depth: {info.MaxDepth}");

facade.SetAggregationMethod(AggregationMethod.WeightedVote);
facade.SetTreeWeight(0, 2.0);
facade.DisableFeature(3);
facade.SaveModel("model.bin");
```

### **Zig API**

```zig
const rf_mod = @import("random_forest");

pub fn main() !void {
    // ── Direct API ──
    var forest = rf_mod.RandomForest.createWithSeed(42);
    defer forest.destroy();

    forest.setNumTrees(100);
    forest.setMaxDepth(10);
    forest.setTaskType(.classification);
    forest.setCriterion(.gini);
    forest.setBackend("auto");

    _ = forest.loadCsv("data.csv", -1, true);
    forest.fit();

    const sample = [_]f64{ 5.1, 3.5, 1.4, 0.2 };
    const pred = forest.predict(&sample);
    const cls = forest.predictClass(&sample);
    const oob = forest.calculateOobError();
    _ = pred; _ = cls; _ = oob;

    _ = forest.saveModel("model.bin");

    // ── Facade API ──
    var facade = rf_mod.RandomForestFacade.create();
    defer facade.destroy();

    facade.setHyperparameter("n_estimators", 100);
    facade.setTaskType(.classification);
    _ = facade.loadCsv("data.csv");
    facade.train();

    const info = facade.inspectTree(0);
    _ = info;

    facade.setTreeWeight(0, 2.0);
    facade.disableFeature(3);
    _ = facade.saveModel("model.bin");
}
```

---

## **CLI Reference**

### Usage

```
facaded_random_forest <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `train` | Train a random forest model from CSV data |
| `predict` | Make predictions on new data |
| `info` | Display model information |
| `help` | Show help message |

### Examples

```bash
# Train a classification model
facaded_random_forest train \
    --data=train.csv --trees=100 --depth=10 \
    --task=classification --criterion=gini \
    --backend=auto --save=model.bin

# Train a regression model
facaded_random_forest train \
    --data=train.csv --trees=200 --depth=15 \
    --task=regression --criterion=mse \
    --save=model.bin

# Predict from CSV
facaded_random_forest predict \
    --model=model.bin --input=test.csv --output=predictions.csv

# Model info
facaded_random_forest info --model=model.bin
```

---

## **Formal Verification with Kani**

### Overview

The implementation includes **175+ Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs.

### Running Verification

```bash
# Install Kani
cargo install --locked kani-verifier
kani setup

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness verify_node_index_bounds
```

### Verification Harnesses (Categories 1–18)

#### 1. Strict Bound Checks
- `verify_node_index_bounds` — Node array index within MAX_NODES
- `verify_oob_index_bounds` — OOB index within MAX_SAMPLES
- `verify_tree_index_bounds` — Tree index within MAX_TREES
- `verify_feature_importances_bounds` — Feature index within MAX_FEATURES
- `verify_data_linear_index_bounds` — Linear index calculation `sample * MAX_FEATURES + feature`
- `verify_targets_bounds` — Target array index bounds
- `verify_tree_weights_bounds` — Weight array index bounds
- `verify_feature_enabled_bounds` — Feature enabled array bounds
- `verify_predict_sample_bounds` — Predict sample slice bounds
- `verify_vote_array_bounds` — Class vote counting array bounds
- `verify_tree_traversal_bounds` — Tree node child index bounds
- `verify_bootstrap_index_bounds` — Bootstrap sample modular index bounds
- `verify_feature_subset_bounds` — Feature subset selection bounds
- `verify_node_child_validation` — Node child index validation
- `verify_flat_tree_node_access` — FlatTreeNode field access safety

#### 2. Pointer Validity Proofs
- `verify_flat_tree_node_alignment` — FlatTreeNode pointer alignment for GPU transfer
- `verify_flat_tree_node_array_layout` — Contiguous array layout
- `verify_vec_pointer_validity` — Vec pointer validity when non-empty
- `verify_slice_pointer_validity` — Slice pointer validity
- `verify_box_pointer_validity` — Box raw pointer validity and alignment
- `verify_option_box_safety` — Option\<Box\> pointer patterns
- `verify_arc_pointer_validity` — Arc clone pointer equality
- `verify_stack_reference_validity` — Stack reference alignment
- `verify_mutable_reference_validity` — Mutable reference pointer operations

#### 3. No-Panic Guarantee
- `verify_flat_tree_node_default_no_panic` — FlatTreeNode::default()
- `verify_task_type_operations_no_panic` — TaskType enum operations
- `verify_split_criterion_operations_no_panic` — SplitCriterion operations
- `verify_aggregation_method_operations_no_panic` — AggregationMethod operations
- `verify_num_trees_clamp_no_panic` — Tree count clamping
- `verify_max_depth_no_panic` — Depth max operation
- `verify_min_samples_leaf_bounds_no_panic` — Min samples leaf max
- `verify_min_samples_split_bounds_no_panic` — Min samples split max
- `verify_feature_bounds_check_no_panic` — Feature index guard
- `verify_tree_bounds_check_no_panic` — Tree index guard
- `verify_node_bounds_check_no_panic` — Node index guard
- `verify_get_weight_pattern_no_panic` — Weight default pattern
- `verify_get_importance_pattern_no_panic` — Importance default pattern
- `verify_accuracy_calculation_no_panic` — Accuracy division safety
- `verify_random_int_bounds_no_panic` — RNG modular bounds
- `verify_vec_bounds_access_no_panic` — Vec guarded access
- `verify_slice_iteration_no_panic` — Slice iteration

#### 4. Integer Overflow Prevention
- `verify_data_index_no_overflow` — Data index `sample × features + feature`
- `verify_tree_count_no_overflow` — Tree count increment
- `verify_node_count_accumulation_no_overflow` — Total node count
- `verify_vote_counting_no_overflow` — Vote counter increment
- `verify_importance_accumulation_no_overflow` — Importance accumulation
- `verify_depth_calculation_no_overflow` — Depth increment
- `verify_bootstrap_index_no_overflow` — Bootstrap modular index
- `verify_feature_subset_no_overflow` — Feature subset index
- `verify_prediction_sum_no_overflow` — Prediction sum accumulation
- `verify_oob_count_no_overflow` — OOB count increment
- `verify_weight_multiplication_no_overflow` — Weight × prediction
- `verify_node_index_increment_no_overflow` — Node index increment
- `verify_rng_wrapping_safe` — RNG wrapping arithmetic
- `verify_2d_to_linear_no_overflow` — 2D to linear index

#### 5. Division-by-Zero Exclusion
- Covered in `division_by_zero.rs`

#### 6. Global State Consistency
- `verify_num_trees_invariant` — num_trees ∈ [1, MAX_TREES]
- `verify_num_features_invariant` — num_features ∈ [0, MAX_FEATURES]
- `verify_num_samples_invariant` — num_samples ≤ MAX_SAMPLES
- `verify_arc_state_consistency` — Arc shared data consistency
- `verify_aggregation_state_consistency` — Aggregation method transitions
- `verify_task_criterion_consistency` — Task/criterion default mapping
- `verify_feature_state_transitions` — Enable/disable/enable cycle
- `verify_tree_weight_state_transitions` — Set/reset weight cycle
- `verify_rng_state_isolation` — Different seeds produce different state
- `verify_forest_initialization_state` — init_forest transition
- `verify_add_remove_tree_consistency` — Add/remove tree count
- `verify_sample_tracking_bounds` — Sample index clamping

#### 7. Deadlock-Free Logic
- Covered in `deadlock_free.rs`

#### 8. Input Sanitization Bounds
- `verify_tree_traversal_bounded` — Traversal terminates within MAX_NODES
- `verify_bootstrap_loop_bounded` — Bootstrap loop ≤ MAX_SAMPLES
- `verify_feature_subset_loop_bounded` — Feature loop ≤ MAX_FEATURES
- `verify_tree_building_depth_bounded` — Depth limit enforced
- `verify_flatten_recursion_bounded` — Flatten stops at MAX_NODES
- `verify_vote_counting_bounded` — Vote scan loop bounded
- `verify_tree_iteration_bounded` — Tree iteration ≤ MAX_TREES
- `verify_sample_prediction_loop_bounded` — Sample loop bounded
- `verify_oob_loop_bounded` — OOB loop bounded
- `verify_node_inspection_bounded` — Node inspection bounded
- `verify_csv_parsing_bounded` — CSV parsing respects MAX_SAMPLES
- `verify_split_finding_bounded` — Split search O(features × samples)
- `verify_importance_loop_bounded` — Importance loop bounded
- `verify_misclassified_enumeration_bounded` — Misclassified enumeration
- `verify_depth_calculation_bounded` — Depth increment saturating

#### 9. Result Coverage Audit
- Covered in `result_coverage.rs`

#### 10. Memory Leak Prevention
- `verify_vec_deallocation` — Vec drop
- `verify_box_deallocation` — Box drop
- `verify_nested_box_deallocation` — Nested Box chain
- `verify_arc_cleanup` — Arc reference counting
- `verify_vec_of_box_cleanup` — Vec\<Box\> clear + drop
- `verify_option_box_cleanup` — Option\<Box\> = None
- `verify_flat_tree_cleanup` — FlatTree node/oob drop
- `verify_random_forest_cleanup` — Forest data/targets/trees drop
- `verify_facade_cleanup` — Facade weights/features drop
- `verify_string_cleanup` — String drop
- `verify_vec_reallocation_cleanup` — Vec realloc drop
- `verify_temporary_cleanup` — Temporary expression cleanup
- `verify_mem_take_cleanup` — mem::take pattern
- `verify_mem_replace_cleanup` — mem::replace pattern
- `verify_forest_tree_option_cleanup` — Tree slot None assignment
- `verify_iterative_drop` — Pop-and-drop loop
- `verify_scope_cleanup` — Scope-based cleanup
- `verify_hashmap_cleanup` — HashMap clear + drop
- `verify_closure_capture_cleanup` — Closure capture drop
- `verify_into_iter_cleanup` — into_iter consumption

#### 11. Constant-Time Execution
- Covered in `constant_time.rs`

#### 12. State Machine Integrity
- Covered in `state_machine.rs`

#### 13. Enum Exhaustion
- `verify_task_type_exhaustive` — All TaskType variants matched
- `verify_split_criterion_exhaustive` — All SplitCriterion variants
- `verify_aggregation_method_exhaustive` — All AggregationMethod variants
- `verify_backend_kind_exhaustive` — All BackendKind variants
- `verify_backend_kind_parsing_exhaustive` — BackendKind::from_str all inputs
- `verify_backend_task_criterion_combination` — All backend × task × criterion
- `verify_task_to_criterion_exhaustive` — Default criterion per task
- `verify_criterion_to_function_exhaustive` — Criterion to function mapping
- `verify_aggregation_weight_usage_exhaustive` — Weight usage per method
- `verify_option_exhaustive` — Option\<T\> Some/None
- `verify_result_exhaustive` — Result\<T, E\> Ok/Err
- `verify_bool_exhaustive` — true/false
- `verify_ordering_exhaustive` — Ordering Less/Equal/Greater
- `verify_task_type_parsing_exhaustive` — Integer to TaskType
- `verify_criterion_parsing_exhaustive` — Integer to SplitCriterion
- `verify_aggregation_parsing_exhaustive` — Integer to AggregationMethod
- `verify_if_let_completeness` — if-let pattern
- `verify_while_let_termination` — while-let loop
- `verify_nested_enum_exhaustive` — Task × Criterion all 8 combinations

#### 14. Floating-Point Sanity
- `verify_gini_finite_result` — Gini probability and contribution
- `verify_entropy_zero_probability` — Entropy handles p=0
- `verify_entropy_valid_probability` — Entropy handles 0<p≤1
- `verify_mse_finite_inputs` — MSE division safety
- `verify_prediction_average_finite` — Average prediction
- `verify_weighted_sum_finite` — Weighted prediction
- `verify_threshold_comparison_special_values` — NaN comparison behavior
- `verify_importance_accumulation_finite` — Importance stays finite
- `verify_division_infinity_check` — Division by zero produces infinity/NaN
- `verify_sqrt_non_negative` — sqrt of non-negative
- `verify_log_positive` — log of positive
- `verify_round_finite` — round of finite
- `verify_abs_finite` — abs of finite
- `verify_powi_bounded` — bounded exponentiation
- `verify_float_to_int_conversion` — f64 to i32 range
- `verify_accuracy_bounded` — accuracy ∈ [0, 1]
- `verify_r_squared_range` — R² ≤ 1
- `verify_nan_comparison_behavior` — NaN comparison semantics
- `verify_infinity_comparison` — Infinity ordering
- `verify_min_max_special_values` — min/max finite preservation

#### 15. Resource Limit Compliance
- `verify_flat_tree_node_size` — FlatTreeNode < 1KB
- `verify_flat_tree_allocation` — FlatTree within 100MB budget
- `verify_forest_data_allocation` — Data arrays within budget
- `verify_forest_trees_allocation` — Tree array within budget
- `verify_facade_allocation` — Facade allocations minimal
- `verify_vec_capacity_limit` — Vec within sample limit
- `verify_bootstrap_allocation` — Bootstrap allocation bounded
- `verify_feature_subset_allocation` — Feature subset bounded
- `verify_split_indices_allocation` — Split indices bounded
- `verify_predictions_allocation` — Predictions bounded
- `verify_oob_votes_allocation` — OOB votes bounded
- `verify_gpu_nodes_allocation` — GPU node buffer bounded
- `verify_gpu_offsets_allocation` — GPU offsets bounded
- `verify_sample_tracking_allocation` — Tracking minimal
- `verify_tree_info_allocation` — Tree info bounded
- `verify_csv_row_allocation` — CSV row bounded
- `verify_csv_data_allocation` — CSV data bounded
- `verify_model_serialization_bound` — Serialization bounded
- `verify_stack_allocation_bound` — Stack arrays < 1KB
- `verify_total_system_allocation` — Base struct sizes small

### FFI Boundary Safety (Category 16)

Located in `src/kani/ffi_safety.rs` — 35 Kani proofs covering all language wrappers:

- Opaque handle lifecycle (Box::into_raw/from_raw pairing, CWE-476/416/401)
- Null pointer guard pattern (ptr.as_mut() / ptr.as_ref() returns None)
- Double-free prevention (pointer nulled after destroy)
- FlatTreeNode repr(C) layout for GPU transfer (CWE-704)
- TaskType repr(i32) matches C header values
- u32↔usize, i32↔usize, usize↔u32 cast safety (CWE-681)
- u64 seed parameter preservation across boundaries
- napi u32→u64 seed round-trip
- Slice construction bounds (n_samples × n_features, CWE-119)
- Output buffer size safety
- CStr→str fallback pattern (CWE-134)
- BackendKind::from_str handles all inputs including empty string
- Enum integer mapping parity (TaskType, Criterion, AggregationMethod)
- Python/Node.js string-to-enum parsing coverage
- Inspect/track output pointer write safety with null guards
- Static metric slice bounds
- RAII lifecycle ordering (create → use → destroy)
- Hyperparameter string parameter safety
- Boolean-to-int C conversion

### GPU Kernel Safety (Category 17)

Located in `src/kani/gpu_kernel_safety.rs` — 25 Kani proofs covering CUDA and OpenCL kernels:

- Thread index bounds check prevents OOB write to predictions[] (CWE-787)
- Sample data offset (sampleIdx × numFeatures) overflow prevention (CWE-190)
- Feature access within sample is bounded (CWE-125)
- Tree offset array access bounded (CWE-125)
- Tree node offset value within allTreeNodes bounds
- Tree traversal terminates within MAX_NODES (CWE-835)
- Traversal depth bounded
- Child index stays within node bounds during traversal
- Class label bounds check prevents OOB vote write (CWE-787)
- Negative and large class labels rejected by guard
- Vote scan loop bounds verification
- Regression average division safety (numTrees > 0, CWE-369)
- Weighted mean totalWeight > 0 guard (CWE-369)
- totalWeight accumulation stays non-negative
- FlatTreeNode field sizes match GPU struct (4/8 byte parity)
- FlatTreeNode field offsets match repr(C) layout
- Node array contiguous for GPU memcpy
- GPU sample/predictions/weights/node buffer allocations bounded
- CUDA and OpenCL task type branching parity
- Weighted kernel guard parity between CUDA and OpenCL
- Vote array size parity (both use 100)

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA** and **NSA** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **FFI boundary safety** (All 9 language bindings verified)
- [x] **GPU kernel safety** (CUDA + OpenCL kernels verified)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
