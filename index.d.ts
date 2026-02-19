/**
 * @file
 * @ingroup RF_Internal_Logic
 */
/**
 * Multi-backend GPU-accelerated Random Forest (CUDA, OpenCL, CPU, Hybrid)
 */

/** Core random forest classifier/regressor with direct control over all parameters. */
export class RandomForest {
  constructor(seed?: number)

  setNumTrees(n: number): void
  setMaxDepth(d: number): void
  setMinSamplesLeaf(m: number): void
  setMinSamplesSplit(m: number): void
  setMaxFeatures(m: number): void

  /** @param task - "classification" | "class" | "regression" | "reg" */
  setTaskType(task: string): void
  /** @param criterion - "gini" | "entropy" | "mse" | "variance" | "var" */
  setCriterion(criterion: string): void
  setRandomSeed(seed: number): void
  /** @param backend - "cuda" | "opencl" | "cl" | "cpu" | "hybrid" | "mixed" | "auto" */
  setBackend(backend: string): void

  /** Load training data from flat arrays. */
  loadData(data: number[], targets: number[], nSamples: number, nFeatures: number): void
  /** Load training data from a CSV file. Returns true on success. */
  loadCsv(filename: string, targetColumn: number, hasHeader: boolean): boolean

  /** Train the forest on loaded data. */
  fit(): void

  /** Predict a single sample (regression value or majority-vote class as float). */
  predict(sample: number[]): number
  /** Predict the class label for a single sample. */
  predictClass(sample: number[]): number
  /** Predict a batch of samples (flat row-major array). */
  predictBatch(samples: number[], nSamples: number): number[]
  /** Predict a batch using the GPU backend. */
  predictBatchGpu(samples: number[], nSamples: number): number[]
  /** Predict a batch using the GPU backend with per-tree weights. */
  predictBatchGpuWeighted(samples: number[], nSamples: number, weights: number[]): number[]

  /** Calculate out-of-bag error estimate. */
  calculateOobError(): number
  /** Get permutation importance for a single feature. */
  getFeatureImportance(featureIndex: number): number
  /** Get permutation importances for all features. */
  getFeatureImportances(): number[]
  printFeatureImportances(): void
  printForestInfo(): void

  /** Serialize the model to a binary file. Returns true on success. */
  saveModel(filename: string): boolean
  /** Load a model from a binary file. Returns true on success. */
  loadModel(filename: string): boolean
  /** Run predictions on a CSV and write results to an output CSV. */
  predictCsv(inputFile: string, outputFile: string, hasHeader: boolean): boolean

  /** Fit and append one additional tree. */
  addNewTree(): void
  /** Remove a tree by index (shifts subsequent trees). */
  removeTreeAt(treeId: number): void
  /** Re-fit a tree at the given index with a fresh bootstrap sample. */
  retrainTreeAt(treeId: number): void

  getNumTrees(): number
  getNumFeatures(): number
  getNumSamples(): number
  getMaxDepth(): number

  static accuracy(predictions: number[], actual: number[]): number
  static precision(predictions: number[], actual: number[], positiveClass: number): number
  static recall(predictions: number[], actual: number[], positiveClass: number): number
  static f1Score(predictions: number[], actual: number[], positiveClass: number): number
  static meanSquaredError(predictions: number[], actual: number[]): number
  static rSquared(predictions: number[], actual: number[]): number
}

/** High-level facade providing tree inspection, feature toggling, weighted aggregation, and OOB tracking. */
export class RandomForestFacade {
  constructor()

  initForest(): void
  /** @param backend - "cuda" | "opencl" | "cl" | "cpu" | "hybrid" | "mixed" | "auto" */
  setBackend(backend: string): void
  /** @param paramName - "n_estimators" | "max_depth" | "min_samples_leaf" | "min_samples_split" | "max_features" */
  setHyperparameter(paramName: string, value: number): void
  /** @param task - "classification" | "class" | "regression" | "reg" */
  setTaskType(task: string): void
  /** @param criterion - "gini" | "entropy" | "mse" | "variance" | "var" */
  setCriterion(criterion: string): void
  printHyperparameters(): void

  /** Load training data from a CSV (last column is the target). Returns true on success. */
  loadCsv(filename: string): boolean
  /** Train the forest. */
  train(): void

  predict(sample: number[]): number
  predictClass(sample: number[]): number
  predictBatch(samples: number[], nSamples: number): number[]
  predictBatchGpu(samples: number[], nSamples: number): number[]

  /** Inspect a tree. Returns [numNodes, maxDepth, numLeaves, numFeaturesUsed]. */
  inspectTree(treeId: number): number[]
  printTreeInfo(treeId: number): void
  printTreeStructure(treeId: number): void

  addTree(): void
  removeTree(treeId: number): void
  replaceTree(treeId: number): void
  retrainTree(treeId: number): void
  getNumTrees(): number

  enableFeature(featureIndex: number): void
  disableFeature(featureIndex: number): void
  resetFeatures(): void
  printFeatureUsage(): void
  printFeatureImportances(): void

  /** @param method - "majority" | "majority-vote" | "weighted" | "weighted-vote" | "mean" | "weighted-mean" */
  setAggregationMethod(method: string): void
  setTreeWeight(treeId: number, weight: number): void
  getTreeWeight(treeId: number): number
  resetTreeWeights(): void

  /** Track a training sample. Returns [treesInfluenced, oobTrees]. */
  trackSample(sampleIndex: number): number[]
  printSampleTracking(sampleIndex: number): void
  printOobSummary(): void
  getGlobalOobError(): number

  highlightMisclassified(predictions: number[], actual: number[]): void

  saveModel(filename: string): boolean
  loadModel(filename: string): boolean
  printForestInfo(): void
}
