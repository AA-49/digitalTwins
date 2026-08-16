# Experiment Plan - Stage 1 and Stage 2 Correctness, Robustness, and Evidence Quality

> Project: Explainable Diabetes-Risk Digital Twin (Training3)  
> Date: 2026-08-16  
> Version: 1.0  
> Scope: `diabetes_risk_stages_1_2.ipynb` and the supplied Stage 1/Stage 2 Tech Debt Register

## 1. Purpose and decision boundary

This plan converts the tech-debt register into a falsifiable and reproducible experiment programme. It does **not** treat every bug as a research variable:

- BUG-01, BUG-03, BUG-04, and BUG-06 are correctness gates. They must be fixed and tested before any result is considered valid.
- BUG-02 is a data-identity and split-sensitivity question. Exact matching rows cannot automatically be labelled duplicate respondents because the dataset has no respondent identifier and most inputs are discrete survey codes.
- BUG-05 is an efficiency-versus-reliability experiment for permutation importance.
- BUG-07 is an explanation-sampling experiment that replaces the single selected patient with a deterministic, pre-specified case cohort.

The plan preserves the existing 21-feature order, three-class target, exact SHAP calculations, and research-only/non-causal interpretation. Stage 2 continues to explain the patient's **predicted class**; this plan does not change Stage 4 class-specific graph semantics.

## 2. Current evidence and audit findings

The current notebook uses 253,680 cleaned rows, a stratified 80/20 split with seed 42, and a 400-tree `RandomForestClassifier(class_weight='balanced_subsample', min_samples_leaf=2)`. Saved Stage 1 results are accuracy 0.8196, balanced accuracy 0.4506, macro-F1 0.4437, and Medium/prediabetes recall 0.0011.

The duplicate-profile audit found:

| Audit quantity | Count | Interpretation |
| --- | ---: | --- |
| Rows after cleaning | 253,680 | Current analysis population |
| Rows duplicating an earlier complete 21-feature + target row | 23,899 | 9.42% of rows are non-unique by observed values |
| Unique complete rows | 229,781 | Does not imply unique respondents |
| Rows duplicating an earlier 21-feature profile | 25,772 | 10.16% are non-unique by predictors |
| Test rows with a complete-row match in training | 6,410 / 50,736 | 12.63% of the current test split |
| Test rows with a feature-profile match in training | 6,923 / 50,736 | 13.65% of the current test split |

These counts show that split sensitivity is worth measuring. They do **not** prove respondent leakage because the public extract lacks a stable respondent ID and different people can submit identical discrete responses.

The local `.venv-stage3` currently contains Python 3.12.13, scikit-learn 1.9.0, SHAP 0.52.0, pandas 3.0.5, and NumPy 2.4.6. The Docker training image instead uses Python 3.11 and pins scikit-learn 1.7.2; the existing model artifact was produced with scikit-learn 1.7.2. Confirmatory experiments must therefore run inside one rebuilt, recorded Docker training image rather than mixing the local environment with the serialized model.

## 3. Research questions and falsifiable hypotheses

### RQ1 - Correctness

Can a single-source metrics and artifact pipeline produce internally consistent Stage 1 results regardless of notebook execution order?

**H1 (correctness):** After consolidation, a clean top-to-bottom notebook execution will produce one run ID, one fitted model, and identical metric values in `metrics.json` and `results_summary.json`; any mismatch greater than `1e-12` falsifies H1.

### RQ2 - Split sensitivity and minority-class prediction

How sensitive are the reported metrics to identical observed profiles appearing on both sides of the split, and can an imbalance-aware learner improve Medium/prediabetes detection without an unacceptable loss in overall class balance?

**H2a (split sensitivity):** The absolute macro-F1 difference between the respondent-row split and the feature-profile-grouped evaluation will be less than 0.03. A difference of 0.03 or more falsifies this stability hypothesis and requires both estimates to be reported.

**H2b (minority-class performance):** `BalancedRandomForestClassifier` will increase Medium/prediabetes recall by at least 0.10 absolute over the current weighted Random Forest while reducing macro-F1 by no more than 0.02. Failure of either condition falsifies H2b.

No causal statement will be made about why one estimator performs differently. In particular, the existing model's poor Medium recall cannot be attributed to the estimator choice without this controlled comparison.

### RQ3 - Local explanation coverage and fidelity

Can Stage 2 explain representative correct and incorrect predictions across all classes without cherry-picking and without violating SHAP additivity?

**H3:** For every available pre-specified confusion stratum, the pipeline will save up to three deterministic cases, and for every saved case the absolute SHAP additivity error will be at most `1e-6`. Missing strata must be reported rather than substituted post hoc.

### RQ4 - Global explanation stability and computational efficiency

Are the Stage 2 global importance conclusions stable under a larger sample and more repeats, and does limited parallelism reduce runtime within the Docker memory ceiling?

**H4a (stability):** Comparing the current 1,000-row/3-repeat setting with the 5,000-row/10-repeat reference, the top-five feature sets will have Jaccard similarity at least 0.80 and the full 21-feature rankings will have Spearman correlation at least 0.80.

**H4b (efficiency):** `n_jobs=2` will reduce wall time by at least 25% relative to `n_jobs=1`, keep peak container memory below 75% of the Docker memory limit, and reproduce importance means within `1e-10` under the same seed. If any condition fails, retain `n_jobs=1` and document the measured reason.

### Hypothesis-to-experiment matrix

| Hypothesis | Experiment | Primary evidence | Passing condition |
| --- | --- | --- | --- |
| H1 | E0 | Artifact equality and correctness tests | Metrics agree within `1e-12`; C1-C6 pass |
| H2a | E2 | Protocol A versus Protocol C macro-F1 | Absolute difference < 0.03 |
| H2b | E1 | Medium recall and macro-F1 deltas | Recall gain >= 0.10 and macro-F1 loss <= 0.02 |
| H3 | E3 | Case manifest and SHAP additivity | Every available stratum represented; error <= `1e-6` |
| H4a | E4 | Ranking correlation and overlap | Spearman >= 0.80 and top-five Jaccard >= 0.80 |
| H4b | E5 | Wall time, memory, numeric agreement | >=25% faster, <75% memory ceiling, difference <= `1e-10` |

## 4. Correctness gates before research experiments

| Gate | Related debt | Required change | Automated acceptance test |
| --- | --- | --- | --- |
| C1: One evaluation source | BUG-01, BUG-04 | Train once; evaluate once; write all report artifacts from the same immutable metrics object | JSON metric keys and values match within `1e-12`; confusion matrix written once per run |
| C2: Class-presence guard | BUG-03 | Assert model classes and held-out labels are exactly `{0,1,2}` before ROC-AUC | Intentionally remove one class in a fixture and verify a descriptive failure naming that class |
| C3: SHAP normalization | BUG-06 | Normalize list/array outputs and validate `expected_value` shape before indexing | Fixtures for list-style and 3D-array SHAP outputs; invalid scalar/mismatched shape fails loudly |
| C4: SHAP fidelity | BUG-06 | Check `base_value + sum(contributions)` against the explained class output | Error no greater than `1e-6` for every selected case |
| C5: Run isolation | BUG-01, BUG-04 | Write to `artifacts_experiments/<run_id>/`; never overwrite a previous run | Re-running creates a new directory; manifest hashes link model, split, metrics, and figures |
| C6: Data audit | BUG-02 | Report complete-row and feature-profile multiplicities before splitting | Counts are present in the run manifest and sum back to 253,680 |

No Stage 1 comparison or Stage 2 explanation experiment may begin until C1-C6 pass on a small pilot subset and on one full seed-42 run.

## 5. Variables

### 5.1 Independent variables

| Variable | Values |
| --- | --- |
| Estimator | Majority dummy; balanced multinomial logistic regression; current weighted Random Forest; Balanced Random Forest; optional weighted XGBoost strong tabular baseline |
| Evaluation protocol | Respondent-row stratified split; exact-row-deduplicated sensitivity; feature-profile-grouped sensitivity |
| Training seed | 42, 123, 456, 789, 1024 |
| Permutation sample size | 1,000; 5,000 |
| Permutation repeats | 3; 10 |
| Permutation parallelism | 1; 2 outer jobs |
| Local explanation stratum | True positive, false positive, and false negative for each target class where available |

### 5.2 Dependent variables

| Category | Measurements |
| --- | --- |
| Primary prediction | Macro-F1; Medium/prediabetes recall |
| Secondary prediction | Balanced accuracy; per-class precision/recall/F1; macro one-vs-rest ROC-AUC; macro average precision; log loss; multiclass Brier score |
| Calibration | Per-class calibration curves and expected calibration error with a pre-specified binning rule |
| Explanation fidelity | SHAP additivity error; missing/invalid case count |
| Explanation stability | Spearman rank correlation; top-five Jaccard similarity; importance mean and standard deviation |
| Efficiency | Fit time; prediction time; permutation time; SHAP time; peak resident/container memory; artifact size |

Accuracy and weighted F1 remain descriptive because they are dominated by the Low class.

### 5.3 Controlled variables

- Dataset file and SHA-256 checksum.
- The exact 21 features, their order, target encoding, numeric conversion, and missing-value policy.
- Locked test indices for the respondent-row confirmatory comparison.
- Identical evaluation functions and thresholds for every estimator.
- No probability-threshold tuning on the test set.
- Docker image digest and explicit package versions.
- CPU allocation and Docker memory limit during efficiency measurements.
- Case-selection algorithm, quantiles, and file naming fixed before reading SHAP values.

## 6. Data split and leakage-sensitivity protocols

### Protocol A - Main respondent-row evaluation

Preserve the current estimand: each survey row is treated as one participant record. Create a locked stratified 20% test set with seed 42. Split the remaining 80% into development folds for model selection. Use the test set only once after model and hyperparameters are frozen.

### Protocol B - Exact-row-deduplicated sensitivity

Keep the first occurrence of each identical 21-feature + target row before splitting. This tests how much frequency replication influences the result. It is a sensitivity analysis, not automatically the preferred dataset, because identical observed rows may represent different respondents.

### Protocol C - Unseen-profile sensitivity

Hash the 21 predictors only and use the hash as the group identifier in `StratifiedGroupKFold`, which attempts to preserve class proportions while keeping each profile in one fold. Conflicting targets for the same feature profile remain in the same group. This protocol asks a stricter and different question: performance on feature profiles absent from training.

Results from A-C must be presented side by side. Protocol C must not silently replace Protocol A because it changes the generalization target.

## 7. Baselines and fair comparison

| Model | Role | Core configuration |
| --- | --- | --- |
| `DummyClassifier(strategy='most_frequent')` | Simple lower bound | No tuning |
| Multinomial logistic regression | Simple interpretable baseline | Standardization inside a pipeline; `class_weight='balanced'`; tune regularization on development folds |
| Current weighted Random Forest | Registered reference | 400 trees; `balanced_subsample`; `min_samples_leaf=2`; otherwise current settings |
| `BalancedRandomForestClassifier` | Primary imbalance-aware candidate | Explicitly set `sampling_strategy='all'`, `replacement=True`, and `bootstrap=False`; do not rely on version-changing defaults |
| Weighted XGBoost | Optional strong tabular baseline | Multiclass objective; class-derived sample weights; tuning budget matched to the tree baselines |

Balanced Random Forest differs from the current weighted forest by drawing balanced class samples for its trees; its parameters must be explicit because defaults have changed across imbalanced-learn releases. XGBoost is a strong tabular comparator, not automatically described as dataset-specific state of the art. A literature search for recent same-dataset methods is required before publication.

Recent BRFSS papers commonly use a binary target, combine prediabetes with diabetes, apply different resampling pipelines, or report accuracy-dominated results. Their published numbers are therefore not fair baselines for this three-class task. They may guide the candidate-model list, but every reported comparison in this project must be rerun on the locked three-class split with the shared evaluation script.

All non-dummy models receive the same development folds and evaluation code. Each model receives a pre-declared tuning budget. Test results are not inspected during tuning.

## 8. Experiment matrix

### E0 - Correctness pilot

Run C1-C6 on a stratified 10,000-row subset, then one full seed-42 run. Expected output: all gates pass, metrics artifacts agree, and an execution-order test detects stale state.

### E1 - Main estimator comparison

Run the four required baselines under Protocol A for five training seeds. Add XGBoost only if its dependency and tuning budget are approved. Select the model using development macro-F1 with Medium recall as a mandatory reported endpoint. Evaluate the frozen candidates on the locked test set.

### E2 - Split sensitivity

Run the current weighted Random Forest and the E1-selected candidate under Protocols A-C. Use identical metric code. Report absolute and relative metric changes and the class distributions in every fold.

### E3 - Local SHAP case cohort

Create confusion strata from locked test predictions: TP, FP, and FN for each class. Within each available stratum, sort cases by the relevant predicted-class probability and select the observations nearest the 10th, 50th, and 90th percentiles. If a stratum contains fewer than three cases, include all cases. Save the selection manifest before calculating SHAP.

For every case, save:

- dataset row number and anonymous case ID;
- recorded class, predicted class, and all three probabilities;
- all 21 predicted-class SHAP values and base value;
- SHAP additivity error;
- one waterfall plot with a case-specific filename.

The report must include both successful and failed predictions. The original patient 104,917 may remain as the Stage 3 continuity case, but it cannot be the sole Stage 2 evidence.

### E4 - Permutation reliability grid

Using the selected Stage 1 model and one locked test sample per seed, run:

| Configuration | Sample size | Repeats | Purpose | Relative scoring workload |
| --- | ---: | ---: | --- | ---: |
| P1 | 1,000 | 3 | Current reference | 1.00x |
| P2 | 1,000 | 10 | Repeat sensitivity | 3.33x |
| P3 | 5,000 | 3 | Sample-size sensitivity | 5.00x |
| P4 | 5,000 | 10 | Reliability reference | 16.67x |

Compute feature-level mean, standard deviation, 95% interval, full-rank Spearman correlation, and top-five Jaccard similarity. Do not interpret negative importance as medical protection.

### E5 - Parallelism and memory profile

Run P1 with `n_jobs=1` and `n_jobs=2` in otherwise identical fresh containers. Repeat each configuration three times after one warm-up. Record wall time and peak container memory. Promote `n_jobs=2` only if H4b passes.

## 9. Ablation and sensitivity design

| Ablation | Variant A | Variant B | Variant C | Question |
| --- | --- | --- | --- | --- |
| Class imbalance treatment | Unweighted RF | Weighted RF | Balanced RF | Does resampling improve Medium recall beyond class weighting? |
| Tree count | 100 | 200 | 400, then 800 only if still improving | Is 400 trees on the performance plateau? |
| Leaf regularization | 1 | 2 | 5 and 10 | Does stronger regularization improve minority generalization? |
| Duplicate/profile protocol | Row split | Deduplicated | Profile-grouped | Are conclusions sensitive to repeated observed patterns? |
| Importance reliability | P1 | P2/P3 | P4 | Are global feature rankings stable? |

Hyperparameter sensitivity is performed only on development data. Use one-at-a-time curves for tree count and leaf size, then confirm only the registered baseline and selected configuration across five seeds. This prevents an unnecessary full factorial grid.

## 10. Statistical analysis

- Report every seed, mean, standard deviation, median, and range.
- For model comparisons on the same locked test observations, use a stratified paired bootstrap with 10,000 resamples to form 95% confidence intervals for macro-F1, balanced accuracy, and Medium recall differences.
- Require the H2b improvement direction in at least four of five seeds.
- Apply Holm correction when testing multiple candidate models against the registered weighted-RF reference.
- Treat a confidence interval crossing zero as inconclusive rather than as evidence of equivalence.
- Report effect sizes in absolute percentage points and relative percentages.
- Pre-register the acceptance thresholds in H1-H4 before running the full experiments.

## 11. Artifact and provenance contract

Every run writes to `artifacts_experiments/<run_id>/` and includes:

```text
run_manifest.json
data_audit.json
split_indices.npz
model_config.json
metrics.json
metrics_by_class.csv
predictions.csv.gz
timings.json
environment.txt
figures/
shap_cases/
```

The manifest must contain the dataset checksum, Git commit if available, notebook hash, Docker image digest, package versions, seed, split protocol, estimator configuration, and hashes of result artifacts. Keep only the promoted model binary and registered reference model; retain predictions and configurations for other runs to avoid storing approximately 405 MB per forest.

## 12. Compute and storage budget

This is a CPU experiment. Let `T_RF` be the measured wall time for one full 400-tree fit in the confirmatory Docker container, and let `P` be the measured time for the current P1 permutation run.

| Work package | Planned cost |
| --- | --- |
| E0 correctness | One 10k-row pilot + one full RF run |
| E1 required baselines | 4 models x 5 seeds = 20 fits; dummy cost is negligible |
| E2 split sensitivity | 2 models x 3 protocols x 5 seeds = 30 fits |
| Confirmatory sensitivity | At most 2 selected configurations x 5 seeds = 10 fits |
| E4 permutation grid | Approximately 26P per seed before parallelism savings |
| E5 resource profile | 6 measured P1 runs + 2 warm-ups |
| E3 local SHAP | Up to 27 cases; reuse one explainer per model |

Upper-bound training cost is approximately `60 x T_RF`, plus Stage 2 analysis. Apply a 1.75 safety factor after measuring `T_RF` and `P` in E0. If one full fit takes 15 minutes, the un-parallelized training upper bound is about 15 hours and the safety-adjusted budget is about 26.25 hours. This is an example, not a measured runtime.

Saving all 60 forests would require roughly 24 GB at the current approximately 405 MB artifact size. The provenance policy instead stores predictions and metadata for all runs and only two model binaries, targeting less than 2 GB excluding optional plots and Docker layers.

## 13. Execution order and stop/go rules

1. **Correctness gate:** implement C1-C6 and run E0. Stop if any artifact mismatch or SHAP additivity failure remains.
2. **Resource pilot:** measure `T_RF`, `P`, Docker memory ceiling, and available disk. Revise only the compute schedule, not hypotheses or endpoints.
3. **Development experiments:** tune baselines and sensitivity settings without accessing the locked test results.
4. **Main Stage 1 experiment:** run E1, freeze configurations, then evaluate the locked test set.
5. **Split sensitivity:** run E2 and decide whether both estimands must be foregrounded.
6. **Stage 2 evidence:** run E3-E5 using the frozen model.
7. **Statistical analysis:** generate tables and confidence intervals from stored predictions.
8. **Report update:** replace report numbers only from a completed run manifest; preserve research-only and non-causal wording.

Promotion rules:

- Do not replace the current model unless H2b passes and no class loses more than 0.05 recall without an explicitly justified trade-off.
- Do not call the global ranking stable unless H4a passes.
- Do not use `n_jobs=2` by default unless H4b passes.
- Do not claim patient-level explanation coverage unless all available pre-specified strata are represented and H3 passes.

## 14. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Identical profiles are mistaken for duplicate respondents | High | Invalid data deletion | Treat deduplication/grouping as sensitivity analyses; state lack of respondent IDs |
| Medium class remains poorly detected | High | Core RQ1 limitation | Report negative result; consider task reformulation or additional data rather than hiding it |
| Test leakage through tuning | Medium | Inflated results | Lock test indices and separate development folds before model comparison |
| Package/version drift corrupts model or SHAP output | High | Incorrect artifacts | Use one Docker image digest and explicit SHAP shape/additivity tests |
| Permutation grid exceeds memory/time | Medium | Incomplete E4 | Run P1/P3 pilot first; retain serial execution; report incomplete cells rather than changing endpoints |
| Case selection becomes cherry-picked | Medium | Weak Stage 2 evidence | Save deterministic confusion-stratum manifest before computing SHAP |
| Model artifacts consume excessive disk | High | Run failure | Save predictions/configs for all runs and only the registered/promoted models |

## 15. Reproducibility checklist

- [ ] Dataset source, license, size, and SHA-256 recorded.
- [ ] Dockerfile, image digest, Python, scikit-learn, SHAP, pandas, NumPy, imbalanced-learn, and optional XGBoost versions recorded.
- [ ] CPU model, logical cores, RAM, Docker CPU allocation, Docker memory limit, and storage type recorded from inside/outside the container where permitted.
- [ ] Seeds fixed to 42, 123, 456, 789, and 1024.
- [ ] Locked test and development indices saved.
- [ ] Every estimator receives the same folds, metric implementation, and tuning budget.
- [ ] All correctness gates covered by automated tests.
- [ ] Failed and incomplete runs retained in the manifest with error messages.
- [ ] Exact commands for notebook execution and result aggregation documented.
- [ ] A fresh Docker run reproduces the promoted metrics within the declared tolerance.

## 16. Method references

- Balanced Random Forest implementation and explicit parameters: [imbalanced-learn documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html).
- Group-aware stratification: [scikit-learn cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html).
- Strong boosted-tree baseline: T. Chen and C. Guestrin, [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785), KDD 2016.
