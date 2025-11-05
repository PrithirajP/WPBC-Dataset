# WPBC – Leakage-safe Stacking & Calibration for Breast Cancer Prognosis

This repository contains three, fully reproducible pipelines for recurrence prediction on the Wisconsin Prognostic Breast Cancer (WPBC) dataset. All variants follow the same **leakage-safe design**:

* **Train-only transforms:** median imputation → IQR outlier handling (winsorize/clip) → standardization → feature selection via RF(SFM)
* **Imbalance handling:** Applied only on train slices using BorderlineSMOTE.
* **Calibration:** Performed on a held-out calibration slice (no resampling in calibration/test).
* **Thresholds:** Fixed from Out-of-Fold (OOF) predictions (Variants 2–3) to avoid test-set tuning.
* **Stacking:** RF + SVC + kNN → Logistic Regression meta-learner (using calibrated base probabilities as meta-features).

Figures (confusion matrices, ROC curves, SHAP) are saved automatically under `figures/` by all three scripts.

---

## 📂 Repository Structure

```bash
.
├── WPBC variant 1.ipynb    # Calibrated bases, base thresholds fixed at 0.5; meta threshold from OOF
├── WPBC variant 2.ipynb    # Calibrated bases, thresholds for bases + meta from OOF (accuracy objective)
├── WPBC variant 3.ipynb    # As Variant 2, but PSO tunes base hyperparams for ROC-AUC
├── prognostic.csv       # WPBC dataset (place here)
````

-----

## ⚙️ Installation

1.  Create and activate a virtual environment:

    ```bash
    python -m venv .venv

    # On macOS/Linux
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\activate
    ```

2.  Install dependencies:

    ```bash
    pip install -U pip
    pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn shap pyswarm
    ```

    > **Note:** `shap` may require `numba` depending on your platform/GPU stack (`pip install numba`).

-----

## 💾 Dataset

1.  Obtain the **prognostic.csv** file (WPBC dataset).
2.  Place it in the repository's root directory.

All scripts automatically perform the following preprocessing steps:

  * Replace `?` with `NaN`.
  * Coerce predictors to numeric (non-numeric values become `NaN`).
  * Drop potential leakage columns (`Time` and `ID` if present).
  * Encode `Outcome`: 1 = Recurrence, 0 = Non-recurrence.
  * Handle imputation *inside* CV folds to prevent data leakage.

-----

## 🚀 How to Run

Run the desired variant from your terminal:

```bash
# Variant 1: 0.5 thresholds for base models (meta thr from OOF)
python "WPBC variant 1.ipynb"

# Variant 2: thresholds for bases + meta from OOF (optimize ACC)
python "WPBC variant 2.ipynb"

# Variant 3: like Variant 2, but PSO objective = ROC-AUC
python "WPBC variant 3.ipynb"
```

### Outputs

  * **Console:** Metrics are printed (Threshold, Accuracy, F1, ROC-AUC, Precision, Recall).
  * **`figures/` directory:** All plots are saved as 300 dpi PNG files.

-----

## 🔬 Why Three Variants?

The three scripts demonstrate different approaches to model optimization and thresholding:

  * **`WPBC variant 1.ipynb`**

      * **What it is:** A strong, simple baseline.
      * **How it works:** Uses calibrated base models. Base model thresholds are fixed at 0.5. The *meta-learner's* threshold is selected from OOF predictions.
      * **Best for:** A conservative baseline with high accuracy and stable behavior.

  * **`WPBC variant 2.ipynb`**

      * **What it is:** A simulated "real-world" policy.
      * **How it works:** Uses OOF-selected thresholds for *all* models (RF/SVC/kNN *and* the meta-learner). Hyperparameters (via PSO) and thresholds are optimized for **accuracy**.
      * **Best for:** Demonstrating how to fix a test-time policy using only training data.

  * **`WPBC variant 3.ipynb`**

      * **What it is:** An AUC-focused alternative to Variant 2.
      * **How it works:** Same OOF-thresholding logic as Variant 2, but the PSO hyperparameter tuning objective is set to **ROC-AUC**.
      * **Best for:** Improving model ranking quality, which is often crucial on imbalanced datasets.

-----

## 📊 Final Results

These results are printed by the scripts upon completion.

| Variant | Model | Thr | Acc | F1 | ROC-AUC | Prec | Rec |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Meta-LR | 0.3000 | 0.7600 | 0.3333 | 0.5461 | 0.5000 | 0.2500 |
| 1 | RF | 0.5000 | 0.7600 | 0.3333 | 0.5844 | 0.5000 | 0.2500 |
| 1 | SVC | 0.5000 | 0.7600 | 0.2500 | 0.5592 | 0.5000 | 0.1667 |
| 1 | kNN | 0.5000 | 0.7800 | 0.2667 | 0.5504 | 0.6667 | 0.1667 |
| 2 | Meta-LR | 0.4000 | 0.7400 | 0.1333 | 0.5921 | 0.3333 | 0.0833 |
| 2 | RF | 0.4500 | 0.7600 | 0.3333 | 0.6107 | 0.5000 | 0.2500 |
| 2 | SVC | 0.6500 | 0.7800 | 0.2667 | 0.5592 | 0.6667 | 0.1667 |
| 2 | kNN | 0.7500 | 0.7600 | 0.0000 | 0.5033 | 0.0000 | 0.0000 |
| 3 | Meta-LR | 0.3000 | 0.7400 | 0.2353 | 0.5746 | 0.4000 | 0.1667 |
| 3 | RF | 0.4000 | 0.7200 | 0.2222 | 0.5384 | 0.3333 | 0.1667 |
| 3 | SVC | 0.3000 | 0.7600 | 0.2500 | 0.6031 | 0.5000 | 0.1667 |
| 3 | kNN | 0.3500 | 0.7600 | 0.3333 | 0.5680 | 0.5000 | 0.2500 |

-----

## 🔧 Configuration & Toggles

Inside each script, you can adjust the following global variables:

  * `OUTLIER_MODE`: `"winsorize"` (default) or `"drop"`. Defines IQR clipping strategy. (`drop` is never applied to validation/test data).
  * `CALIB_METHOD`: `"isotonic"` (default) or `"sigmoid"`. Sets the probability calibration method.
  * `THRESH_METRIC`: `"accuracy"` (default) or `"f1_macro"`. The metric used to select OOF thresholds.

-----

## 📈 What Gets Plotted

All figures are automatically saved to the `figures/` directory.

  * **Confusion Matrices:** For RF, SVC, kNN, and the Meta-LR (e.g., `cm_rf.png`).
  * **ROC Curves:** Compares all base models and the final Meta-LR (e.g., `roc_test.png`).
  * **SHAP Interpretability:** Beeswarm and bar plots for the Random Forest model (e.g., `shap_beeswarm_...png`, `shap_bar_...png`).

-----

## 🛡️ Reproducibility & Leakage-Safe Design

  * **Fixed Seeds:** `random_state` is fixed for `train_test_split`, CV folds, and all stochastic learners.
  * **No Test-Set Peeking:** No test-set information is *ever* used to fit transforms (imputation, scaling), tune hyperparameters, calibrate models, or select decision thresholds.
  * **Leakage-Safe Transforms:** Imputation, IQR clipping, scaling, and feature selection are *always* fitted *only* on the training slice for a given fold.
  * **Safe Resampling:** BorderlineSMOTE is only applied to the training slice, never to validation, calibration, or test sets.
  * **Pre-Committed Thresholds:** In Variants 2 and 3, thresholds are selected based *only* on OOF predictions from the training set.

-----

## ⚠️ Troubleshooting

  * **No positives predicted (F1/Recall = 0):** This can happen on small, imbalanced sets, especially if OOF threshold selection (e.g., for kNN) results in a very high threshold.
      * **Solution:** Consider changing `THRESH_METRIC` to `"f1_macro"` or implementing a recall-floor policy for deployment.
  * **SHAP errors:** Ensure `shap` and `numba` are correctly installed. SHAP is computed for the uncalhibited Random Forest (using the tree explainer).

-----

## 📜 Citation

If you use this code or these results in your work, please cite our manuscript, the original WPBC dataset, and the methods we rely on (SMOTE/Borderline-SMOTE, stacking, calibration, SHAP).

-----
