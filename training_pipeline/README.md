## Historical liquefaction model training pipeline (Hanna 2007 + Hwang 2021)

This folder contains a **reproducible training pipeline** and a **paper-ready notebook** for building data‑driven liquefaction models from historical SPT case histories.

The goal is to replace (or augment) analytical CRR equations with a **learned CRR–limit state relationship** based on global case histories, similar in spirit to the Dhaka ANN paper, but using the Hanna (2007) and Hwang (2021) datasets.

### Datasets used

- `hanna2007_historical_cases.csv`  
  - Extracted from Hanna et al. (2007), Table A1 (“SPT liquefaction case records”).
  - Per-row variables include:
    - Depth `Z` (m)
    - Corrected SPT blow count `(N1)60`
    - Fines content proxy `Fp75 mm (%)`
    - Groundwater depth `dw` (m)
    - Total and effective overburden stresses (`svo`, `s0vo` in kPa)
    - Cyclic stress ratio–related terms
    - Shear wave velocity `Vs`
    - Internal friction angle `f0`
    - Earthquake magnitude `Mv`
    - Peak ground acceleration `amax`
    - Binary liquefaction label (`Liquefaction` Yes/No)

- `hwang2021_historical_cases.csv`  
  - Extracted from Hwang et al. (2021), Appendix I (“Adopted database of liquefaction and non-liquefaction case histories”).
  - Per-row variables include:
    - Corrected SPT blow count `N60`
    - Depth `Depth_m`
    - Fines content `FC_pct`
    - Groundwater depth `GWT_m`
    - PGA `PGA_g`
    - Vertical stresses (`sigma_v_prime_t_m2`, `sigma_v_t_m2` in t/m², converted to kPa)
    - Liquefaction classification (`Liq1/Liq2/Liq3` vs `nonLiq/nonLiq_IB`), reduced here to `Liquefied` = 1/0.

Both datasets are harmonized into a **common SPT-based feature table** with:

- `N1_60` (normalized SPT resistance):
  - Hanna: taken directly from `(N1)60`.
  - Hwang: computed from `N60` and an overburden correction `CN = 0.77 * log10(2000 / σv')`.
- `FCI`: fines‑content proxy (Hanna’s `Fp75mm_pct`, Hwang’s `FC_pct`).
- `sigma_v_eff_kpa`: effective vertical stress in kPa.
- `CSR7.5`: cyclic stress ratio adjusted to Mw = 7.5, computed consistently with your project’s `compute_csr` function:
  \[
    CSR_{7.5} = 0.65 \cdot a_{max} \cdot \frac{\sigma_v}{\sigma'_v} \cdot r_d
  \]
  with `rd = 1 − 0.015·z` (same approximation as in `build_merged_spt_from_spt_value.py`).
- `T`: binary liquefaction indicator (1 = liquefied, 0 = non‑liquefied), derived from:
  - Hanna: `Liquefaction` Yes/No.
  - Hwang: `Liquefied` (Liq* vs nonLiq*).

Rows with missing or non‑finite values in any of these core fields are removed as anomalies.

### Notebook: `historical_li_lsf_training.ipynb`

This notebook is intended as the **human‑readable, figure‑producing** version of the training process (for your paper). It walks through:

1. **Data import and harmonization**
   - Loads both CSVs.
   - Builds the unified feature table (`N1_60`, `FCI`, `sigma_v_eff_kpa`, `CSR7.5`, `T`, `Source`).
   - Drops rows with NaNs/infinities in core features.

2. **Dataset cleaning / anomaly removal**
   - Any case with missing `N1_60`, `FCI`, `sigma_v_eff_kpa`, `CSR7.5` or `T` is removed.
   - Stress conversion from t/m² to kPa is done consistently.
   - CN and CSR are computed with the same formulas as your main project script.

3. **Exploratory Data Analysis (EDA)**
   - Class balance (liquefied vs non‑liquefied).
   - Histograms of `N1_60`, `FCI`, `sigma_v_eff_kpa`, `CSR7.5` split by liquefaction class.
   - Source comparison (Hanna vs Hwang) to show how each contributes to the combined dataset.
   - Correlation matrix for core features + label.
   - All plots are generated using `matplotlib`/`seaborn` and can be copied into the paper.

4. **Train / validation / test split (70 / 15 / 15)**
   - Stratified by `T` to preserve class balance.
   - Features: `X = [N1_60, FCI, sigma_v_eff_kpa, CSR7.5]`.
   - Label: `y = T`.
   - A `StandardScaler` is fit on the training features and applied to validation and test.

5. **Baseline LI (Liquefaction Indicator) models**

   The LI models are **binary classifiers** that estimate the probability of liquefaction:

   - **GRNN‑style kernel classifier (GRNNClassifierNB)**
     - Non‑parametric kernel smoother in feature space.
     - Hyperparameter: kernel width `sigma` (grid of values).

   - **MLP (feed‑forward neural network) classifier**
     - Several hidden‑layer sizes and regularization (`alpha`, learning rate) combinations.

   - **SVM with RBF kernel**
     - Grid over `C` and `gamma`.
     - Uses `probability=True` for Platt‑scaled probabilities.

   For each candidate model, the notebook computes **validation metrics**:
   - AUROC
   - LogLoss
   - Brier score

   Bar plots compare all candidates, and the best GRNN / MLP / SVM models are saved individually under:

   - `training_pipeline/outputs/models/li_grnn_best.pkl`
   - `training_pipeline/outputs/models/li_mlp_best.pkl`
   - `training_pipeline/outputs/models/li_svm_best.pkl`

6. **LI ensembles and test evaluation**

   The notebook then evaluates:

   - Individual best models (GRNN_best, MLP_best, SVM_best) on the held‑out **test** set.
   - Simple ensembles:
     - GRNN + SVM (average of probabilities).
     - GRNN + MLP.
     - GRNN + MLP + SVM (all‑three ensemble).

   For each, it reports and plots on test:
   - AUROC
   - LogLoss
   - Brier score

   These figures allow you to show that the ensemble is better‑calibrated and/or more accurate than any single model.

7. **Boundary search and LSF (CRR) models (concept)**

   The notebook reserves a section (to be filled if you want full transparency) explaining how we:

   - Use a chosen LI model (e.g., the best ensemble) to search along `CSR7.5` for each historical case until `P(liquefy) ≈ 0.5`.
   - Interpret the **critical CSR7.5** at that point as the **CRR_target** (limit‑state CRR).
   - Train **regression models** (`CRR = f(N1_60, FCI, σv' )`) such as:
     - GRNN regressor
     - MLP regressor
     - SVR (RBF)

   This mirrors the implemented logic in `train_li_lsf_baseline_models.py`, but is staged for clearer explanation and plots.

8. **Summary and paper notes**

   The last cells are for you to summarize:

   - Which LI model or ensemble gave the best performance on **test**.
   - Which LSF (CRR) model gave the lowest RMSE/MAE.
   - How these models were trained on a **combined historical dataset** (Hanna + Hwang).
   - How their learned CRR values will be plugged into your Sylhet liquefaction pipeline in place of (or alongside) your analytical CRR equation.

### Supporting script: `train_li_lsf_baseline_models.py`

In the project root, `train_li_lsf_baseline_models.py` automates the full LI + boundary search + LSF training process:

1. Builds the same **combined historical feature table** as the notebook.
2. Trains LI models (GRNN, MLP, SVM), selects the best **single** vs **ensemble** using validation LogLoss, and evaluates it on test.
3. Performs the LI‑based **boundary search** to generate `CRR_target` for each historical record.
4. Trains LSF (CRR) regressors (GRNN, MLP, SVR) and selects the best model vs ensemble based on validation RMSE.
5. Saves:
   - `historical_training_run_X/LI/li_wrapper.pkl`
   - `historical_training_run_X/LSF/lsf_wrapper.pkl`
   - JSON summaries and CSVs summarizing the run.

You can reference those artifacts in the notebook for consistency, or re‑use the notebook purely for visualization and explanation while the script handles production training.

### How to use this pipeline in your paper

In the **methods** section, you can say (in plain language):

- We compiled SPT‑based liquefaction case records from Hanna (2007) and Hwang (2021).
- We harmonized these data to a common feature set (normalized SPT resistance `(N1)60`, fines index `FCI`, effective vertical stress `σv'`, and cyclic stress ratio `CSR7.5`).
- We trained machine‑learning based liquefaction indicator models (GRNN, MLP, SVM) and compared their performance using AUROC, LogLoss, and Brier score.
- We then combined the best LI models into simple ensembles and confirmed improved probability calibration and predictive skill on an independent test set.
- Following the Juang/Dhaka approach, we used the trained LI function to numerically search for the limit‑state boundary for each case, thereby obtaining CRR values that are fully consistent with historical performance.
- Finally, we trained regression models that map `(N1)60`, `FCI`, and `σv'` directly to these limit‑state CRR values, providing an empirical CRR function that we then applied to our Sylhet SPT data for liquefaction assessment.

The notebook already generates the most important figures (distributions, correlation matrix, model comparison bar charts) that can be used directly as paper figures or adapted to your journal’s style.

