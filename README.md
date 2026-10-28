# Linear Null Model — Reproduction

Reproduction of the tabular-only **Linear Null Model** baseline from:

> Chattopadhyay, T., et al. (2026). *Deep learning to predict future cognitive decline: a comparison of volumetric fusion and AutoML fusion.* Frontiers in Neuroimaging, 5:1726037. https://doi.org/10.3389/fnimg.2026.1726037

The model predicts the **sum-of-boxes Clinical Dementia Rating (sobCDR) ~2 years after baseline** from four baseline variables (age, sex, BMI, baseline sobCDR) using ordinary least squares, pooled across three cohorts (ADNI, NACC, OASIS-3), and evaluated with stratified 5-fold cross-validation.

> ⚠️ **No data is included in this repository.** ADNI, NACC and OASIS-3 are available only to approved investigators under their respective Data Use Agreements, which prohibit redistribution. `.gitignore` excludes all data and output files.

---

## Pipeline

```
Step 1  sobCDR table        (pid, t, sobcdr)       one row per CDR visit
Step 2  MRI table           (pid, t)               QC-passed FreeSurfer sessions → first MRI per subject
Step 3  baseline sobCDR     nearest CDR within ±90 days of the baseline MRI
Step 4  follow-up sobCDR    CDR 1.75–2.25 years after baseline (closest to 2.0 years)
Step 5  covariates          age at MRI, sex, BMI; ADNI restricted to ADNI1 / ADNIGO / ADNI2
Step 6  stratified 5-fold   strata = sex × age bin × ΔsobCDR bin
Step 7  model               StandardScaler + OneHotEncoder → LinearRegression (Pipeline)
Step 8  evaluation          per-fold MAE, R², RMSE, MedAE, MASE → Table 2 format
```

All time variables are converted to **days**. ADNI and NACC dates are converted to days from a fixed origin; OASIS-3 uses `days_to_visit` (days from entry). Only within-subject time differences are used, so the choice of origin does not affect results.

---

## Inclusion criteria

### Stated in the paper

| Criterion | Implementation |
|---|---|
| 3D T1-weighted MRI at baseline | QC-passed FreeSurfer session (ADSP-PHC for ADNI and NACC; OASIS-3 FreeSurfer output) |
| Baseline sobCDR within 90 days of the imaging session | `merge_asof(direction="nearest", tolerance=90)` |
| Follow-up sobCDR 1.75–2.25 years after the baseline visit | closest visit to 2.0 years within the window |
| Baseline age, sex and BMI available | complete-case |
| ADNI phases 1, GO and 2 (Fig. 2) | filtered on `ORIGPROT` |

### Not specified in the paper — decisions made here

| Decision | Choice in this code |
|---|---|
| Baseline when a subject has several MRI sessions | **first MRI only**; if it does not meet all criteria the subject is excluded (no fallback to later sessions) |
| Start of the follow-up window | baseline **sobCDR** visit date (`fu_from="cdr"`; `"mri"` is also supported) |
| BMI time window | most recent measurement **≤ 1 year before** the MRI; if none, the closest measurement **≤ 30 days after** the MRI |
| Height | per-subject median (height is often recorded only once) |
| Age | nearest age record within 3 years, shifted to the MRI date |
| Stratification bins | age `<60 / 60–70 / 70–80 / ≥80`; ΔsobCDR `<0 / 0 / (0,1] / (1,3] / >3`; strata with < 5 subjects merged |
| Fold assignment | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| MASE denominator | MAE of predicting the training-fold mean |

---

## Data sources

| | sobCDR | MRI | Age | Sex | BMI |
|---|---|---|---|---|---|
| **ADNI** | `CDR.parquet` (`CDRSB`) | `ADSP_PHC_T1_FS.parquet` | `PHC_Age_T1` | `ADSP_PHC_DEMODX.parquet` | `VITALS.parquet` (weight, height, units) |
| **NACC** | `investigator_nacc75.parquet` (`CDRSUM`) | `NACC_ADSP_PHC_T1_FS_2026.02.18.csv` | `NACCAGE` | `NACCSEX` | `NACCBMI` |
| **OASIS-3** | `OASIS3_UDSb4_cdr.csv` (`CDRSUM`) | `OASIS3_Freesurfer_output.csv` | `age at visit` | `OASIS3_demographics.csv` | `OASIS3_UDSb1_physical_eval.csv` |

Missing codes (e.g. `-1`, `-4`, `88`, `888.8`, `9999`) are removed by range checks (sobCDR 0–18, BMI 12–60, plausible height/weight).

### Expected folder layout

The script reads data from `~/Desktop`:

```
~/Desktop/
├── ADNI/ADNIMERGE2/ADNI_parquet/
│   ├── CDR.parquet
│   ├── ADSP_PHC_T1_FS.parquet
│   ├── ADSP_PHC_DEMODX.parquet
│   └── VITALS.parquet
├── NACC/
│   ├── investigator_nacc75.parquet
│   └── adsp_phc/extracted/Imaging_T1/NACC_ADSP_PHC_T1_FS_2026.02.18.csv
└── OASIS3/OAS3_data_files/OASIS3_data_files/scans/
    ├── UDSb4*/resources/csv/files/OASIS3_UDSb4_cdr.csv
    ├── UDSb1*/resources/csv/files/OASIS3_UDSb1_physical_eval.csv
    ├── demo-demographics*/resources/csv/files/OASIS3_demographics.csv
    └── FS-Freesurfer*/resources/csv/files/OASIS3_Freesurfer_output.csv
```

To use a different location, change `ROOT` at the top of `my_linear_null.py`.

---

## Usage

```bash
pip install numpy pandas pyarrow scikit-learn
python my_linear_null.py
```

Outputs (written to `output/`, git-ignored):

| File | Content |
|---|---|
| `my_linear_null_dataset.csv` | analysis dataset, one row per subject, with the `fold` column — reuse it so every future model is evaluated on identical folds |
| `my_table2_linear_null.csv` | per-fold metrics in the layout of the paper's Table 2 |

---

## Results

Paper (Table 2, Linear Null Model, N = 2,319):

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg |
|---|---|---|---|---|---|---|
| MAE | 0.980 | 1.027 | 1.130 | 0.998 | 1.012 | 1.029* |
| R² | 0.786 | 0.804 | 0.757 | 0.740 | 0.744 | 0.767 |

\* Recomputed from the fold values; the paper's table reports 1.096.

This reproduction:

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg |
|---|---|---|---|---|---|---|
| MAE | 0.974 | 1.001 | 0.872 | 0.945 | 0.828 | 0.924 |
| R² | 0.730 | 0.649 | 0.764 | 0.780 | 0.756 | 0.736 |
| RMSE | 1.737 | 1.873 | 1.490 | 1.555 | 1.380 | 1.607 |
| MedAE | 0.346 | 0.355 | 0.381 | 0.373 | 0.352 | 0.361 |
| MASE | 0.434 | 0.444 | 0.385 | 0.426 | 0.361 | 0.410 |

N = 2,664 (ADNI 974, NACC 1,396, OASIS-3 294)

Difference from the paper: MAE −0.105, R² −0.031. Lower MAE together with lower R² indicates a sample with lower outcome variance (milder disease on average) rather than a better model.

---

## Known differences from the paper

- **NACC sample.** The reproduced NACC subset is larger, younger and majority female, whereas the paper's is majority male (413 F / 529 M). The paper's MRI source and data freeze for NACC are not reported; this reproduction uses the ADSP-PHC FreeSurfer release.
- **Data versions.** Data were downloaded in 2026; the paper used earlier releases.
- **Unspecified choices.** See *Not specified in the paper* above; fold assignments cannot match the paper's.
- **Inconsistency in the paper.** The text lists the tabular inputs as age, sex, BMI and baseline sobCDR, while Fig. 4 also lists diagnosis (Dx) for the hybrid CNN. This reproduction follows the text.

---

## Repository contents

```
my_linear_null.py   full pipeline (data → dataset → 5-fold CV → Table 2)
README.md
.gitignore          excludes data, outputs and environment files
```
