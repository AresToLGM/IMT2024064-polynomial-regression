# Polynomial Regression Assignment

**Name:** Shaurya Singh  
**Roll number:** IMT2024064

This project predicts the target `y` for two personalized datasets using
polynomial regression. Polynomial terms are standardized and fitted with
Lasso. Every term's total degree is at most the selected degree.

## Final models

- **Var1:** all six inputs, degree **5**, Lasso alpha **0.01**.
- **Var2:** all three inputs, degree **10**, Lasso alpha **0.001**.

The final prediction models train on all **1,000 labeled rows** per problem.
They then predict the **1,000 rows in each corresponding test CSV**.

## Files

- `train_predict.py`: data checks, polynomial pipelines, full-data training,
  and prediction CSV generation. Final settings are in `FINAL_MODELS`.
- `model_selection.py`: deterministic development/holdout splits, staged
  feature/degree/alpha searches, repeated CV, and additional checks.
- `requirements.txt`: pinned Python dependencies.
- `data/`: the four original training and test CSVs.
- `predictions/`: the two completed test CSVs, with original input columns
  and row order preserved, followed by predicted `y`.
- `report/IMT2024064_report.pdf`: the four-page report, including the two
  graphs and evaluation limitations.

Experiment logs and temporary files are generated locally and ignored by Git.

## Setup and final prediction

Use Python **3.12**. Create an environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_predict.py
```

On Windows, activate with `.venv\Scripts\activate` instead.

To run only one problem or choose an output directory:

```sh
python train_predict.py --problem 1
python train_predict.py --problem 2 --output-dir regenerated_predictions
```

The code checks finite values, row counts, solver convergence, and preservation
of the source test inputs. Duplicate inputs receive consistent predictions.
No predictions are clipped or manually adjusted.

## Reproducing model selection

The original split has **800 development rows and 200 holdout rows** per
problem, using seed **42** and endpoint-count strata. Var2 keeps repeated
coordinates together in both the holdout split and grouped CV folds.
All feature expansion and scaling are fitted inside each training fold.

The following staged searches reproduce the core search spaces. Some stages
fit thousands of models and take substantially longer than final inference.

```sh
python model_selection.py --problem 1 --stage initial
python model_selection.py --problem 1 --stage refine
python model_selection.py --problem 1 --stage repeat

python model_selection.py --problem 2 --stage initial
python model_selection.py --problem 2 --stage refine
python model_selection.py --problem 2 --stage extend
python model_selection.py --problem 2 --stage local
python model_selection.py --problem 2 --stage repeat
```

Var1's initial Ridge search covers all 63 nonempty input subsets. Degrees
1-10 are screened for smaller subsets and 1-5 for larger subsets; refinement
tests all six inputs through degree 10 and sparse models at degrees 3-6.
Var2's initial search checks all seven subsets and Ridge degrees 1-20,
followed by sparse refinement, higher-degree/basis checks, and local tuning.

The `repeat` stage uses ten shuffled five-fold repeats with seeds 100-109.
It includes the original fixed finalists and the degree-10 var2 follow-up.
Var1 uses KFold and var2 uses GroupKFold by the exact complete coordinate.
MSE is pooled across every development row within a repeat, then averaged
across repeats. The unlabeled test inputs are used only for descriptive
endpoint-mix proxy scores in this stage.

Optional follow-up checks:

```sh
python model_selection.py --problem 1 --stage weights
python model_selection.py --problem 1 --stage alpha-check
python model_selection.py --problem 1 --stage holdout
python model_selection.py --problem 2 --stage holdout
```

The weighting check uses strengths 0, 0.25, 0.5, 0.75 and 1 with seeds 200-209.
The alpha check screens nine values on seeds 600-609, then confirms the best
proxy candidate against alpha 0.01 on seeds 700-719. It retains the practical
comparison criterion of at least 1% proxy improvement and 16 of 20 wins.
The `local` stage for var1 is an optional seed-42 grid; its repeated alpha
confirmation is reproduced by `alpha-check`.

Search logs are saved under `analysis_results/`. Searches report candidates;
they do not automatically replace the frozen submission models. `--jobs 1`
can reduce memory use. `--repeats 1` is a faster diagnostic for the repeated
comparison, rather than the ten-repeat result reported below.

## Recorded validation results

- **Var1:** mean repeated-CV MSE **0.33705**; diagnostic holdout MSE
  **0.31909**, holdout R squared **0.96959**.
- **Var2:** mean repeated-CV MSE **0.35379**; diagnostic holdout MSE
  **0.23984**, holdout R squared **0.99496**.

These holdout results belong to the 800-row fits. Submission predictions
come from separate 1,000-row fits. The holdouts were reused for three model
comparisons in var1 and two in var2, so their scores are diagnostic rather
than fresh independent test estimates. CV results were also used for selection
and repeats share observations. Hidden-test MSE and R squared are unknown.

The report's var1 graph shows seed-42 five-fold Lasso scores for degrees 3-6
at alpha 0.01. Its var2 graph shows repeated-CV Lasso finalists of degrees
10, 12 and 13; the degree-13 alpha is 0.0013, while the others use 0.001.
Those graphs describe the compared configurations, not a guaranteed global
minimum over all possible polynomials.

Var1's test distribution has more endpoint-heavy inputs, including 16
six-endpoint rows absent from training. Var2 has 14 three-endpoint test rows
and 16 such training rows, but none in its holdout. These coverage differences
are discussed in the report.
