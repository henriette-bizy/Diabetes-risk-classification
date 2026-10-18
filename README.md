# Predicting Diabetes and Prediabetes from Health Survey Answers

**Author:** Henriette Biziyaremye, BSc Software Engineering, African Leadership University

**Report:** _add link_ · **Google Drive (raw data + saved models):** _add link_ · **Video:** _add link_

## What this project is about

Many people living with diabetes don't know they have it, and prediabetes, the stage where the disease can still be prevented, is even easier to miss. Blood tests are the reliable way to find out, but they cost money and need a clinic. This project asks whether answers to a short health survey (blood pressure, BMI, physical activity, general health, age and so on) can flag people who should go for a proper test.

I compare traditional machine learning (scikit-learn) with deep learning (TensorFlow/Keras) on a three-class problem: no diabetes, prediabetes and diabetes.

## Dataset

The data is the **CDC Diabetes Health Indicators** dataset, built from the 2015 Behavioral Risk Factor Surveillance System (BRFSS). It has 253,680 survey responses, 21 features and the target `Diabetes_012`.

- Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

The raw data is **not** stored in this repository. The notebook downloads it automatically with `kagglehub`, or you can download the zipped copy from the Google Drive folder and place `diabetes_012_health_indicators_BRFSS2015.csv` in a `data/` folder.

## Repository structure

```
diabetes-risk-classification/
├── diabetes_classification.ipynb   # the full project, Parts 1-4
├── src/
│   ├── preprocessing.py            # data split, BMI capping, feature engineering, scaling
│   ├── experiments.py              # metrics, experiment tracker (CSV + W&B offline), plots
│   └── deep_learning.py            # tf.data pipeline, Keras models, callbacks, training runner
├── requirements.txt
└── README.md
```

Running the notebook creates three more folders, which are not committed: `figures/` (every plot used in the report), `artifacts/` (processed data, saved models, experiment and test result tables) and `logs/` (W&B offline runs and TensorBoard logs).

## How to run it

**Locally (VS Code or Jupyter):**

```bash
git clone <this repo>
cd diabetes-risk-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook diabetes_classification.ipynb
```

**On Google Colab:** upload this folder to Google Drive, open the notebook with Colab, and run this as the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/diabetes-risk-classification
```

Then use **Runtime → Run all**. A GPU is not needed. A full run takes roughly an hour on a standard Colab runtime, and most of that time goes into the hyperparameter searches in Part 2.

Every source of randomness uses seed 42 (Python, NumPy, scikit-learn, TensorFlow with op determinism enabled), so the results in the report can be reproduced.

## What the notebook does

1. **Data and preprocessing.** Exploration of the data, a stratified 70/15/15 train/validation/test split made before any fitting, BMI capping at training-set percentiles, nine engineered features based on medical definitions, scaling, mutual information ranking and class weights.
2. **Traditional ML (ML-00 to ML-11).** A majority-class baseline, Logistic Regression, Random Forest and Histogram Gradient Boosting, with grid and randomised searches, class weights vs SMOTE, engineered vs original features, and top-15 feature selection.
3. **Deep learning (DL-01 to DL-09).** Sequential MLPs and a Functional-API Wide & Deep network trained through a `tf.data` pipeline, testing class weighting, capacity, dropout, batch normalisation, L2, learning-rate schedules, a too-high learning rate and Adam vs SGD.
4. **Final evaluation.** The best model from each approach is evaluated once on the untouched test set, with bootstrap confidence intervals, ROC and precision-recall curves, error analysis, subgroup checks and a screening-threshold analysis.

## Experiment tracking

Every run goes through one tracker, so all experiments are scored the same way:

- `artifacts/experiment_log.csv` has one row per experiment with its settings and validation metrics.
- `logs/wandb/` holds a Weights & Biases run per experiment (offline mode). To view them online: `wandb login`, then `wandb sync logs/wandb/offline-run-*`.
- `logs/tensorboard/` has the per-epoch curves for the deep learning runs: `tensorboard --logdir logs/tensorboard`.

## Saved models

The best models are saved in `artifacts/` and uploaded to the Google Drive folder:

- `best_traditional_model_<ID>.joblib` contains the fitted preprocessing and the model together, so it accepts raw survey columns directly. Loading it needs the `src/` folder on the Python path.
- `best_deep_learning_model_<ID>.keras` (and `.h5`) holds the network. Its inputs are the output of `artifacts/preprocessor.joblib`.

```python
import joblib, pandas as pd
model = joblib.load("artifacts/best_traditional_model_<ID>.joblib")
probabilities = model.predict_proba(raw_rows)   # columns: no diabetes, prediabetes, diabetes
```

## Main results

_Add after the final run: the test macro-F1 and ROC-AUC of the best traditional and best deep learning model, and two or three sentences on what they mean._

## Limitations

The survey answers are self-reported, prediabetes is very rare (under 2% of rows) and probably under-diagnosed, so its labels are noisy, and the data comes from the United States in 2015. A model like this would need to be retrained and validated on local data before it could be used anywhere else, including in Rwanda.
