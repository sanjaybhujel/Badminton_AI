# Explainable ML for Badminton Match Prediction

This project trains machine learning models to predict men's singles BWF World Tour match outcomes and generates model interpretation outputs using SHAP and LIME.

Author: Sanjay Bhujel  
Dataset: BWF World Tour dataset from Kaggle, `sanderp/badminton-bwf-world-tour`

## What the Pipeline Does

- Loads the men's singles match dataset from `ms.csv`
- Removes retirements and walkovers
- Engineers match-performance and context features
- Trains Logistic Regression, Random Forest, gradient boosting, and a Stacking Ensemble
- Evaluates models with accuracy, precision, recall, F1, ROC-AUC, calibration, cross-validation, ROC curves, confusion matrices, and learning curves
- Saves the best model and selected feature list
- Creates SHAP and LIME explainability outputs in the pipeline and interactive app

## Repository Structure

```text
.
├── badminton_final_pipeline.py
├── streamlit_app.py
├── ms.csv
├── requirements.txt
├── README.md
└── outputs/
```

`outputs/` is generated when the pipeline runs and is ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place `ms.csv` in the project root, or pass a custom path:

```bash
python badminton_final_pipeline.py --data /path/to/ms.csv
```

The local dataset used during development has 3,761 rows and 38 columns.

## Run

Full run:

```bash
python badminton_final_pipeline.py
```

Fast smoke test without SHAP, LIME, or hyperparameter tuning:

```bash
python badminton_final_pipeline.py --skip-explainability --skip-tuning
```

Write outputs somewhere else:

```bash
python badminton_final_pipeline.py --output-dir results
```

Use more CPU cores for cross-validation and tuning:

```bash
python badminton_final_pipeline.py --n-jobs 4
```

The default is `--n-jobs 1`, which is the most portable setting.

On macOS, XGBoost may require the OpenMP runtime. If `libomp.dylib` is not installed, the script automatically falls back to scikit-learn Gradient Boosting. To enable XGBoost locally, install OpenMP first, for example with Homebrew:

```bash
brew install libomp
```

## Interactive App

Run the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```

The app includes:

- sidebar navigation similar to a full web system
- manual prediction form for match-performance and context values
- historical-match prediction mode
- result interpretation with confidence level
- analytics dashboard with dataset and model-performance summaries
- Random Forest feature importance
- SHAP global feature importance
- SHAP local explanation and waterfall plot for the selected match or manual case
- LIME local explanation for the selected match or manual case
- dataset explorer and missing-value summary
- about/methodology section aligned with the MSc project specification

## Free Hosting

The easiest free deployment path is Streamlit Community Cloud:

1. Push this folder to a GitHub repository.
2. Go to `https://share.streamlit.io`.
3. Sign in with GitHub and choose `Create app`.
4. Select the repository, branch, and `streamlit_app.py` as the entrypoint.
5. Deploy the app.

If you do not upload `ms.csv` to GitHub, viewers can still use the app by uploading the CSV in the sidebar. If you want the public app to run immediately without upload, remove `ms.csv` from `.gitignore` and commit the dataset only if you have permission to redistribute it.

## Main Outputs

- `model_comparison.csv`
- `cv_scores.csv`
- `badminton_best_model.pkl`
- `scaler.pkl`
- `selected_features.txt`
- EDA charts
- ROC, calibration, confusion matrix, and learning curve charts
- SHAP plots when explainability is enabled
- LIME HTML and PNG explanations when explainability is enabled

## Notes for GitHub Upload

- Generated files are ignored by `.gitignore`.
- `ms.csv` is ignored by default because it comes from an external Kaggle dataset. If you have redistribution rights and want to include it, remove `ms.csv` from `.gitignore`.
- Add a license file before publishing publicly if you want others to reuse the code.
