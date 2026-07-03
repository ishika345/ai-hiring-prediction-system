# AI-Based Hiring Prediction System

An end-to-end machine learning project that predicts whether a candidate will be **Hired** or **Rejected** based on resume attributes — skills, experience, education, certifications, job role, salary expectation, and project count. Simulates a real-world AI resume-screening system used in HR analytics.

## Dataset

1,000 synthetic resumes with the following columns:

| Column | Description |
|---|---|
| `Resume_ID` | Unique identifier |
| `Name` | Candidate name |
| `Skills` | Comma-separated list of technical skills |
| `Experience (Years)` | Total work experience |
| `Education` | Highest qualification (B.Sc, B.Tech, M.Tech, MBA, PhD) |
| `Certifications` | Relevant certifications (or none) |
| `Job Role` | Applied job role |
| `Recruiter Decision` | **Target** — Hire / Reject |
| `Salary Expectation ($)` | Expected salary |
| `Projects Count` | Number of projects completed |
| `AI Score (0-100)` | A pre-computed score (see note below) |

##  Key finding: target leakage

EDA revealed that `AI Score (0-100)` **perfectly separates the two classes** — every score ≥ 61 is "Hire," every score ≤ 60 is "Reject," with zero exceptions. This indicates the score was used to *generate* the label rather than being an independent signal. Including it as a feature would let the model learn a trivial threshold rule instead of genuine resume signals, and would not generalize to any real screening scenario where that score isn't already computed by the same rule.

**This column is deliberately excluded from the feature set.** The notebook documents the evidence for this decision in Section 3.3 (threshold test + boxplot).

## Approach

1. **EDA** — missing values, class balance, target leakage check, feature-vs-target relationships, skill frequency analysis.
2. **Preprocessing**
   - `Certifications` nulls filled with an explicit `"None"` category (not dropped — "no certification" is real information).
   - `Skills` multi-hot encoded (14 binary columns) since each resume lists multiple skills.
   - `Education`, `Job Role`, `Certifications` one-hot encoded.
   - Numeric features (`Experience`, `Salary Expectation`, `Projects Count`) kept as-is, scaled separately for distance-based models.
   - Stratified 80/20 train/test split to preserve the ~81% / 19% class balance.
3. **Modeling** — five classifiers trained and tuned with 5-fold stratified cross-validation (`GridSearchCV`, scoring on F1 due to class imbalance):
   - Logistic Regression (`class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`, tuned)
   - Gradient Boosting (tuned)
   - XGBoost (`scale_pos_weight`, tuned)
   - SVM / RBF kernel (`class_weight='balanced'`, tuned)
4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves.
5. **Feature importance** — Random Forest and XGBoost importances to identify which resume attributes actually drive predictions.

## Results

After excluding the leaked `AI Score` column, all five models perform strongly and realistically:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting (tuned) | 0.965 | 0.975 | 0.981 | 0.978 | 0.995 |
| SVM (tuned) | 0.960 | 0.981 | 0.969 | 0.975 | 0.996 |
| XGBoost (tuned) | 0.955 | 0.994 | 0.951 | 0.972 | 0.995 |
| Random Forest (tuned) | 0.945 | 0.969 | 0.963 | 0.966 | 0.989 |
| Logistic Regression | 0.940 | 1.000 | 0.926 | 0.962 | 0.999 |

**Top predictive features:** `Experience (Years)` and `Projects Count` dominate feature importance by a wide margin in both tree-based models — far more influential than education tier, specific certifications, or individual skill flags. This is a realistic, interpretable result: demonstrated track record matters more than credentials alone.

## Repository structure

```
.
├── notebooks/
│   └── AI_Hiring_Prediction_System.ipynb   # Full end-to-end pipeline
├── data/
│   └── hiring_data.csv                     # Source dataset
├── requirements.txt
└── README.md
```

## Setup & usage

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/AI_Hiring_Prediction_System.ipynb
```

Run all cells top to bottom. The notebook reads `../data/hiring_data.csv` by default — adjust `DATA_PATH` in the second code cell if you reorganize folders.

## Possible extensions

- SMOTE / SMOTENC oversampling as an alternative to class-weighting
- SHAP values for instance-level explainability (relevant for an HR decision-support tool)
- Probability calibration if predicted confidence is shown to recruiters
- Fairness/bias audit across Education and Job Role groups
