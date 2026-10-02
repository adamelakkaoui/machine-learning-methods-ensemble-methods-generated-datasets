# Machine Learning Methods (Ensemble methods & Generated datasets)

Academic study combining pedagogical implementations of Random Forest and Gradient Boosting with a separate exercise on generated classification datasets. The manual estimators are compared with scikit-learn while preserving the original step-by-step French function naming.

## Contents and data provenance

- `src/random_forest.py` loads scikit-learn's Diabetes dataset, converts its continuous target to binary labels with the coursework threshold `target > 140`, then compares a manual bootstrap forest with `RandomForestClassifier`.
- `src/gradient_boosting.py` loads the same Diabetes features and keeps the original continuous target for regression, comparing manual residual fitting with `GradientBoostingRegressor`.
- `src/generated_datasets.py` is a distinct synthetic-data exercise: it creates Gaussian class clusters manually and compares them with `sklearn.datasets.make_classification`.
- `notebooks/` contains the notebooks corresponding to the three academic exercises.
- [French academic report (PDF)](docs/academic-report-fr.pdf).
- [French presentation (PPTX)](presentations/ensemble-methods-generated-datasets-fr.pptx).

The Diabetes dataset is bundled by scikit-learn and is used for the ensemble-method exercises. The third exercise creates generated classification datasets.

## Methods

The manual forest performs bootstrap sampling, Gini-based splits, per-node random feature selection and majority voting. The boosting implementation starts from the training-target mean and fits shallow regression trees sequentially to residuals. These implementations are educational and intentionally much simpler than the optimized scikit-learn estimators.

## Installation and use

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/random_forest.py
python src/gradient_boosting.py
python src/generated_datasets.py
```

## Results

The academic report presents the following manual/scikit-learn comparisons:

| Method | Manual implementation | scikit-learn |
|---|---:|---:|
| Random Forest accuracy | `0.719` | `0.730` |
| Gradient Boosting MSE | `2858.725` | `2849.616` |

The project also compares a manually generated classification dataset with scikit-learn's `make_classification`, illustrating how generated datasets can be used to study and test machine-learning methods under controlled conditions.

The report concludes that ensemble methods improve predictive performance by combining multiple models, while the from-scratch implementations make the underlying mechanisms—Gini impurity, bootstrapping, majority voting, residual fitting and additive prediction—explicit.

## Author

Adam El Akkaoui
