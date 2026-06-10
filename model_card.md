# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model type:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Number of estimators:** 100
- **Random state:** 42
- **Training framework:** scikit-learn 1.7.2
- **Developer:** Udacity MLOps Nanodegree — Project 3
- **Version:** 1.0.0

## Intended Use

This model is intended to predict whether a US adult's annual income exceeds $50,000,
based on demographic and employment attributes from the 1994 US Census Bureau database.

**Intended users:** students and developers demonstrating MLOps deployment pipelines.
**Out-of-scope uses:** this model should **not** be used for any real-world hiring, lending,
insurance, or eligibility decisions. The training data is from 1994 and may not reflect
current economic conditions or population demographics.

## Training Data

- **Source:** UCI Machine Learning Repository — [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)
- **Size:** ~32,561 records after cleaning
- **Preprocessing:** Whitespace stripped from all fields; categorical features encoded with
  `OneHotEncoder`; target label (`salary`) binarized with `LabelBinarizer`
- **Train split:** 80% (~26,048 rows)

**Categorical features used:**
`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`,
`native-country`

**Continuous features:** `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`,
`hours-per-week`

**Label:** `salary` — binary: `<=50K` (0) or `>50K` (1)

## Evaluation Data

- **Test split:** 20% (~6,513 rows), held out before training
- Same preprocessing pipeline applied (encoder and label binarizer fitted on training
  data only, then applied to test data)

## Metrics

Performance on the held-out test set:

| Metric    | Value  |
|-----------|--------|
| Precision | ~0.87  |
| Recall    | ~0.62  |
| F1 (Fβ=1)| ~0.73  |

*Exact values are printed to stdout and written to `slice_output.txt` when `train_model.py`
is executed. Values above are representative; retrain to get precise figures.*

## Ethical Considerations

- The dataset contains sensitive demographic attributes including `race`, `sex`, and
  `native-country`. Model performance varies across these slices (see `slice_output.txt`).
- The model is trained on 1994 US Census data and reflects historical socioeconomic
  disparities of that era. Predictions should not be interpreted as reflecting an
  individual's true earning potential.
- Female and minority sub-groups historically earn less, so the model may amplify
  those disparities rather than reflect underlying productivity.
- **This model must not be used in production systems that affect people's lives**
  (credit, employment, housing, etc.) without rigorous fairness auditing.

## Caveats and Recommendations

- **Data staleness:** The 1994 census data is 30+ years old. Income thresholds,
  occupational categories, and demographic distributions have changed substantially.
- **Class imbalance:** Roughly 75% of records are `<=50K`, which may bias the model.
  Consider oversampling or class-weight adjustments for production use.
- **Slice performance:** Review `slice_output.txt` carefully. Certain `native-country`
  and `occupation` slices have very small sample sizes, leading to unstable metrics.
- **Recommended improvements:** Use cross-validation, tune hyperparameters (e.g.,
  `max_depth`, `min_samples_leaf`), and run a fairness audit (e.g., with `aequitas`)
  before any real-world deployment.
