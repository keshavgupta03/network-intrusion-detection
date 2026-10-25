# Network Intrusion Detection (NSL-KDD)

ML pipeline comparing 6 classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, MLP) plus a Streamlit app.

## Setup
1. `pip install -r requirements.txt`
2. Put `KDDTrain+.txt` and `KDDTest+.txt` (NSL-KDD) inside `data/`
3. `python pipeline.py`  -> trains, evaluates, saves figures/tables/model to `outputs/`
4. `streamlit run app.py`

Set `TASK = "multiclass"` in `pipeline.py` for Normal/DoS/Probe/R2L/U2R.

## Method
Duplicate removal -> constant-column drop -> stratified 70/15/15 split -> one-hot + standard scaling inside a sklearn Pipeline (no leakage) -> 3-fold CV -> model selection on validation F1 -> final report on held-out test and on KDDTest+ (unseen attacks).

## Live demo
<add your deployed URL here>
