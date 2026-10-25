"""Streamlit UI for the NSL-KDD intrusion detector. Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(OUT / "best_model.joblib")
    meta = json.loads((OUT / "meta.json").read_text())
    samples = pd.read_csv(OUT / "samples.csv")
    return model, meta, samples


model, meta, samples = load()
classes, features = meta["classes"], meta["features"]


def predict(df):
    X = df[features]
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    res = df.copy()
    res["prediction"] = [classes[i] for i in pred]
    res["confidence"] = proba.max(axis=1).round(3)
    return res


st.title("🛡️ Network Intrusion Detection")
st.caption(f"Model: **{meta['best_model']}** · Task: {meta['task']} · Dataset: NSL-KDD")

m = meta["test_metrics"]
c = st.columns(5)
for col, k in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
    col.metric(k, f"{m[k]:.3f}")

tab1, tab2, tab3 = st.tabs(["Live demo (sample traffic)", "Upload CSV", "Manual entry"])

with tab1:
    st.write("Pick a captured connection record and let the model classify it.")
    idx = st.selectbox("Sample record", range(len(samples)),
                       format_func=lambda i: f"Record #{i}")
    row = samples.iloc[[idx]]
    st.dataframe(row[features].T.rename(columns={row.index[0]: "value"}), height=250)
    if st.button("Classify this record", type="primary"):
        r = predict(row).iloc[0]
        (st.error if r["prediction"] not in ("normal", "Normal") else st.success)(
            f"Prediction: **{r['prediction']}** (confidence {r['confidence']:.1%})")
        st.caption(f"Ground truth label: {row['true_label'].iloc[0]}")

with tab2:
    st.write("Upload a CSV with the NSL-KDD feature columns (header row required).")
    st.download_button("Download a sample CSV", samples.to_csv(index=False), "sample_traffic.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            res = predict(data)
            st.write(res["prediction"].value_counts().rename("count").to_frame().T)
            st.dataframe(res[["prediction", "confidence"] + features], height=350)
            st.download_button("Download results", res.to_csv(index=False), "predictions.csv")

with tab3:
    st.write("Set the key fields; all other features default to the sample median.")
    base = samples[features].copy()
    defaults = {f: (base[f].median() if pd.api.types.is_numeric_dtype(base[f]) else base[f].mode()[0]) for f in features}
    a, b, d = st.columns(3)
    defaults["protocol_type"] = a.selectbox("Protocol", meta["categories"]["protocol_type"])
    defaults["service"] = b.selectbox("Service", meta["categories"]["service"])
    defaults["flag"] = d.selectbox("Flag", meta["categories"]["flag"])
    defaults["src_bytes"] = a.number_input("src_bytes", 0, 10**9, int(defaults["src_bytes"]))
    defaults["dst_bytes"] = b.number_input("dst_bytes", 0, 10**9, int(defaults["dst_bytes"]))
    defaults["count"] = d.number_input("count", 0, 600, int(defaults["count"]))
    defaults["serror_rate"] = a.slider("serror_rate", 0.0, 1.0, float(defaults["serror_rate"]))
    defaults["logged_in"] = b.selectbox("logged_in", [0, 1], index=int(defaults["logged_in"]))
    if st.button("Classify manual entry"):
        r = predict(pd.DataFrame([defaults])).iloc[0]
        (st.error if r["prediction"] not in ("normal", "Normal") else st.success)(
            f"Prediction: **{r['prediction']}** (confidence {r['confidence']:.1%})")
