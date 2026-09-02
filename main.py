"""
DataVerse Data Playground — Backend (Complete: Phases 1-6)

Endpoints:
  GET  /                        -> health check
  POST /upload                   -> Phase 1: validation report
  POST /analyze/eda              -> Phase 2: statistics + missing-value deep dive
  POST /analyze/visualize        -> Phase 3: chart-ready data (histograms, bar charts, scatter, correlation)
  POST /analyze/suggest-features -> Phase 4: rule-based feature suggestions
  POST /model/train              -> Phase 5: auto model selection + training + metrics
  POST /model/explain            -> Phase 6: natural-language explanation of metrics (Gemini)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

app = FastAPI(title="DataVerse Data Playground API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_csv_upload(contents: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV has no data")
    return df


def _require_csv(filename: str):
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "DataVerse Playground API is running"}


# ---------------------------------------------------------------------------
# PHASE 1 — Upload + Validation
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    _require_csv(file.filename)
    df = _read_csv_upload(await file.read())

    n_rows, n_cols = df.shape
    column_report = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        column_report.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_pct": round((missing_count / n_rows) * 100, 2),
            "unique_values": int(df[col].nunique()),
        })

    preview = df.head(5).astype(object).where(pd.notnull(df.head(5)), None).to_dict(orient="records")

    return {
        "filename": file.filename,
        "row_count": n_rows,
        "column_count": n_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": column_report,
        "preview": preview,
    }


# ---------------------------------------------------------------------------
# PHASE 2 — Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
@app.post("/analyze/eda")
async def analyze_eda(file: UploadFile = File(...)):
    _require_csv(file.filename)
    df = _read_csv_upload(await file.read())

    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_stats = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        numeric_stats.append({
            "column": col,
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std_dev": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
            "outlier_count": int(len(outliers)),
        })

    categorical_stats = []
    for col in categorical_cols:
        series = df[col].dropna()
        top_values = series.value_counts().head(5)
        categorical_stats.append({
            "column": col,
            "unique_count": int(df[col].nunique()),
            "top_values": [{"value": str(v), "count": int(c)} for v, c in top_values.items()],
        })

    missing_report = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_report.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": round((missing_count / n_rows) * 100, 2),
        })
    high_missing_columns = [r["column"] for r in missing_report if r["missing_pct"] > 50]

    correlation_matrix = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True).round(3)
        correlation_matrix = corr_df.where(pd.notnull(corr_df), None).to_dict()

    return {
        "row_count": n_rows,
        "column_count": len(df.columns),
        "numeric_column_count": len(numeric_cols),
        "categorical_column_count": len(categorical_cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "missing_value_report": missing_report,
        "total_missing_cells": int(df.isna().sum().sum()),
        "high_missing_columns": high_missing_columns,
        "correlation_matrix": correlation_matrix,
    }


# ---------------------------------------------------------------------------
# PHASE 3 — Visualization-ready data
# ---------------------------------------------------------------------------
@app.post("/analyze/visualize")
async def analyze_visualize(file: UploadFile = File(...)):
    _require_csv(file.filename)
    df = _read_csv_upload(await file.read())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    histograms = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        counts, bin_edges = np.histogram(series, bins=10)
        histograms[col] = {
            "bin_edges": [round(float(b), 4) for b in bin_edges],
            "counts": [int(c) for c in counts],
        }

    categorical_bar_charts = {}
    for col in categorical_cols:
        top = df[col].dropna().value_counts().head(10)
        categorical_bar_charts[col] = [{"value": str(v), "count": int(c)} for v, c in top.items()]

    scatter_pairs = []
    for i in range(min(3, len(numeric_cols))):
        for j in range(i + 1, min(3, len(numeric_cols))):
            x_col, y_col = numeric_cols[i], numeric_cols[j]
            pair_df = df[[x_col, y_col]].dropna()
            if len(pair_df) > 300:
                pair_df = pair_df.sample(300, random_state=42)
            scatter_pairs.append({
                "x_col": x_col,
                "y_col": y_col,
                "points": [{"x": round(float(r[x_col]), 4), "y": round(float(r[y_col]), 4)}
                           for _, r in pair_df.iterrows()],
            })

    correlation_heatmap = {}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True).round(3)
        correlation_heatmap = corr_df.where(pd.notnull(corr_df), None).to_dict()

    return {
        "histograms": histograms,
        "categorical_bar_charts": categorical_bar_charts,
        "scatter_pairs": scatter_pairs,
        "correlation_heatmap": correlation_heatmap,
    }


# ---------------------------------------------------------------------------
# PHASE 4 — Rule-based feature suggestions
# ---------------------------------------------------------------------------
@app.post("/analyze/suggest-features")
async def suggest_features(file: UploadFile = File(...)):
    _require_csv(file.filename)
    df = _read_csv_upload(await file.read())

    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    suggestions = []

    for col in df.columns:
        unique_count = df[col].nunique()
        missing_pct = (df[col].isna().sum() / n_rows) * 100

        if unique_count == 1:
            suggestions.append({"column": col, "issue": "constant_column",
                                 "message": f"'{col}' has only one unique value — consider dropping it, it adds no information."})
            continue

        if missing_pct > 30:
            suggestions.append({"column": col, "issue": "high_missing",
                                 "message": f"'{col}' is {round(missing_pct,1)}% missing — consider dropping or carefully imputing this column."})

        if col not in numeric_cols and unique_count > 0.5 * n_rows:
            suggestions.append({"column": col, "issue": "high_cardinality",
                                 "message": f"'{col}' has {unique_count} unique values (high cardinality) — consider grouping rare categories or dropping before modeling."})

        if col in numeric_cols:
            skew = df[col].dropna().skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                suggestions.append({"column": col, "issue": "skewed_distribution",
                                     "message": f"'{col}' is skewed {direction} (skew={round(float(skew),2)}) — consider a log transform before modeling."})

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        seen = set()
        for col_a in numeric_cols:
            for col_b in numeric_cols:
                if col_a == col_b or (col_b, col_a) in seen:
                    continue
                seen.add((col_a, col_b))
                val = corr.loc[col_a, col_b]
                if pd.notnull(val) and val > 0.9:
                    suggestions.append({
                        "column": f"{col_a} & {col_b}", "issue": "high_correlation",
                        "message": f"'{col_a}' and '{col_b}' are highly correlated ({round(float(val),2)}) — consider removing one to avoid redundancy."
                    })

    return {"suggestion_count": len(suggestions), "suggestions": suggestions}


# ---------------------------------------------------------------------------
# PHASE 5 — Model selection + training
# ---------------------------------------------------------------------------
@app.post("/model/train")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...)):
    _require_csv(file.filename)
    df = _read_csv_upload(await file.read())

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' not found in the uploaded file")

    df = df.dropna(subset=[target_column])
    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Not enough rows with a non-missing target to train a model (need at least 20)")

    y_raw = df[target_column]
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No numeric feature columns available to train on (this version only uses numeric features)")

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)
    task_type = "regression" if (is_numeric_target and y_raw.nunique() > 15) else "classification"

    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    results = {}

    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_score": round(float(f1_score(y_test, preds, average="weighted")), 4),
                "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds, average="weighted")), 4),
            }
        best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                "r2_score": round(float(r2_score(y_test, preds)), 4),
            }
        best_model_name = max(results, key=lambda k: results[k]["r2_score"])

    rf_key = "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"
    feature_importance = []
    if rf_key in models:
        importances = models[rf_key].feature_importances_
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, importances)],
            key=lambda d: d["importance"], reverse=True,
        )

    return {
        "task_type": task_type,
        "target_column": target_column,
        "features_used": feature_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model_results": results,
        "best_model": best_model_name,
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# PHASE 6 — Natural-language explanation of results (Gemini)
# ---------------------------------------------------------------------------
@app.post("/model/explain")
async def explain_results(payload: dict):
    """
    Accepts a JSON body containing results from /model/train or /analyze/eda,
    and returns a plain-English explanation generated by Gemini.

    Requires an environment variable GEMINI_API_KEY set on the server
    (Render: Dashboard -> your service -> Environment -> Add Environment Variable).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on the server")

    prompt = (
        "You are explaining a data science result to someone reviewing a portfolio project. "
        "Explain the following results in plain, clear English in 3-5 sentences. "
        "Be specific about what the numbers mean and whether the result looks strong or weak, "
        "but do not invent any numbers that are not given below.\n\n"
        f"Results:\n{payload}"
    )

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini API: {e}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API returned an error ({response.status_code}): {response.text[:300]}",
        )

    data = response.json()
    try:
        explanation = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected response format from Gemini API")

    return {"explanation": explanation}
