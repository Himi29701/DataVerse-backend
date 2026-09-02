# DataVerse Data Playground — Backend (Complete)

One file, `main.py`, containing all 6 phases:

| Phase | Endpoint | What it does |
|---|---|---|
| 1 | `POST /upload` | Row/column counts, dtypes, missing values, duplicates, preview |
| 2 | `POST /analyze/eda` | Full descriptive stats, outlier counts, correlation matrix, missing-value deep dive |
| 3 | `POST /analyze/visualize` | Chart-ready data: histograms, bar charts, scatter points, correlation heatmap |
| 4 | `POST /analyze/suggest-features` | Rule-based suggestions (high missing, skew, high cardinality, constant columns, correlated pairs) |
| 5 | `POST /model/train` | Auto-detects classification vs regression, trains 2 models, returns metrics + feature importance |
| 6 | `POST /model/explain` | Sends results to Gemini, returns a plain-English explanation |

## Deploying (first time)

1. **GitHub:** Create a repo (e.g. `dataverse-backend`), upload `main.py`,
   `requirements.txt`, and this `README.md`, commit.
2. **Render:** New + → Web Service → connect the repo →
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Instance Type: Free
   - Create Web Service, wait for the public URL.

## Updating an existing deployment (you already did Phase 1/2)

1. Open your repo on GitHub → click `main.py` → pencil/edit icon
2. Select all, delete, paste in this new complete version
3. Commit changes
4. Also update `requirements.txt` the same way (new libraries added: numpy,
   scikit-learn, requests)
5. Render auto-redeploys within a minute or two — check the "Logs" tab for "Live"

## IMPORTANT — Setting up the Gemini key for Phase 6

Phase 6 needs your Gemini API key available to the *Python backend*, separately
from whatever key you set up in Lovable (these are two different systems).

1. Go to your **Render dashboard** → click your web service
2. Go to the **Environment** tab
3. Click **Add Environment Variable**
4. Key: `GEMINI_API_KEY`   Value: (paste your Gemini key)
5. Save — Render will redeploy automatically with the key available

**Known issue:** if your Gemini key starts with `AQ.` instead of `AIzaSy`,
it may be rejected by Google's API right now — this is a known, currently
unresolved bug on Google's side affecting some accounts (not a bug in this
code). If `/model/explain` returns a Gemini error, this is very likely why.
Every other endpoint (Phases 1-5) works independently and does not need this
key at all.

## Testing each endpoint
All endpoints except `/model/explain` expect a CSV file upload (form-data,
key name `file`). `/model/train` also needs a `target_column` form field
(the column name you want to predict). `/model/explain` expects a JSON body
with whatever results you want explained.

## Note on free tier
Render's free tier sleeps after 15 minutes of no traffic — first request
after sleeping takes ~30-50 seconds. This is normal, not a bug.

## What this version does NOT do (by design, for honesty in interviews)
- Only uses **numeric columns** as model features (Phase 5) — categorical
  encoding is a natural next improvement, not yet built
- Model selection is limited to 2 candidate models per task type, not a
  full AutoML search
- No hyperparameter tuning — models use reasonable defaults
