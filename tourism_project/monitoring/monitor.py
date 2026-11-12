"""Model monitoring: feature drift, prediction drift, model integrity and app health.

Run on a schedule by .github/workflows/monitor.yml. Exits with code 1 (so the workflow fails and
GitHub notifies the owner) when significant drift, a model/metadata mismatch or an unhealthy app is found.

Population Stability Index (PSI) guide: < 0.10 stable, 0.10-0.25 moderate shift (warning), > 0.25 significant shift (alert).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEPLOY = Path("tourism_project/deployment")
REF_PATH = "tourism_project/monitoring/reference_sample.csv"   # training features at training time
DATA_PATH = "tourism_project/data/tourism.csv"                  # latest data: replace/append with new campaign data
PSI_WARN, PSI_ALERT = 0.10, 0.25
SHARE_ALERT = 0.10        # alert if the share of customers above the threshold moves by 10 percentage points
MIN_ROWS = 200            # PSI is too noisy to act on with fewer rows


def clean(df):
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    if "Gender" in df:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    return df


def _psi(r, c, eps=1e-4):
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref.dropna(), edges)[0] / ref.notna().sum()
    c = np.histogram(cur.dropna(), edges)[0] / max(cur.notna().sum(), 1)
    return _psi(r, c)


def psi_categorical(ref, cur):
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    return _psi(r, c)


def status(psi):
    return "ALERT" if psi >= PSI_ALERT else "WARN" if psi >= PSI_WARN else "OK"


def app_health(url, tries=3, wait=10):
    for attempt in range(1, tries + 1):
        try:
            import requests
            code = requests.get(url.rstrip("/") + "/_stcore/health", timeout=30).status_code
            if code == 200:
                return True, f"HTTP 200 from {url}"
            msg = f"HTTP {code} from {url}"
        except Exception as exc:  # network error, DNS, timeout
            msg = f"request failed: {exc}"
        if attempt < tries:
            time.sleep(wait)
    return False, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH, help="CSV with the latest customer data")
    ap.add_argument("--app-url", default=os.environ.get("APP_URL"), help="Deployed Streamlit app URL")
    ap.add_argument("--deploy-dir", default=str(DEPLOY), help="Folder holding best_model.joblib and model_metadata.json")
    args = ap.parse_args()
    deploy = Path(args.deploy_dir)

    artifact = joblib.load(deploy / "best_model.joblib")
    model, threshold, features = artifact["model"], artifact["threshold"], artifact["features"]
    meta = json.loads((deploy / "model_metadata.json").read_text())
    report, failed = {}, False

    # 1. Model integrity: the deployed file must match the metadata committed with it
    sha = hashlib.sha256((deploy / "best_model.joblib").read_bytes()).hexdigest()
    ok = sha == meta["model_sha256"]
    failed |= not ok
    report["model_integrity"] = {"status": "OK" if ok else "ALERT", "sha256": sha[:12]}
    print(f"[{'OK' if ok else 'ALERT'}] model integrity: file {sha[:12]} {'matches' if ok else 'DOES NOT MATCH'} metadata")

    # 2. Feature drift (PSI per feature, reference = training sample)
    ref = clean(pd.read_csv(REF_PATH))[features]
    cur = clean(pd.read_csv(args.data))[features]
    print(f"reference rows: {len(ref)}, current rows: {len(cur)}")
    if len(cur) < MIN_ROWS:
        print(f"[WARN] only {len(cur)} current rows (< {MIN_ROWS}); drift values are unreliable")
    feat = {}
    for col in features:
        small = ref[col].nunique() <= 10
        feat[col] = psi_categorical(ref[col], cur[col]) if (small or ref[col].dtype == object) else psi_numeric(ref[col], cur[col])
    print("\nFeature drift (PSI):")
    for col, v in sorted(feat.items(), key=lambda kv: -kv[1]):
        print(f"  [{status(v):<5}] {col:<26} {v:.3f}")
    worst = max(feat.values())
    failed |= worst >= PSI_ALERT
    report["feature_drift"] = {k: round(v, 4) for k, v in feat.items()}

    # 3. Prediction drift using the CURRENTLY deployed model on both samples
    p_ref, p_cur = model.predict_proba(ref)[:, 1], model.predict_proba(cur)[:, 1]
    score_psi = psi_numeric(pd.Series(p_ref), pd.Series(p_cur))
    share_ref, share_cur = float((p_ref >= threshold).mean()), float((p_cur >= threshold).mean())
    share_bad = abs(share_cur - share_ref) > SHARE_ALERT
    failed |= (score_psi >= PSI_ALERT) or share_bad
    print(f"\n[{status(score_psi)}] score PSI {score_psi:.3f}")
    print(f"[{'ALERT' if share_bad else 'OK'}] share of customers above threshold {threshold:.2f}: reference {share_ref:.1%}, current {share_cur:.1%}")
    report["prediction_drift"] = {"score_psi": round(score_psi, 4), "share_above_threshold_reference": round(share_ref, 4),
                                  "share_above_threshold_current": round(share_cur, 4)}

    # 4. Deployment health
    if args.app_url:
        healthy, msg = app_health(args.app_url)
        failed |= not healthy
        report["app_health"] = {"status": "OK" if healthy else "ALERT", "detail": msg}
        print(f"\n[{'OK' if healthy else 'ALERT'}] app health: {msg}")
    else:
        print("\n[SKIPPED] app health: no APP_URL provided")
        report["app_health"] = {"status": "SKIPPED"}

    report["overall"] = "ALERT" if failed else ("WARN" if worst >= PSI_WARN else "OK")
    Path("monitoring_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nOverall: {report['overall']}  (details in monitoring_report.json)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
