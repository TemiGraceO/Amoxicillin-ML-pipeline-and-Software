# %% [markdown]
# # Amoxicillin: ML prediction of plasma concentration and PD target attainment
# Reference data: PK-Sim exports (three regimens). Run the parts **in order**, one at a time.
# Every part saves its output to Google Drive. If Colab restarts, run **Part 1** again,
# then continue from the part you were on.

# %% PART 1: Setup and settings (run first, and again after any restart)
import os, sys, json, time, warnings, platform, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")

# ---- Folder that contains your six PK-Sim CSV files --------------------------------------
BASE_DIR = "/content/drive/MyDrive/amoxicillin"
OUT_DIR = os.path.join(BASE_DIR, "results")
for sub in ["", "data", "models", "tables", "figures"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

# ---- Regimens (file names must match what you exported) ----------------------------------
FILES = {
    "500mg_q8h":   dict(patients="500mg_q8h_patients.csv",   results="500mg_q8h_results.csv",   dose_mg=500.0,  tau_h=8.0),
    "1000mg_q8h":  dict(patients="1000mg_q8h_patients.csv",  results="1000mg_q8h_results.csv",  dose_mg=1000.0, tau_h=8.0),
    "1000mg_q12h": dict(patients="1000mg_q12h_patients.csv", results="1000mg_q12h_results.csv", dose_mg=1000.0, tau_h=12.0),
}
END_H = 72.0            # length of each simulation (hours)
GRID_STEP_H = 0.25      # analysis grid: one point every 15 minutes
N_INDIVIDUALS = 1000    # how many virtual people to use (None = all). 1000 keeps Colab light.
SEED = 42

# ---- Pharmacodynamic settings (Chapter 3) ------------------------------------------------
PD_TARGET_PCT = 40.0
PRIMARY_MIC = 1.0
MIC_GRID = [0.125, 0.25, 0.5, 1, 2, 4, 8]
CONCENTRATION_TYPE = "total"     # "total" or "free" (what the exported column contains)
FRACTION_UNBOUND = None          # REQUIRED if "total": value from the PK-Sim compound (see Part 8)

# ---- Modelling settings ------------------------------------------------------------------
TEST_FRACTION = 0.20
CV_FOLDS = 3
TUNE_N_IDS = 150        # individuals used while tuning only
TUNE_TREES = 60
FINAL_TREES = 150
ALPHA = 0.05
N_BOOT = 2000

random.seed(SEED); np.random.seed(SEED)
P = lambda *a: os.path.join(OUT_DIR, *a)


# ---- Helper functions used by later parts -----------------------------------------------
def detect_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {s: first.count(s) for s in [",", ";", "\t"]}
    return max(counts, key=counts.get)


def clean_ids(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all():
        return num.astype("int64").astype(str)
    return series.astype(str).str.strip()


def to_hours(values, unit):
    if unit == "hours":
        return values
    if unit == "minutes":
        return values / 60.0
    raise ValueError("TIME_UNIT must be 'hours' or 'minutes'")


def to_mg_per_l(values, unit):
    if unit in ("mg/L", "ug/mL"):
        return values
    if unit == "umol/L":
        return values * 0.3654          # amoxicillin 365.4 g/mol
    raise ValueError("CONC_UNIT must be 'mg/L', 'ug/mL' or 'umol/L'")


versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "seed": SEED}
import scipy, sklearn, matplotlib
versions.update({"scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                 "matplotlib": matplotlib.__version__})
json.dump(versions, open(P("software_versions.json"), "w"), indent=2)

# Reload column choices saved by Part 3 (so a restart does not lose them)
COLS_FILE = P("columns.json")
if os.path.exists(COLS_FILE):
    globals().update(json.load(open(COLS_FILE)))
    print("Loaded saved column choices:", {k: globals()[k] for k in
          ["ID_COL", "TIME_COL", "CONC_COL", "PATIENT_FEATURES", "TIME_UNIT", "CONC_UNIT"]})
print("Part 1 done. Versions:", versions)

# %% PART 2: Look at your files (reads only a little, will not crash)
raw_info = {}
for regimen, cfg in FILES.items():
    rp = os.path.join(BASE_DIR, cfg["results"])
    pp = os.path.join(BASE_DIR, cfg["patients"])
    for p in (rp, pp):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}\nCheck BASE_DIR and the file names in FILES.")
    rs, ps = detect_sep(rp), detect_sep(pp)
    rhead = pd.read_csv(rp, sep=rs, nrows=5, encoding="utf-8-sig")
    phead = pd.read_csv(pp, sep=ps, nrows=5, encoding="utf-8-sig")
    rhead.columns = rhead.columns.astype(str).str.strip()
    phead.columns = phead.columns.astype(str).str.strip()
    n_res = sum(1 for _ in open(rp, "rb")) - 1
    n_pat = sum(1 for _ in open(pp, "rb")) - 1
    raw_info[regimen] = dict(rs=rs, ps=ps, rcols=list(rhead.columns), pcols=list(phead.columns))
    print("=" * 70)
    print(regimen, f"| result rows: {n_res:,} | patient rows: {n_pat:,}")
    print("RESULT columns :", list(rhead.columns))
    print("PATIENT columns:", list(phead.columns))
    if regimen == list(FILES)[0]:
        print("\nFirst rows of the results file:\n", rhead.head(3).to_string())
        print("\nFirst rows of the patients file:\n", phead.head(3).to_string())


def guess_columns(info):
    rc, pc = info["rcols"], info["pcols"]
    low = lambda s: s.lower()
    idc = next((c for c in rc if "individualid" in low(c).replace(" ", "").replace("_", "")), rc[0])
    timec = next((c for c in rc if low(c).startswith("time")), None)
    conc_cands = [c for c in rc if c not in (idc, timec) and "plasma" in low(c)]
    if not conc_cands:
        conc_cands = [c for c in rc if c not in (idc, timec)]
    feats = []
    for key in ["age", "weight", "height", "gfr", "gender", "sex"]:
        for c in pc:
            if key in low(c) and c not in feats and "individualid" not in low(c):
                feats.append(c)
                break
    return dict(id=idc, time=timec, conc_candidates=conc_cands, features=feats)


G = guess_columns(raw_info[list(FILES)[0]])
if G["time"] is None:
    raise ValueError("I could not find a time column. Look at the RESULT columns printed above, then set "
                     "TIME_COL by hand in Part 3 (and set G['time'] to it before Part 2 continues).")
print("\n" + "=" * 70)
print("MY GUESSES (check them):")
for k, v in G.items():
    print(f"  {k}: {v}")

# Check the time unit and concentration scale on the first regimen (reads one column only)
first = list(FILES)[0]
cfg = FILES[first]
rp = os.path.join(BASE_DIR, cfg["results"])
conc_guess = G["conc_candidates"][0]
tmax, cmax, cmin = -np.inf, -np.inf, np.inf
for ch in pd.read_csv(rp, sep=raw_info[first]["rs"], usecols=[G["time"], conc_guess],
                      chunksize=1_000_000, encoding="utf-8-sig"):
    ch.columns = ch.columns.astype(str).str.strip()
    tmax = max(tmax, float(ch[G["time"]].max()))
    cmax = max(cmax, float(ch[conc_guess].max()))
    cmin = min(cmin, float(ch[conc_guess].min()))
G["time_unit"] = "hours" if tmax < END_H * 3 else "minutes"
print(f"\nLargest time value in file: {tmax:g}  -> I guess the time unit is: {G['time_unit']}")
print(f"Concentration column '{conc_guess}': min {cmin:.4g}, max {cmax:.4g} (raw units)")
print("Compare the raw max with the PK-Sim chart: if the chart peak is about 100 umol/L and the")
print("raw max is about 100, the unit is umol/L. If the raw max is about 36, it is mg/L.")

# %% PART 3: Confirm the column choices (edit only if my guesses were wrong)
ID_COL = G["id"]
TIME_COL = G["time"]
CONC_COL = G["conc_candidates"][0]        # if several candidates were printed, choose the TOTAL plasma one
PATIENT_FEATURES = G["features"]          # for example Age, Weight, Height, Gender
TIME_UNIT = G["time_unit"]                # "hours" or "minutes"
CONC_UNIT = "umol/L"                      # "umol/L", "mg/L" or "ug/mL"  (CONFIRM from the PK-Sim chart)

# Expected ranges of the Witowski1982 population (age 25 to 44 years, weight 51 to 86 kg).
# Used only to catch unit mistakes. Set to {} to skip.
EXPECTED_RANGES = {"age": (25, 44), "weight": (51, 86)}

CONFIRM_COLUMNS = False                   # change to True once the lines above are correct

print("ID:", ID_COL, "| time:", TIME_COL, "| conc:", CONC_COL)
print("Features:", PATIENT_FEATURES, "| units:", TIME_UNIT, CONC_UNIT)
if CONFIRM_COLUMNS:
    json.dump(dict(ID_COL=ID_COL, TIME_COL=TIME_COL, CONC_COL=CONC_COL,
                   PATIENT_FEATURES=PATIENT_FEATURES, TIME_UNIT=TIME_UNIT, CONC_UNIT=CONC_UNIT,
                   EXPECTED_RANGES=EXPECTED_RANGES), open(COLS_FILE, "w"), indent=2)
    print("Saved. Go to Part 4.")
else:
    print("NOT saved yet. Set CONFIRM_COLUMNS = True and run this part again.")

# %% PART 4: Build the dataset (reads each file in chunks, keeps only the 15-minute grid)
assert os.path.exists(COLS_FILE), "Run Part 3 with CONFIRM_COLUMNS = True first."
t0 = time.time()

# 4a. choose the individuals once, and use the same ones for every regimen
pat_tables = {}
for regimen, cfg in FILES.items():
    pp = os.path.join(BASE_DIR, cfg["patients"])
    pat = pd.read_csv(pp, sep=detect_sep(pp), encoding="utf-8-sig",
                      usecols=[ID_COL] + PATIENT_FEATURES)
    pat.columns = pat.columns.astype(str).str.strip()
    pat[ID_COL] = clean_ids(pat[ID_COL])
    if pat[ID_COL].duplicated().any():
        raise ValueError(f"{regimen}: duplicate individual IDs in the patient file.")
    pat_tables[regimen] = pat.set_index(ID_COL)

common = set.intersection(*[set(p.index) for p in pat_tables.values()])
if len(common) == 0:
    raise ValueError("No individual IDs are shared between the three patient files.")
common = sorted(common, key=lambda s: (len(s), s))
rng = np.random.default_rng(SEED)
selected = list(common) if (N_INDIVIDUALS is None or N_INDIVIDUALS >= len(common)) \
    else list(rng.choice(common, size=N_INDIVIDUALS, replace=False))
selected_set = set(selected)
print(f"Individuals in all files: {len(common):,} | using: {len(selected):,}")

# the same person must have the same covariates in every regimen file
ref_pat = pat_tables[list(FILES)[0]].loc[selected]
for regimen, pat in pat_tables.items():
    other = pat.loc[selected]
    for c in PATIENT_FEATURES:
        if pd.api.types.is_numeric_dtype(ref_pat[c]):
            if not np.allclose(ref_pat[c].to_numpy(float), other[c].to_numpy(float), rtol=1e-4, atol=1e-6):
                raise ValueError(f"{regimen}: covariate '{c}' differs from the first regimen. "
                                 "The three simulations must use the SAME population.")
        elif not (ref_pat[c].astype(str).to_numpy() == other[c].astype(str).to_numpy()).all():
            raise ValueError(f"{regimen}: covariate '{c}' differs from the first regimen.")

# 4b. encode text covariates (for example gender) as numbers
cov = ref_pat.copy()
encodings = {}
for c in PATIENT_FEATURES:
    if not pd.api.types.is_numeric_dtype(cov[c]):
        cats = sorted(cov[c].astype(str).unique())
        encodings[c] = {k: i for i, k in enumerate(cats)}
        cov[c] = cov[c].astype(str).map(encodings[c])
        print(f"Encoded '{c}':", encodings[c])
cov = cov.astype("float32")

# 4c. check covariate units against the expected population ranges
for key, (lo, hi) in EXPECTED_RANGES.items():
    for c in PATIENT_FEATURES:
        if key in c.lower():
            mn, mx = float(cov[c].min()), float(cov[c].max())
            ok = (mn >= lo * 0.97) and (mx <= hi * 1.03)
            print(f"{c}: min {mn:.1f}, max {mx:.1f}, expected {lo} to {hi} -> {'OK' if ok else 'CHECK UNITS'}")
            if not ok:
                raise ValueError(f"'{c}' is outside the expected range. Check units, or edit EXPECTED_RANGES.")

# 4d. read results in chunks, keep only grid times
expected_pts = int(round(END_H / GRID_STEP_H)) + 1
frames = []
for regimen, cfg in FILES.items():
    rp = os.path.join(BASE_DIR, cfg["results"])
    kept = []
    for ch in pd.read_csv(rp, sep=detect_sep(rp), usecols=[ID_COL, TIME_COL, CONC_COL],
                          chunksize=1_000_000, encoding="utf-8-sig"):
        ch.columns = ch.columns.astype(str).str.strip()
        ch[ID_COL] = clean_ids(ch[ID_COL])
        ch = ch[ch[ID_COL].isin(selected_set)]
        if ch.empty:
            continue
        t = to_hours(ch[TIME_COL].to_numpy(dtype="float64"), TIME_UNIT)
        k = t / GRID_STEP_H
        keep = (np.abs(k - np.round(k)) < 1e-4) & (t <= END_H + 1e-9) & (t >= -1e-9)
        ch = ch.loc[keep]
        out = pd.DataFrame({
            "id": ch[ID_COL].to_numpy(),
            "time_h": (np.round(k[keep]) * GRID_STEP_H).astype("float32"),
            "reference_mg_L": to_mg_per_l(ch[CONC_COL].to_numpy(dtype="float64"), CONC_UNIT).astype("float32"),
        })
        kept.append(out)
    d = pd.concat(kept, ignore_index=True)
    n_dup = int(d.duplicated(["id", "time_h"]).sum())
    if n_dup:
        print(f"{regimen}: {n_dup:,} duplicate (id, time) rows; keeping the last of each "
              "(this is normal if the export repeats the dosing time points).")
        d = d.drop_duplicates(["id", "time_h"], keep="last")
    per_id = d.groupby("id").size()
    if per_id.nunique() != 1 or int(per_id.iloc[0]) != expected_pts:
        raise ValueError(f"{regimen}: each individual should have {expected_pts} grid points but has "
                         f"{sorted(per_id.unique())[:5]}. Re-export with 4 output points per hour, "
                         f"a simulation length of {END_H:g} h, and the same time unit.")
    d["regimen"] = regimen
    d["dose_mg"] = np.float32(cfg["dose_mg"])
    d["tau_h"] = np.float32(cfg["tau_h"])
    d["time_after_dose_h"] = np.mod(d["time_h"].to_numpy(), cfg["tau_h"]).astype("float32")
    n_doses = int(round(END_H / cfg["tau_h"]))
    d["dose_number"] = np.minimum(np.floor(d["time_h"].to_numpy() / cfg["tau_h"]) + 1, n_doses).astype("float32")
    d = d.join(cov, on="id")
    d = d.rename(columns={c: c for c in PATIENT_FEATURES})
    frames.append(d)
    print(f"{regimen}: {len(d):,} rows, {d['id'].nunique():,} individuals")
    del kept

dataset = pd.concat(frames, ignore_index=True)
del frames
dataset.to_csv(P("data", "dataset.csv.gz"), index=False, compression="gzip")
json.dump(dict(selected_ids=selected, encodings=encodings, features_extra=PATIENT_FEATURES),
          open(P("data", "dataset_info.json"), "w"))
print(f"\nSaved dataset: {dataset.shape} in {time.time() - t0:.0f} s")
print(dataset.head())
del dataset

# %% PART 5: Data checks and descriptive statistics (Chapter 3, section 3.5)
dataset = pd.read_csv(P("data", "dataset.csv.gz"))
need = ["id", "regimen", "time_h", "reference_mg_L", "dose_mg", "tau_h"] + PATIENT_FEATURES
checks = {
    "missing_values": int(dataset[need].isna().sum().sum()),
    "duplicate_rows (id, regimen, time)": int(dataset.duplicated(["id", "regimen", "time_h"]).sum()),
    "non_finite_concentrations": int((~np.isfinite(dataset["reference_mg_L"])).sum()),
    "negative_concentrations": int((dataset["reference_mg_L"] < -1e-6).sum()),
    "time_outside_0_to_end": int(((dataset["time_h"] < 0) | (dataset["time_h"] > END_H)).sum()),
}
rows_per = dataset.groupby(["regimen", "id"]).size().groupby("regimen").nunique()
checks["regimens_with_unbalanced_rows"] = int((rows_per > 1).sum())
print("Integrity checks (all must be 0):")
for k, v in checks.items():
    print(f"  {k}: {v}")
assert all(v == 0 for v in checks.values()), "Fix the data problem above before continuing"
pd.Series(checks, name="count").to_csv(P("tables", "table_integrity_checks.csv"))

per_person = dataset.drop_duplicates("id")[PATIENT_FEATURES]
desc_cov = per_person.describe().T.round(2)
desc_conc = dataset.groupby("regimen")["reference_mg_L"].describe().round(2)
print("\nCovariates per individual:\n", desc_cov)
print("\nReference concentration (mg/L) by regimen:\n", desc_conc)
desc_cov.to_csv(P("tables", "table_descriptive_covariates.csv"))
desc_conc.to_csv(P("tables", "table_descriptive_concentrations.csv"))
del dataset, per_person

# %% PART 6: Split by individual, tune the random forest, train both models
import joblib
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv(P("data", "dataset.csv.gz"))
FEATURES = ["time_h", "time_after_dose_h", "dose_number", "dose_mg", "tau_h"] + PATIENT_FEATURES
TARGET = "reference_mg_L"

ids = np.array(sorted(dataset["id"].unique(), key=lambda s: (len(str(s)), str(s))))
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(ids))
n_test = int(round(TEST_FRACTION * len(ids)))
test_ids, train_ids = set(ids[perm[:n_test]]), set(ids[perm[n_test:]])
assert train_ids.isdisjoint(test_ids), "LEAKAGE: an individual is in both sets"
json.dump(dict(train_ids=sorted(map(str, train_ids)), test_ids=sorted(map(str, test_ids))),
          open(P("data", "split_ids.json"), "w"))

train = dataset[dataset["id"].isin(train_ids)]
test = dataset[dataset["id"].isin(test_ids)].copy()
del dataset
print(f"Train: {train['id'].nunique()} individuals ({len(train):,} rows) | "
      f"Test: {test['id'].nunique()} individuals ({len(test):,} rows)")
X_train, y_train = train[FEATURES].astype("float32"), train[TARGET]

lr = LinearRegression().fit(X_train, y_train)
print("Linear regression fitted.")

tune_ids = rng.choice(np.array(sorted(train_ids)), size=min(TUNE_N_IDS, len(train_ids)), replace=False)
tune = train[train["id"].isin(set(tune_ids))]
param_grid = {"max_depth": [10, 16, None], "min_samples_leaf": [30, 60, 120]}
t0 = time.time()
gs = GridSearchCV(RandomForestRegressor(n_estimators=TUNE_TREES, random_state=SEED, n_jobs=-1),
                  param_grid, cv=GroupKFold(n_splits=CV_FOLDS),
                  scoring="neg_root_mean_squared_error", n_jobs=1)
gs.fit(tune[FEATURES].astype("float32"), tune[TARGET], groups=tune["id"])
print(f"Tuning finished in {time.time() - t0:.0f} s. Best settings: {gs.best_params_}")
for pname, vals in param_grid.items():
    if gs.best_params_[pname] in (vals[0], vals[-1]):
        print(f"NOTE: best {pname} is at the edge of the grid {vals}. Mention this in the report.")
pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]].to_csv(
    P("tables", "table_rf_tuning.csv"), index=False)

t0 = time.time()
rf = RandomForestRegressor(n_estimators=FINAL_TREES, random_state=SEED, n_jobs=-1,
                           **gs.best_params_).fit(X_train, y_train)
print(f"Random forest fitted in {time.time() - t0:.0f} s.")

cv_rmse = {}
for name, model in {"Linear regression": LinearRegression(),
                    "Random forest": RandomForestRegressor(n_estimators=TUNE_TREES, random_state=SEED,
                                                           n_jobs=-1, **gs.best_params_)}.items():
    s = cross_val_score(model, tune[FEATURES].astype("float32"), tune[TARGET], groups=tune["id"],
                        cv=GroupKFold(n_splits=CV_FOLDS), scoring="neg_root_mean_squared_error")
    cv_rmse[name] = round(float(-s.mean()), 4)
print("Grouped CV RMSE on training data (mg/L):", cv_rmse)
pd.Series(cv_rmse, name="cv_rmse_mg_L").to_csv(P("tables", "table_cv_rmse.csv"))

test["pred_lr"] = lr.predict(test[FEATURES].astype("float32")).astype("float32")
test["pred_rf"] = rf.predict(test[FEATURES].astype("float32")).astype("float32")
test.to_csv(P("data", "test_predictions.csv.gz"), index=False, compression="gzip")
joblib.dump(lr, P("models", "linear_regression.joblib"))
joblib.dump(rf, P("models", "random_forest.joblib"), compress=3)
json.dump(dict(FEATURES=FEATURES, rf_best_params={k: v for k, v in gs.best_params_.items()},
               cv_rmse=cv_rmse), open(P("models", "model_info.json"), "w"), indent=2, default=str)
print("Saved models and test predictions.")
del train, tune, X_train, y_train

# %% PART 7: Concentration prediction results and figures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

test = pd.read_csv(P("data", "test_predictions.csv.gz"))
TARGET = "reference_mg_L"


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), "Bias (pred - ref)": float(np.mean(p - y)),
            "Negative predictions": int((p < 0).sum())}


perf = pd.DataFrame({"Linear regression": metrics(test[TARGET], test["pred_lr"]),
                     "Random forest": metrics(test[TARGET], test["pred_rf"])}).T.round(4)
print("Test-set concentration prediction (mg/L):\n", perf)
perf.to_csv(P("tables", "table_prediction_performance.csv"))

rows = []
for reg, g in test.groupby("regimen"):
    for m, col in [("Linear regression", "pred_lr"), ("Random forest", "pred_rf")]:
        d = metrics(g[TARGET], g[col]); d.update({"regimen": reg, "model": m}); rows.append(d)
pd.DataFrame(rows).round(4).to_csv(P("tables", "table_prediction_performance_by_regimen.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

samp = test.sample(min(8000, len(test)), random_state=SEED)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, col, ttl in [(ax[0], "pred_lr", "Linear regression"), (ax[1], "pred_rf", "Random forest")]:
    a.scatter(samp[TARGET], samp[col], s=4, alpha=0.3)
    lim = [min(0, samp[col].min()), max(samp[TARGET].max(), samp[col].max())]
    a.plot(lim, lim, "r--"); a.set(title=ttl, xlabel="Reference concentration (mg/L)",
                                  ylabel="Predicted concentration (mg/L)")
plt.tight_layout(); plt.savefig(P("figures", "fig1_pred_vs_ref.png"), dpi=150); plt.show(); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, col, ttl in [(ax[0], "pred_lr", "Linear regression"), (ax[1], "pred_rf", "Random forest")]:
    a.scatter(samp["time_h"], samp[col] - samp[TARGET], s=4, alpha=0.3)
    a.axhline(0, color="r", ls="--"); a.set(title=f"Residuals: {ttl}", xlabel="Time (h)",
                                          ylabel="Predicted minus reference (mg/L)")
plt.tight_layout(); plt.savefig(P("figures", "fig2_residuals.png"), dpi=150); plt.show(); plt.close()

reg0 = sorted(test["regimen"].unique())[0]
ex_ids = test["id"].drop_duplicates().sample(4, random_state=SEED).tolist()
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for a, i in zip(ax.ravel(), ex_ids):
    g = test[(test["id"] == i) & (test["regimen"] == reg0)].sort_values("time_h")
    a.plot(g["time_h"], g[TARGET], "k-", label="Reference")
    a.plot(g["time_h"], g["pred_lr"], "b--", label="Linear")
    a.plot(g["time_h"], g["pred_rf"], "g:", label="Random forest")
    a.set(title=f"Individual {i} ({reg0})", xlabel="Time (h)", ylabel="mg/L")
ax[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(P("figures", "fig3_example_profiles.png"), dpi=150); plt.show(); plt.close()

# %% PART 8: Target attainment (PD) from the reference and the predicted profiles
from sklearn.metrics import cohen_kappa_score

test = pd.read_csv(P("data", "test_predictions.csv.gz"))

# Free concentration factor
if CONCENTRATION_TYPE == "total":
    if FRACTION_UNBOUND is None or not (0 < FRACTION_UNBOUND <= 1):
        raise ValueError("Set FRACTION_UNBOUND in Part 1 (value from the PK-Sim Amoxicillin compound), "
                         "then run Part 1 and this part again.")
    FU = float(FRACTION_UNBOUND)
elif CONCENTRATION_TYPE == "free":
    FU = 1.0
else:
    raise ValueError("CONCENTRATION_TYPE must be 'total' or 'free'")

# The analysed interval is the LAST dosing interval of each regimen
PD_START = {r: END_H - cfg["tau_h"] for r, cfg in FILES.items()}
print("Analysed interval starts (h):", PD_START)


def interval_matrix(frame, col, regimen):
    cfg = FILES[regimen]
    s = frame[(frame["regimen"] == regimen) & (frame["time_h"] >= PD_START[regimen] - 1e-9)
              & (frame["time_h"] < END_H - 1e-9)].sort_values(["id", "time_h"])
    ids_ = s["id"].unique()
    return ids_, s[col].to_numpy(dtype="float64").reshape(len(ids_), -1)


def pct_above(conc, step, mic):
    """Percent of the interval above the MIC, using linear interpolation between grid points."""
    c0, c1 = conc[:, :-1], conc[:, 1:]
    a0, a1 = c0 > mic, c1 > mic
    both, cross = a0 & a1, a0 ^ a1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (mic - c0) / (c1 - c0)
    t = np.where(both, step, 0.0)
    t = t + np.where(cross & a0, step * frac, 0.0)
    t = t + np.where(cross & ~a0, step * (1 - frac), 0.0)
    return 100.0 * np.nansum(t, axis=1) / (step * (conc.shape[1] - 1))


rows, store = [], {}
for reg in FILES:
    ids_, ref = interval_matrix(test, "reference_mg_L", reg)
    mats = {"Linear regression": interval_matrix(test, "pred_lr", reg)[1],
            "Random forest": interval_matrix(test, "pred_rf", reg)[1]}
    for mic in MIC_GRID:
        a_ref = pct_above(FU * ref, GRID_STEP_H, mic) >= PD_TARGET_PCT
        for m, mat in mats.items():
            a_ml = pct_above(FU * mat, GRID_STEP_H, mic) >= PD_TARGET_PCT
            store[(reg, mic, m)] = (a_ref, a_ml)
            kappa = cohen_kappa_score(a_ref, a_ml) if len(np.unique(np.r_[a_ref, a_ml])) > 1 else np.nan
            rows.append({"regimen": reg, "MIC": mic, "model": m,
                         "PTA_reference_%": 100 * a_ref.mean(), "PTA_predicted_%": 100 * a_ml.mean(),
                         "PTA_difference_pp": 100 * (a_ml.mean() - a_ref.mean()),
                         "agreement_%": 100 * np.mean(a_ref == a_ml), "kappa": kappa,
                         "n_individuals": len(a_ref)})
pta = pd.DataFrame(rows).round(3)
pta.to_csv(P("tables", "table_pta_agreement.csv"), index=False)
print(f"PTA at the primary MIC ({PRIMARY_MIC} mg/L):\n",
      pta[pta["MIC"] == PRIMARY_MIC].to_string(index=False))

regs = list(FILES)
fig, ax = plt.subplots(1, len(regs), figsize=(5 * len(regs), 4.2), sharey=True)
ax = np.atleast_1d(ax)
for a, reg in zip(ax, regs):
    s = pta[pta["regimen"] == reg]
    q = s[s.model == "Linear regression"]
    a.plot(q["MIC"], q["PTA_reference_%"], "k-o", label="Reference")
    for m, st in [("Linear regression", "b--x"), ("Random forest", "g:s")]:
        q = s[s.model == m]; a.plot(q["MIC"], q["PTA_predicted_%"], st, label=m)
    a.set(xscale="log", title=reg, xlabel="MIC (mg/L)"); a.axhline(90, color="gray", ls=":")
ax[0].set_ylabel(f"PTA (%) for {PD_TARGET_PCT:.0f}% fT>MIC"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(P("figures", "fig4_pta_curves.png"), dpi=150); plt.show(); plt.close()

import pickle
pickle.dump(store, open(P("data", "pd_store.pkl"), "wb"))

# %% PART 9: Statistical tests for H01 and H02
from scipy.stats import wilcoxon, binomtest
import pickle

test = pd.read_csv(P("data", "test_predictions.csv.gz"))
store = pickle.load(open(P("data", "pd_store.pkl"), "rb"))
TARGET = "reference_mg_L"

# H01: linear regression versus random forest (paired by individual)
test["ae_lr"] = (test["pred_lr"] - test[TARGET]).abs()
test["ae_rf"] = (test["pred_rf"] - test[TARGET]).abs()
test["se_lr"] = (test["pred_lr"] - test[TARGET]) ** 2
test["se_rf"] = (test["pred_rf"] - test[TARGET]) ** 2
per_id = test.groupby("id").agg(mae_lr=("ae_lr", "mean"), mae_rf=("ae_rf", "mean"),
                                sse_lr=("se_lr", "sum"), sse_rf=("se_rf", "sum"), n=("ae_lr", "size"))
w = wilcoxon(per_id["mae_lr"], per_id["mae_rf"])
rng_b = np.random.default_rng(SEED)
idx = rng_b.integers(0, len(per_id), size=(N_BOOT, len(per_id)))
a, b, nn = per_id["sse_lr"].to_numpy(), per_id["sse_rf"].to_numpy(), per_id["n"].to_numpy()
d_rmse = np.sqrt(a[idx].sum(1) / nn[idx].sum(1)) - np.sqrt(b[idx].sum(1) / nn[idx].sum(1))
lo, hi = np.percentile(d_rmse, [2.5, 97.5])
h01 = pd.DataFrame([{
    "test": "Wilcoxon signed-rank on per-individual MAE (linear vs random forest)",
    "n_individuals": len(per_id), "median_MAE_linear": per_id["mae_lr"].median(),
    "median_MAE_rf": per_id["mae_rf"].median(), "p_value": w.pvalue,
    "reject_H0_at_alpha": bool(w.pvalue < ALPHA),
    "RMSE_difference_linear_minus_rf": float(np.sqrt(a.sum() / nn.sum()) - np.sqrt(b.sum() / nn.sum())),
    "bootstrap_95CI_low": lo, "bootstrap_95CI_high": hi}])
print("H01:\n", h01.T)
h01.to_csv(P("tables", "table_h01_test.csv"), index=False)

# H02: predicted versus reference target attainment (exact McNemar test, Holm correction)
res = []
for (reg, mic, m), (a_ref, a_ml) in store.items():
    if mic != PRIMARY_MIC:
        continue
    b01 = int(np.sum(a_ref & ~a_ml))
    c01 = int(np.sum(~a_ref & a_ml))
    p = binomtest(min(b01, c01), b01 + c01, 0.5).pvalue if (b01 + c01) > 0 else 1.0
    res.append({"regimen": reg, "model": m, "ref_yes_pred_no": b01, "ref_no_pred_yes": c01, "p_raw": p})
h02 = pd.DataFrame(res)
order = h02["p_raw"].argsort().to_numpy()
mt = len(h02)
adj, running = np.empty(mt), 0.0
for rank, i in enumerate(order):
    running = max(running, (mt - rank) * h02["p_raw"].iloc[i])
    adj[i] = min(1.0, running)
h02["p_holm"] = adj
h02["reject_H0_at_alpha"] = h02["p_holm"] < ALPHA
print("H02:\n", h02.to_string(index=False))
h02.to_csv(P("tables", "table_h02_test.csv"), index=False)

# %% PART 10: Save settings and bundle everything
import shutil
mi = json.load(open(P("models", "model_info.json")))
settings = dict(PD_TARGET_PCT=PD_TARGET_PCT, PRIMARY_MIC=PRIMARY_MIC, MIC_GRID=MIC_GRID,
                CONCENTRATION_TYPE=CONCENTRATION_TYPE, FRACTION_UNBOUND=FRACTION_UNBOUND,
                END_H=END_H, GRID_STEP_H=GRID_STEP_H, N_INDIVIDUALS=N_INDIVIDUALS,
                TEST_FRACTION=TEST_FRACTION, CV_FOLDS=CV_FOLDS, ALPHA=ALPHA, N_BOOT=N_BOOT,
                FILES=FILES, SEED=SEED, **mi)
json.dump(settings, open(P("settings_used.json"), "w"), indent=2, default=str)

imp = pd.DataFrame({"feature": mi["FEATURES"],
                    "importance": joblib.load(P("models", "random_forest.joblib")).feature_importances_})
imp.sort_values("importance", ascending=False).to_csv(P("tables", "rf_feature_importances.csv"), index=False)
print(imp.sort_values("importance", ascending=False).round(4).to_string(index=False))

shutil.make_archive(os.path.join(BASE_DIR, "results_bundle"), "zip", OUT_DIR)
print("Everything is saved in:", OUT_DIR, "\nA zip copy is in:", BASE_DIR)