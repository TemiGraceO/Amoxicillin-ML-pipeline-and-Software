"""Engine for the Amoxicillin Stewardship Monitor.

Loads the trained models saved by the Colab notebook and turns them into
dose decisions for individual patients. No Streamlit code lives here, so this
file can be tested on its own.
"""
import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.environ.get("AMX_DB", os.path.join(HERE, "patients.json"))   # patient records are saved here
HR = timedelta(hours=1)


def find_base():
    """Find the folder that holds settings_used.json (the unzipped results bundle)."""
    cands = [os.environ.get("AMX_RESULTS")]
    cands += [os.path.join(HERE, n) for n in ("results", "results_bundle", "Results", os.path.join("results", "results"))]
    cands += [HERE, os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "settings_used.json")):
            return c
    for root, dirs, files in os.walk(HERE):          # look up to three levels down
        if "settings_used.json" in files:
            return root
        if root[len(HERE):].count(os.sep) >= 3:
            dirs[:] = []
    return None


BASE = find_base()


def P(*parts):
    return os.path.join(BASE or "results", *parts)


def pct_above(conc, step, mic):
    """Percent of the interval above the MIC (same rule as the notebook, Part 8)."""
    c0, c1 = conc[:, :-1], conc[:, 1:]
    a0, a1 = c0 > mic, c1 > mic
    both, cross = a0 & a1, a0 ^ a1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (mic - c0) / (c1 - c0)
    t = np.where(both, step, 0.0)
    t = t + np.where(cross & a0, step * frac, 0.0)
    t = t + np.where(cross & ~a0, step * (1 - frac), 0.0)
    return 100.0 * np.nansum(t, axis=1) / (step * (conc.shape[1] - 1))


class Engine:
    def __init__(self):
        s = json.load(open(P("settings_used.json")))
        self.fu = float(s["FRACTION_UNBOUND"])
        self.target = float(s["PD_TARGET_PCT"])
        self.mics = [float(m) for m in s["MIC_GRID"]]
        self.primary_mic = float(s["PRIMARY_MIC"])
        self.end = float(s["END_H"])
        self.step = float(s["GRID_STEP_H"])
        self.reg = {k: (float(v["dose_mg"]), float(v["tau_h"])) for k, v in s["FILES"].items()}
        self.features = list(s["FEATURES"])
        self.settings = s
        self.rf = joblib.load(P("models", "random_forest.joblib"))
        self.lr = joblib.load(P("models", "linear_regression.joblib"))

        # match the patient columns by name, so the exact PK-Sim column names do not matter
        pf = self.features[5:]
        def find(*keys):
            return next((f for f in pf if any(k in f.lower() for k in keys)), None)
        self.col = dict(age=find("age"), weight=find("weight"), height=find("height"),
                        gfr=find("gfr", "kidney"), sex=find("gender", "sex"))
        self.patient_cols = pf

        self.sexcode = {1: 1.0, 0: 0.0}
        try:
            info = json.load(open(P("data", "dataset_info.json")))
            enc = {str(k).upper(): v for k, v in info["encodings"].get(self.col["sex"], {}).items()}
            if "MALE" in enc and "FEMALE" in enc:
                self.sexcode = {1: float(enc["MALE"]), 0: float(enc["FEMALE"])}
        except Exception:
            pass
        self._res = None

    # ---------- unit conversion from the screen units to the model units ----------
    def _height(self, cm):
        n = (self.col["height"] or "").lower()
        return cm / 10.0 if "[dm]" in n else cm / 100.0 if "[m]" in n else cm

    def _gfr(self, ml_min):
        n = (self.col["gfr"] or "").lower()
        return ml_min / 1000.0 if "[l/min]" in n else ml_min

    # ---------- model input and predictions ----------
    def X(self, reg, t, p):
        dose, tau = self.reg[reg]
        nd = round(self.end / tau)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        d = {"time_h": t, "time_after_dose_h": np.mod(t, tau),
             "dose_number": np.minimum(np.floor(t / tau) + 1, nd),
             "dose_mg": dose, "tau_h": tau}
        vals = {"age": float(p["age"]), "weight": float(p["wt"]), "height": self._height(float(p["ht"])),
                "gfr": self._gfr(float(p["gfr"])), "sex": self.sexcode[int(p["sex"])]}
        for k, col in self.col.items():
            if col:
                d[col] = vals[k]
        return pd.DataFrame(d)[self.features].astype("float32")

    def curve(self, reg, p):
        t = np.arange(0.0, self.end + 1e-9, self.step)
        return t, np.clip(self.rf.predict(self.X(reg, t, p)), 0, None)

    def level_at(self, reg, p, e):
        """Estimated total concentration (mg/L) e hours after a dose, at steady state."""
        tau = self.reg[reg][1]
        tt = self.end - tau + min(max(e, 0.0), tau - 0.001)
        return float(max(0.0, self.rf.predict(self.X(reg, [tt], p))[0]))

    def ft_mic(self, conc, reg, mic):
        """%fT>MIC over the last dosing interval (notebook definition)."""
        tau = self.reg[reg][1]
        a = int(round((self.end - tau) / self.step))
        seg = self.fu * np.asarray(conc)[a:-1]
        return float(pct_above(seg[None, :], self.step, mic)[0])

    # ---------- chance of reaching the target, calibrated on the held-out test individuals ----------
    def residuals(self):
        if self._res is None:
            t = pd.read_csv(P("data", "test_predictions.csv.gz"))
            t["id"] = t["id"].astype(str)
            out = {}
            for reg, (dose, tau) in self.reg.items():
                s = t[(t["regimen"] == reg) & (t["time_h"] >= self.end - tau - 1e-9)
                      & (t["time_h"] < self.end - 1e-9)].sort_values(["id", "time_h"])
                n = s["id"].nunique()
                ref = s["reference_mg_L"].to_numpy(float).reshape(n, -1) * self.fu
                prd = s["pred_rf"].to_numpy(float).reshape(n, -1) * self.fu
                for m in self.mics:
                    out[(reg, m)] = pct_above(ref, self.step, m) - pct_above(prd, self.step, m)
            self._res = out
        return self._res

    def assess(self, reg, ft, mic):
        v = np.clip(ft + self.residuals()[(reg, float(mic))], 0, 100)
        return float(np.mean(v >= self.target)), float(np.percentile(v, 10)), float(np.percentile(v, 90))

    # ---------- the decision ----------
    def decide(self, p, now):
        reg, mic = p["reg"], float(p["mic"])
        tau = self.reg[reg][1]
        start = datetime.fromisoformat(p["start"])
        end = start + timedelta(days=float(p["days"]))
        doses = sorted(datetime.fromisoformat(d) for d in p["doses"])
        last = doses[-1] if doses else None
        t, c = self.curve(reg, p)
        ft = self.ft_mic(c, reg, mic)
        pr, lo, hi = self.assess(reg, ft, mic)
        o = dict(t=t, c=c, ft=ft, pr=pr, lo=lo, hi=hi, flags=[], n=len(doses), tau=tau, mic=mic,
                 e=None, next=None, now_level=None, lv=None, msg="")

        if pr < 0.6:
            o["flags"].append(("r", f"This regimen is unlikely to reach the {self.target:.0f}% fT>MIC goal at this MIC "
                                    f"({pr * 100:.0f}% chance). Under exposure promotes resistance: ask the prescriber "
                                    "to review the dose or interval."))
        if float(p["gfr"]) < 60:
            o["flags"].append(("w", "Low kidney function entered: amoxicillin may accumulate. "
                                    "Confirm the dose with the prescriber."))

        high = False
        levels = sorted(({"t": datetime.fromisoformat(l["t"]), "c": float(l["c"])} for l in p["levels"]),
                        key=lambda x: x["t"])
        if levels:
            L = levels[-1]
            before = [d for d in doses if d <= L["t"]]
            if before:
                e = (L["t"] - before[-1]) / HR
                exp = self.level_at(reg, p, e)
                ratio = L["c"] / exp if exp > 0 else float("inf")
                o["lv"] = dict(e=e, c=L["c"], exp=exp, ratio=ratio)
                if ratio > 1.5:
                    high = True
                    o["flags"].append(("r", f"The latest lab result ({L['c']:g} mg/L) is {ratio:.1f} times the expected "
                                            f"level ({exp:.1f} mg/L). Possible reduced clearance."))
                elif e >= tau * 0.75 and L["c"] * self.fu < mic:
                    o["flags"].append(("r", f"The latest lab trough ({L['c']:g} mg/L) gives an active (free) level "
                                            "below the MIC. Exposure is low."))

        if not last:
            if now < start:
                o.update(state="wait", title=f"Do not take yet: treatment starts {start:%a %d %b %H:%M}")
            else:
                o.update(state="go", title="First dose due", msg="No dose has been logged for this patient yet.")
            return o

        e = (now - last) / HR
        nxt = last + timedelta(hours=tau)
        o.update(e=e, next=nxt, now_level=self.level_at(reg, p, e))
        if now >= end:
            o.update(state="stop", title="Course complete: no further dose",
                     msg=f"The planned course ended {end:%a %d %b %H:%M}. Do not continue unless the prescriber "
                         "extends it. Unneeded extra doses add to resistance pressure.")
        elif high:
            o.update(state="hold", title="Hold and review before the next dose",
                     msg="Measured exposure is higher than expected. Contact the prescriber or pharmacist.")
        elif e < tau - 1:
            o.update(state="wait", title="Do not take another dose yet",
                     msg=f"The last dose was {e:.1f} h ago. Taking doses early raises exposure without added benefit.")
        elif e <= tau + 1:
            o.update(state="go", title="Next dose is due now",
                     msg=f"The last dose was {e:.1f} h ago, which matches the {tau:g} hourly schedule.")
        elif e <= 2 * tau:
            o.update(state="go", title="Dose overdue: take it now",
                     msg=f"The last dose was {e:.1f} h ago. Take it as soon as possible and do not double the next "
                         "dose. Late doses let drug levels fall below the MIC.")
        else:
            o.update(state="hold", title="Several doses missed: contact the prescriber",
                     msg=f"The last dose was {e:.0f} h ago, more than two intervals.")
        return o


# ---------- patient records (saved in a JSON file next to the app) ----------
def load_db():
    try:
        return json.load(open(DB_FILE))
    except Exception:
        return {"patients": [], "last": None}


def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=1)


def new_patient(now):
    return dict(id=now.strftime("%Y%m%d%H%M%S%f"), name="New patient", age=34.5, sex=1, wt=68.0, ht=168.0,
                gfr=120.0, reg="1000mg_q12h", mic=1.0, days=7, start=now.strftime("%Y-%m-%dT%H:%M"),
                doses=[], levels=[])