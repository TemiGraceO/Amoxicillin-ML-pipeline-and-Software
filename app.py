"""Amoxicillin Stewardship Monitor (Flask).

The screens are exactly the same as the web page you approved. The random forest model and the
dose decision rules now run here in Python, and the page asks this server for each answer.
Patient records stay in the user's own browser, so a server restart does not erase them.

Folder layout:
    app.py, requirements.txt
    results/        <- the unzipped results bundle from the Colab notebook
Run locally:   python app.py            (then open http://localhost:5000)
Run on Render: gunicorn app:app
"""
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
HR = timedelta(hours=1)


def find_base():
    """Find the folder that holds settings_used.json (the unzipped results bundle)."""
    cands = [os.environ.get("AMX_RESULTS")]
    cands += [os.path.join(HERE, n) for n in ("results", "results_bundle", "Results", os.path.join("results", "results"))]
    cands += [HERE, os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "settings_used.json")):
            return c
    for root, dirs, files in os.walk(HERE):
        if "settings_used.json" in files:
            return root
        if root[len(HERE):].count(os.sep) >= 3:
            dirs[:] = []
    return None


BASE = find_base()


def P(*parts):
    return os.path.join(BASE or "results", *parts)


def pct_above(conc, step, mic):
    """Percent of the interval above the MIC, with linear interpolation between grid points."""
    c0, c1 = conc[:, :-1], conc[:, 1:]
    a0, a1 = c0 > mic, c1 > mic
    both, cross = a0 & a1, a0 ^ a1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (mic - c0) / (c1 - c0)
    t = np.where(both, step, 0.0)
    t = t + np.where(cross & a0, step * frac, 0.0)
    t = t + np.where(cross & ~a0, step * (1 - frac), 0.0)
    return 100.0 * np.nansum(t, axis=1) / (step * (conc.shape[1] - 1))


def js_round(x):
    return int(math.floor(x + 0.5))


def js_fixed(x, digits):
    """Same rounding as JavaScript toFixed (ties round up)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(x).quantize(q, rounding=ROUND_HALF_UP))


class Engine:
    def __init__(self):
        s = json.load(open(P("settings_used.json")))
        self.fu = float(s["FRACTION_UNBOUND"])
        self.target = float(s["PD_TARGET_PCT"])
        self.mics = [float(m) for m in s["MIC_GRID"]]
        self.end = float(s["END_H"])
        self.step = float(s["GRID_STEP_H"])
        self.reg = {k: (float(v["dose_mg"]), float(v["tau_h"])) for k, v in s["FILES"].items()}
        self.features = list(s["FEATURES"])
        self.rf = joblib.load(P("models", "random_forest.joblib"))
        self.rf.n_jobs = 1

        pf = self.features[5:]
        find = lambda *keys: next((f for f in pf if any(k in f.lower() for k in keys)), None)
        self.col = dict(age=find("age"), weight=find("weight"), height=find("height"),
                        gfr=find("gfr", "kidney"), sex=find("gender", "sex"))
        self.sexcode = {1: 1.0, 0: 0.0}
        try:
            info = json.load(open(P("data", "dataset_info.json")))
            enc = {str(k).upper(): v for k, v in info["encodings"].get(self.col["sex"], {}).items()}
            if "MALE" in enc and "FEMALE" in enc:
                self.sexcode = {1: float(enc["MALE"]), 0: float(enc["FEMALE"])}
        except Exception:
            pass
        self.res = self._residuals()

    # screen units to model units
    def _height(self, cm):
        n = (self.col["height"] or "").lower()
        return cm / 10.0 if "[dm]" in n else cm / 100.0 if "[m]" in n else cm

    def _gfr(self, ml_min):
        n = (self.col["gfr"] or "").lower()
        return ml_min / 1000.0 if "[l/min]" in n else ml_min

    def X(self, reg, t, p):
        dose, tau = self.reg[reg]
        nd = round(self.end / tau)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        d = {"time_h": t, "time_after_dose_h": np.mod(t, tau),
             "dose_number": np.minimum(np.floor(t / tau) + 1, nd), "dose_mg": dose, "tau_h": tau}
        vals = {"age": p["age"], "weight": p["wt"], "height": self._height(p["ht"]),
                "gfr": self._gfr(p["gfr"]), "sex": self.sexcode[int(p["sex"])]}
        for k, col in self.col.items():
            if col:
                d[col] = vals[k]
        return pd.DataFrame(d)[self.features].astype("float32")

    def curve(self, reg, p):
        t = np.arange(0.0, self.end + 1e-9, self.step)
        return t, np.clip(self.rf.predict(self.X(reg, t, p)), 0, None)

    def level_at(self, reg, p, e):
        tau = self.reg[reg][1]
        tt = self.end - tau + min(max(e, 0.0), tau - 0.001)
        return float(max(0.0, self.rf.predict(self.X(reg, [tt], p))[0]))

    def ft_mic(self, conc, reg, mic):
        tau = self.reg[reg][1]
        a = int(round((self.end - tau) / self.step))
        seg = self.fu * np.asarray(conc)[a:]
        return float(pct_above(seg[None, :], self.step, mic)[0])

    def _residuals(self):
        """Gap between the reference and the model's %fT>MIC on the held-out test individuals."""
        t = pd.read_csv(P("data", "test_predictions.csv.gz"))
        t["id"] = t["id"].astype(str)
        out = {}
        for reg, (dose, tau) in self.reg.items():
            s = t[(t["regimen"] == reg) & (t["time_h"] >= self.end - tau - 1e-9)].sort_values(["id", "time_h"])
            n = s["id"].nunique()
            ref = s["reference_mg_L"].to_numpy(float).reshape(n, -1) * self.fu
            prd = s["pred_rf"].to_numpy(float).reshape(n, -1) * self.fu
            for m in self.mics:
                d = pct_above(ref, self.step, m) - pct_above(prd, self.step, m)
                out[f"{reg}|{m:g}"] = [round(float(x), 1) for x in d]
        return out

    def assess(self, reg, ft, mic):
        r = self.res.get(f"{reg}|{mic:g}", [0.0])
        v = sorted(min(100.0, max(0.0, ft + e)) for e in r)
        pr = sum(1 for x in v if x >= self.target) / len(v)
        return pr, v[int(math.floor(len(v) * 0.1))], v[min(len(v) - 1, int(math.floor(len(v) * 0.9)))]

    def decide(self, p, now):
        reg, mic = p["reg"], p["mic"]
        tau = self.reg[reg][1]
        start = p["start"]
        end = start + timedelta(days=p["days"])
        ds = sorted(p["doses"])
        last = ds[-1] if ds else None
        t, c = self.curve(reg, p)
        ft = self.ft_mic(c, reg, mic)
        pr, lo, hi = self.assess(reg, ft, mic)
        o = dict(t=[float(x) for x in t], c=[float(x) for x in c], ft=ft, pr=pr, lo=lo, hi=hi, flags=[],
                 n=len(ds), tau=tau, mic=mic, e=None, next=None, now_level=None, lv=None, msg="",
                 start=start.isoformat(), end=end.isoformat())
        if pr < 0.6:
            o["flags"].append(["r", "Planned regimen is unlikely to reach the 40% fT>MIC target at this MIC ("
                                    f"{js_round(pr * 100)}% chance). Under exposure promotes resistance: ask the "
                                    "prescriber to review dose or interval."])
        if p["gfr"] < 60:
            o["flags"].append(["w", "Low kidney function entered: amoxicillin may accumulate. "
                                    "Confirm the dose with the prescriber."])
        high = False
        if p["levels"]:
            L = sorted(p["levels"], key=lambda x: x["t"])[-1]
            before = [d for d in ds if d <= L["t"]]
            if before:
                e = (L["t"] - before[-1]) / HR
                pe = self.level_at(reg, p, e)
                ratio = L["c"] / pe if pe > 0 else 1e9
                o["lv"] = dict(e=e, exp=pe, ratio=ratio, c=L["c"])
                if ratio > 1.5:
                    high = True
                    o["flags"].append(["r", f"Latest measured level ({L['c']:g} mg/L) is {js_fixed(ratio, 1)} times the "
                                            f"expected level ({js_fixed(pe, 1)}). Possible reduced clearance."])
                elif e >= tau * 0.75 and L["c"] * self.fu < mic:
                    o["flags"].append(["r", f"Latest measured trough ({L['c']:g} mg/L total) gives a free level "
                                            "below the MIC. Exposure is low."])
        if not last:
            if now < start:
                o.update(state="wait", title="Do not take yet: therapy starts {{start}}")
            else:
                o.update(state="go", title="First dose due",
                         msg="No dose has been logged for this patient. Log the dose when it is taken.")
            return o
        e = (now - last) / HR
        nxt = last + timedelta(hours=tau)
        o.update(e=e, next=nxt.isoformat(), now_level=self.level_at(reg, p, e))
        if now >= end:
            o.update(state="stop", title="Course complete: no further dose",
                     msg="The planned course ended {{end}}. Do not continue unless the prescriber extends it. "
                         "Unneeded extra doses add to resistance pressure.")
        elif high:
            o.update(state="hold", title="Hold and review before the next dose",
                     msg="Measured exposure is higher than expected. Contact the prescriber or pharmacist.")
        elif e < tau - 1:
            o.update(state="wait", title="Do not take another dose yet",
                     msg=f"Last dose was {js_fixed(e, 1)} h ago. Next dose is due about {{{{next}}}}. "
                         "Taking doses early raises exposure without added benefit.")
        elif e <= tau + 1:
            o.update(state="go", title="Next dose is due now",
                     msg=f"Last dose was {js_fixed(e, 1)} h ago, matching the {tau:g} hourly schedule.")
        elif e <= 2 * tau:
            o.update(state="go", title="Dose overdue: take it now",
                     msg=f"Last dose was {js_fixed(e, 1)} h ago. Take it as soon as possible and do not double the next "
                         "dose. Late doses let drug levels fall below the MIC.")
        else:
            o.update(state="hold", title="Several doses missed: contact the prescriber",
                     msg=f"Last dose was {js_fixed(e, 0)} h ago, more than two intervals.")
        return o


ENGINE = Engine() if BASE else None


def _num(v, name):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Enter a number for {name}.")
    if not math.isfinite(x):
        raise ValueError(f"Enter a number for {name}.")
    return x


def _time(v, name):
    try:
        return datetime.fromisoformat(str(v)[:19])
    except ValueError:
        raise ValueError(f"Enter the {name}.")


def decide_payload(data):
    """Validate what the page sent and return (json_dict, http_status)."""
    try:
        raw = data["p"]
        now = _time(data["now"], "current time")
        if raw.get("reg") not in ENGINE.reg:
            raise ValueError("Choose a regimen.")
        p = dict(reg=raw["reg"], age=_num(raw.get("age"), "age"), wt=_num(raw.get("wt"), "weight"),
                 ht=_num(raw.get("ht"), "height"), gfr=_num(raw.get("gfr"), "GFR"),
                 sex=int(_num(raw.get("sex"), "sex")), mic=_num(raw.get("mic"), "MIC"),
                 days=_num(raw.get("days"), "course length"), start=_time(raw.get("start"), "treatment start time"),
                 doses=[_time(d, "dose time") for d in raw.get("doses", [])],
                 levels=[{"t": _time(l["t"], "sample time"), "c": _num(l["c"], "lab result")}
                         for l in raw.get("levels", [])])
        return ENGINE.decide(p, now), 200
    except (ValueError, KeyError, TypeError) as err:
        msg = str(err) if isinstance(err, ValueError) else "Check the patient details."
        return {"error": msg}, 400


app = Flask(__name__)

NO_RESULTS = """<!DOCTYPE html><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results folder not found</title><body style="font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px;line-height:1.5">
<h2>The results folder was not found</h2><p>The app looked for <code>settings_used.json</code> inside <code>%s</code> and its sub folders.
Put the unzipped results bundle in a folder named <code>results</code> next to <code>app.py</code>, commit it to your Git
repository, and redeploy.</p><p>Needed inside <code>results</code>: <code>settings_used.json</code>,
<code>models/random_forest.joblib</code>, <code>data/dataset_info.json</code>, <code>data/test_predictions.csv.gz</code>.</p></body>"""


@app.get("/")
def index():
    if ENGINE is None:
        return Response(NO_RESULTS % HERE, status=500, mimetype="text/html")
    return Response(PAGE, mimetype="text/html")


@app.post("/api/decide")
def api_decide():
    if ENGINE is None:
        return jsonify({"error": "The results folder was not found on the server."}), 500
    body, status = decide_payload(request.get_json(silent=True) or {})
    return jsonify(body), status


@app.get("/healthz")
def healthz():
    return "ok"


PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Amoxicillin Stewardship Monitor</title>
<style>

:root{--bg:#f6f8f7;--card:#fff;--ink:#14231f;--mute:#5b6b66;--acc:#0f766e;--acc2:#b45309;--line:#dce5e1;--bad:#b91c1c;--good:#15803d;--warn:#a16207;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1715;--card:#15221f;--ink:#e8f1ee;--mute:#9bb0a9;--acc:#2dd4bf;--acc2:#fbbf24;--line:#264039;--bad:#f87171;--good:#4ade80;--warn:#facc15}}
:root[data-theme="dark"]{--bg:#0d1715;--card:#15221f;--ink:#e8f1ee;--mute:#9bb0a9;--acc:#2dd4bf;--acc2:#fbbf24;--line:#264039;--bad:#f87171;--good:#4ade80;--warn:#facc15}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:22px 5vw 8px;max-width:1200px;margin:auto}h1{font-size:clamp(24px,3.4vw,34px);margin:0}header p{color:var(--mute);margin:4px 0 0}
main{max-width:1200px;margin:auto;padding:12px 5vw 40px;display:grid;gap:16px;grid-template-columns:320px 1fr}
@media(max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}h2{font-size:15px;margin:0 0 10px;color:var(--acc);letter-spacing:.06em;text-transform:uppercase}
label{display:block;font-size:13px;color:var(--mute);margin:10px 0 3px}input,select{width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}button{cursor:pointer;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 12px;font:inherit}button.p{background:var(--acc);color:#fff;border-color:var(--acc);width:100%;margin-top:14px;padding:11px;font-weight:600}
.kpis{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:16px}.k b{display:block;font-size:30px;line-height:1.1;color:var(--acc)}.k span{color:var(--mute);font-size:13px}
.verdict{font-weight:700;font-size:19px}.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
svg{width:100%;height:auto;display:block}svg text{fill:var(--mute);font-size:11px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{color:var(--mute);font-weight:600}tr.sel td{font-weight:700;color:var(--acc)}
.alert{background:color-mix(in srgb,var(--warn) 15%,transparent);border-left:4px solid var(--warn);padding:8px 12px;border-radius:0 8px 8px 0;font-size:14px;margin-top:12px;display:none}
.mute{color:var(--mute);font-size:13px}.right{display:grid;gap:16px}

.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:6;display:flex;gap:10px;align-items:center;padding:10px 4vw;background:var(--bg);border-bottom:1px solid var(--line)}
.bar b{font-size:17px;white-space:nowrap}.bar select{max-width:260px;flex:1}.bar button{white-space:nowrap}
.pill{margin-left:auto;padding:6px 14px;border-radius:999px;font-weight:700;font-size:14px;border:2px solid var(--line);white-space:nowrap}
.pill.go{color:var(--good);border-color:var(--good)}.pill.wait{color:var(--warn);border-color:var(--warn)}.pill.stop,.pill.hold{color:var(--bad);border-color:var(--bad)}
main{max-width:1300px;margin:auto;padding:14px 4vw 40px;display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:16px;align-items:start}
.colL,.colR{display:grid;gap:14px;position:sticky;top:calc(var(--barh,64px) + 8px + env(safe-area-inset-top,0px));max-height:calc(100vh - var(--barh,64px) - 24px - env(safe-area-inset-top,0px));max-height:calc(100dvh - var(--barh,64px) - 24px - env(safe-area-inset-top,0px));overflow:auto;align-content:start;padding-bottom:4px}
.tabs{display:none}
.dec{display:flex;gap:16px;flex-wrap:wrap;border-radius:16px;padding:20px;border:2px solid var(--line);background:var(--card)}
.dec .ic{font-size:44px;line-height:1}.dec .tx{flex:1;min-width:220px}.dec h3{margin:0 0 6px;font-size:clamp(22px,2.6vw,30px);line-height:1.2}.dec p{margin:4px 0}
.dec.go{border-color:var(--good)}.dec.go h3{color:var(--good)}.dec.wait{border-color:var(--warn)}.dec.wait h3{color:var(--warn)}.dec.stop,.dec.hold{border-color:var(--bad)}.dec.stop h3,.dec.hold h3{color:var(--bad)}
.nx{font-size:18px;font-weight:600;margin-top:8px}
.flags{margin:10px 0 0;padding:0;list-style:none;width:100%}.flags li{padding:8px 12px;margin-top:6px;border-left:4px solid var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);border-radius:0 8px 8px 0;font-size:14px}.flags li.r{border-color:var(--bad);background:color-mix(in srgb,var(--bad) 12%,transparent)}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}.tile .v{font-size:32px;font-weight:700;line-height:1.1}.tile .l{color:var(--mute);font-size:13px;margin-bottom:4px}.tile .s{font-size:13px;color:var(--mute);margin-top:4px}
.meter{position:relative;height:12px;border-radius:6px;background:var(--line);margin:10px 0 4px}.meter i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:var(--acc)}.meter u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink)}
.g{color:var(--good)}.w{color:var(--warn)}.r{color:var(--bad)}
.lg{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin-top:6px}.lg i{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle;background:var(--acc)}.lg i.o{background:var(--acc2)}.lg i.d{background:none;border-top:2px dashed var(--acc);height:0}
.sub{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:14px 0 2px}.f2{display:grid;grid-template-columns:1fr 1fr;gap:8px 10px}.f2 label{margin-top:4px}
.row2{display:flex;gap:8px;align-items:end}.row2>div{flex:1}.x{padding:2px 8px;font-size:13px}.list td{padding:6px 4px}
@media(max-width:720px){main{grid-template-columns:1fr;padding-top:8px}.colL,.colR{position:static;max-height:none;overflow:visible}
.bar b{display:none}.tabs{display:flex;gap:6px;padding:8px 4vw 0;position:sticky;top:calc(var(--barh,56px) + env(safe-area-inset-top,0px));z-index:5;background:var(--bg)}.tabs button{flex:1;padding:9px 4px}.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.t-result,.t-details,.t-hist{display:none!important}main[data-tab=result] .t-result{display:grid!important}main[data-tab=details] .t-details{display:block!important}main[data-tab=hist] .t-hist{display:block!important}}
html{-webkit-text-size-adjust:100%}body{overflow-x:hidden}
.bar,.tabs,main{padding-left:max(4vw,env(safe-area-inset-left,0px));padding-right:max(4vw,env(safe-area-inset-right,0px))}
input,select,button{min-height:44px}.x{min-height:0}input,select{min-width:0;max-width:100%}input[type=datetime-local]{width:100%;-webkit-appearance:none;appearance:none;text-align:left}
.f2>*,.row2>*,.tiles>*,.dec .tx{min-width:0}.dec .tx{flex:1 1 0}.list{width:100%}.list td{white-space:normal;overflow-wrap:anywhere}.card{overflow-wrap:anywhere}
@media(max-width:720px){.bar{flex-wrap:wrap;gap:8px;padding-top:8px;padding-bottom:8px}.bar select{flex:1 1 55%;max-width:none}.pill{margin-left:0;flex:1 1 100%;text-align:center;font-size:15px;padding:8px}
.dec{padding:16px;gap:10px}.dec .ic{font-size:32px}.dec button.p{width:100%!important}.tile .v{font-size:28px}.card{padding:14px}.tiles{grid-template-columns:1fr}
.row2{flex-wrap:wrap}.row2>div,.row2>button{flex:1 1 100%}.tabs button{min-height:44px}h2{font-size:14px}}
@media(min-width:480px) and (max-width:720px){.tiles{grid-template-columns:1fr 1fr}}
@media(max-width:360px){.f2{grid-template-columns:1fr}.bar button{flex:1 1 100%}}
</style></head><body>
<div class="bar"><b>Amoxicillin Monitor</b><select id="pk" aria-label="Patient"></select><button id="addp">+ New patient</button><span class="pill" id="pill"></span></div>
<div class="tabs" id="tabs"><button data-t="result">Result</button><button data-t="details">Details</button><button data-t="hist">Labs and doses</button></div>
<main id="mn" data-tab="result">
<section class="colL"><div class="card t-details" id="formBox"></div><div class="card t-hist" id="labBox"></div><div class="card t-hist" id="histBox"></div></section>
<section class="colR t-result" id="resBox"></section>
</main><span id="stn" class="mute" style="display:block;text-align:center"></span>
<script>
const FU=0.82,END=72,STEP=0.25,TARGET=40;
const REG={'500mg_q8h':[500,8],'1000mg_q8h':[1000,8],'1000mg_q12h':[1000,12]},MICS=[0.125,0.25,0.5,1,2,4,8];
const $=i=>document.getElementById(i),n=i=>parseFloat($(i).value);

const HR=36e5,KEY='amx_stewardship_v1';let DB={patients:[],last:null},memOnly=false;
const REGN={'500mg_q8h':'500 mg every 8 h','1000mg_q8h':'1000 mg every 8 h','1000mg_q12h':'1000 mg every 12 h'};
const loc=d=>new Date(d-d.getTimezoneOffset()*6e4).toISOString().slice(0,16);
const fmt=d=>d.toLocaleString([],{weekday:'short',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function load(){try{const s=localStorage.getItem(KEY);if(s)DB=JSON.parse(s)}catch(e){memOnly=true}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(DB))}catch(e){memOnly=true}$('stn').textContent=memOnly?'Storage is blocked in this browser: records will be lost when you close the page.':''}
const cur=()=>DB.patients.find(p=>p.id==DB.last);
function addP(){const p={id:Date.now().toString(36),name:'New patient',age:34.5,sex:1,wt:68,ht:168,gfr:120,reg:'1000mg_q12h',mic:1,days:7,start:loc(new Date()),doses:[],levels:[],del:0};DB.patients.push(p);DB.last=p.id;save();draw()}
const pat=p=>({age:+p.age,wt:+p.wt,ht:+p.ht,gfr:+p.gfr,sex:+p.sex});

let SEQ=0,ut;const updD=()=>{clearTimeout(ut);ut=setTimeout(update,250)};
async function decide(p,now){const stamp=new Date(now-now.getTimezoneOffset()*6e4).toISOString().slice(0,19);
const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({p:p,now:stamp})}).catch(()=>{throw new Error('Could not reach the server. Check your internet connection.')});
const o=await r.json().catch(()=>({error:'The server sent an unexpected reply.'}));if(!r.ok)throw new Error(o.error||'Something went wrong.');
const tk=s=>s.replace('{{start}}',fmt(new Date(o.start))).replace('{{end}}',fmt(new Date(o.end))).replace('{{next}}',o.next?fmt(new Date(o.next)):'');
o.title=tk(o.title);o.msg=tk(o.msg);o.a={pr:o.pr,lo:o.lo,hi:o.hi};o.r={ts:o.t,cs:o.c,tau:o.tau};o.now=o.now_level;o.next=o.next?new Date(o.next):null;if(o.lv)o.lv.pr=o.lv.exp;return o}
let armed=false;
const when=ms=>{const m=Math.round(Math.abs(ms)/6e4),h=Math.floor(m/60);return(h?h+' h ':'')+(m%60)+' min'};
const ST={go:['✔','Take dose'],wait:['⏸','Wait'],stop:['⛔','Stop'],hold:['⚠','Hold']};
function picker(){$('pk').innerHTML=DB.patients.map(p=>`<option value="${p.id}" ${p.id==DB.last?'selected':''}>${esc(p.name||'Unnamed')}</option>`).join('')||'<option>No patients yet</option>'}
$('pk').onchange=()=>{DB.last=$('pk').value;armed=false;save();draw()};
function form(){const P=cur();if(!P){$('formBox').innerHTML='<h2>Details</h2><p class="mute">Add a patient to begin.</p>';return}
const f=(k,l,t,x='')=>`<div><label>${l}</label><input data-k="${k}" type="${t}" ${x} value="${esc(P[k])}"></div>`;
$('formBox').innerHTML=`<h2>Patient details</h2>${f('name','Name or ID','text')}<div class="f2">${f('age','Age (years)','number')}<div><label>Sex</label><select data-k="sex"><option value="1" ${P.sex==1?'selected':''}>Male</option><option value="0" ${P.sex==0?'selected':''}>Female</option></select></div>${f('wt','Weight (kg)','number')}${f('ht','Height (cm)','number')}</div>${f('gfr','Kidney function, GFR (mL/min)','number')}
<div class="sub">Treatment</div><label>Prescribed regimen</label><select data-k="reg">${Object.keys(REGN).map(k=>`<option value="${k}" ${P.reg==k?'selected':''}>${REGN[k]}</option>`).join('')}</select>
<div class="f2"><div><label>Germ MIC (mg/L)</label><select data-k="mic">${MICS.map(m=>`<option ${P.mic==m?'selected':''}>${m}</option>`).join('')}</select></div>${f('days','Course length (days)','number')}</div>${f('start','Treatment start','datetime-local')}
<button style="margin-top:14px" id="delp">Delete this patient</button>`;
document.querySelectorAll('#formBox [data-k]').forEach(el=>el.oninput=()=>{P[el.dataset.k]=el.value;save();if(el.dataset.k=='name')picker();updD()});
armed=false;$('delp').onclick=()=>{if(!armed){armed=true;$('delp').textContent='Click again to confirm';return}DB.patients=DB.patients.filter(p=>p.id!=P.id);DB.last=DB.patients.length?DB.patients[0].id:null;save();draw()}}
function lists(){const P=cur();if(!P){$('labBox').innerHTML='';$('histBox').innerHTML='';return}
const lr=P.levels.map((l,i)=>[l,i]).sort((a,b)=>new Date(b[0].t)-new Date(a[0].t)).map(([l,i])=>`<tr><td>${fmt(new Date(l.t))}</td><td>${l.c} mg/L</td><td><button class="x" onclick="rm('levels',${i})">Remove</button></td></tr>`).join('');
const dr=P.doses.map((d,i)=>[d,i]).sort((a,b)=>new Date(b[0])-new Date(a[0])).map(([d,i])=>`<tr><td>${fmt(new Date(d))}</td><td><button class="x" onclick="rm('doses',${i})">Remove</button></td></tr>`).join('');
$('labBox').innerHTML=`<h2>Lab result</h2><p class="mute" style="margin-top:0">If a lab has measured this patient's amoxicillin level, enter it here.</p><div class="row2"><div><label>Sample taken</label><input id="lt" type="datetime-local" value="${loc(new Date())}"></div><div><label>Level (mg/L)</label><input id="lc" type="number" step="0.1"></div></div><button class="p" onclick="addLevel()">Save lab result</button><table class="list">${lr}</table>`;
$('histBox').innerHTML=`<h2>Doses taken</h2><div class="row2"><div><label>Took a dose at a different time</label><input id="dt" type="datetime-local" value="${loc(new Date())}"></div><button onclick="logAt()">Add</button></div><table class="list">${dr||'<tr><td class="mute">No doses logged yet.</td></tr>'}</table>`}
function drawChart(o,W=700){const r=o.r,mic=o.mic,H=W<480?250:300,L=40,B=34,T=12,R=10,ymax=Math.max(...r.cs,mic*1.2)*1.1,X=t=>L+(W-L-R)*t/END,Y=v=>T+(H-T-B)*(1-v/ymax);
let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Amoxicillin level over time">`;
for(let i=0;i<=4;i++){const v=ymax*i/4;s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(v<10?1:0)}</text>`}
for(let t=0;t<=END;t+=12)s+=`<text x="${X(t)}" y="${H-16}" text-anchor="middle">${t}</text>`;
s+=`<rect x="${X(END-r.tau)}" y="${T}" width="${X(END)-X(END-r.tau)}" height="${H-T-B}" fill="var(--acc)" opacity=".1"/>`;
const pth=k=>r.ts.map((t,i)=>(i?'L':'M')+X(t).toFixed(1)+' '+Y(r.cs[i]*k).toFixed(1)).join('');
s+=`<path d="${pth(1)}" fill="none" stroke="var(--acc)" stroke-width="2"/><path d="${pth(FU)}" fill="none" stroke="var(--acc)" stroke-width="1.5" stroke-dasharray="5 4"/><line x1="${L}" x2="${W-R}" y1="${Y(mic)}" y2="${Y(mic)}" stroke="var(--acc2)" stroke-width="1.5"/>`;
if(o.now!=null){const t=END-o.tau+Math.min(o.e,o.tau-0.001);s+=`<circle cx="${X(t)}" cy="${Y(o.now)}" r="6" fill="var(--acc2)" stroke="var(--bg)" stroke-width="2"/><text x="${X(t)}" y="${Y(o.now)-11}" text-anchor="middle" style="fill:var(--ink);font-weight:700">Now</text>`}
s+=`<text x="${(L+W)/2}" y="${H-2}" text-anchor="middle">Hours</text><text transform="translate(11 ${H/2}) rotate(-90)" text-anchor="middle">mg/L</text></svg>`;return s}
let LO=null;function renderChart(){const c=$('ch');if(!c||!LO)return;c.innerHTML=drawChart(LO,Math.round(Math.min(760,Math.max(280,c.clientWidth||600))))}
function setBarH(){const b=document.querySelector('.bar');if(b)document.documentElement.style.setProperty('--barh',b.offsetHeight+'px')}
async function update(){const my=++SEQ;const P=cur(),pill=$('pill'),R=$('resBox');
if(!P){pill.textContent='';pill.className='pill';R.innerHTML='<div class="card" style="text-align:center;padding:40px"><h2>Welcome</h2><p>Add your first patient to check dosing and drug levels.</p><button class="p" style="width:auto" onclick="addP()">+ Add patient</button></div>';return}
let o;try{o=await decide(P,new Date())}catch(err){if(my===SEQ)R.innerHTML='<div class="card"><h2>Check the details</h2><p>'+esc(err.message)+'</p></div>';return}if(my!==SEQ)return;
const pr=Math.round(o.a.pr*100),ft=Math.round(o.ft),s=ST[o.state];
pill.className='pill '+o.state;pill.textContent=s[0]+' '+s[1];
const nx=o.next?(o.next>new Date()?`Next dose in ${when(o.next-new Date())} (${fmt(o.next)})`:`Dose was due ${when(new Date()-o.next)} ago`):'';
const word=pr>=90?['Very good','g']:pr>=60?['Good','g']:pr>=30?['Fair','w']:['Low','r'];
let lab='';if(o.lv){const q=o.lv.ratio,wd=q<.5?['lower than expected','r']:q>1.5?['higher than expected','r']:['as expected','g'];lab=`<div class="card"><div class="l mute">Lab check</div>Result of <b>${o.lv.c} mg/L</b>, taken ${o.lv.e.toFixed(1)} h after a dose. Expected about <b>${o.lv.pr.toFixed(1)} mg/L</b>. This is <b class="${wd[1]}">${wd[0]}</b>.</div>`}
R.innerHTML=`<div class="dec ${o.state}"><div class="ic">${s[0]}</div><div class="tx"><h3>${o.title}</h3><p>${o.msg}</p>${nx?`<div class="nx">${nx}</div>`:''}<button class="p" style="width:auto;padding:11px 22px" onclick="logNow()">I took a dose now</button></div>${o.flags.length?'<ul class="flags">'+o.flags.map(f=>`<li class="${f[0]}">${f[1]}</li>`).join('')+'</ul>':''}</div>
<div class="tiles"><div class="card tile"><div class="l">Drug in blood now</div><div class="v">${o.now!=null?o.now.toFixed(1)+' <small>mg/L</small>':'Not yet'}</div><div class="s">${o.now!=null?`Active (free) part ${(o.now*FU).toFixed(1)} mg/L, MIC ${o.mic}: <b class="${o.now*FU>=o.mic?'g':'r'}">${o.now*FU>=o.mic?'above, working':'below, not enough'}</b>`:'Log a dose to see this'}</div></div>
<div class="card tile"><div class="l">Time above MIC in each interval</div><div class="v">${ft}%</div><div class="meter"><i style="width:${Math.min(100,ft)}%;background:var(${ft>=TARGET?'--good':'--bad'})"></i><u style="left:${TARGET}%"></u></div><div class="s">Goal: at least ${TARGET}% (black mark)</div></div>
<div class="card tile"><div class="l">Chance this regimen works</div><div class="v ${word[1]}">${word[0]}</div><div class="s">${pr}% of similar patients reach the goal</div></div></div>${lab}
<div class="card"><h2>Drug level over time</h2><div id="ch"></div><div class="lg"><span><i></i>Amoxicillin in blood</span><span><i class="d"></i>Active (free) part</span><span><i class="o"></i>MIC: level needed to stop the germ</span><span style="color:var(--acc2)">&#9679; Now</span></div></div>`;LO=o;renderChart();setBarH()}
function draw(){picker();form();lists();update()}
function logNow(){cur().doses.push(loc(new Date()));save();lists();update()}
function logAt(){const v=$('dt').value;if(!v)return;cur().doses.push(v);save();lists();update()}
function addLevel(){const c=parseFloat($('lc').value);if(!(c>=0)||!$('lt').value)return;cur().levels.push({t:$('lt').value,c});save();lists();update()}
function rm(k,i){cur()[k].splice(i,1);save();lists();update()}
function setTab(t){$('mn').dataset.tab=t;if(t=='result')setTimeout(renderChart,0);document.querySelectorAll('#tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t==t))}
document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>setTab(b.dataset.t));$('pill').onclick=()=>setTab('result');
$('addp').onclick=()=>{addP();setTab('details')};load();if(!DB.last&&DB.patients.length)DB.last=DB.patients[0].id;setTab('result');draw();setInterval(update,60000);setBarH();let rt;const rs=()=>{clearTimeout(rt);rt=setTimeout(()=>{setBarH();renderChart()},150)};addEventListener('resize',rs);addEventListener('orientationchange',rs);if(document.fonts&&document.fonts.ready)document.fonts.ready.then(setBarH);
</script></body></html>
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))