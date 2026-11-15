"""Amoxicillin Stewardship Monitor (Streamlit).

Folder layout expected next to this file:
    app.py, engine.py, requirements.txt
    results/            <- the unzipped results_bundle from the Colab notebook
Run with:  streamlit run app.py
"""
import html
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import BASE, HERE, Engine, P, load_db, new_patient, pct_above, save_db

if BASE is None:
    st.set_page_config(page_title="Amoxicillin Stewardship Monitor", page_icon="💊")
    st.error("The results folder was not found.")
    st.write(f"The app looked for `settings_used.json` inside `{HERE}` and its sub folders. "
             "Put the unzipped results bundle in a folder named `results` next to `app.py`, "
             "commit it to your Git repository, and redeploy.")
    st.write("Files needed inside `results`: `settings_used.json`, `models/random_forest.joblib`, "
             "`models/linear_regression.joblib`, `models/model_info.json`, `data/dataset_info.json`, "
             "`data/test_predictions.csv.gz`, the `tables` folder and the `figures` folder.")
    st.stop()

st.set_page_config(page_title="Amoxicillin Stewardship Monitor", page_icon="💊", layout="wide")

_V = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

st.markdown("""
<style>
.dec{border:2px solid #ccc;border-radius:16px;padding:18px 20px;margin-bottom:10px}
.dec .t{font-size:1.7rem;font-weight:700;line-height:1.2;margin-bottom:6px}
.dec .nx{font-size:1.1rem;font-weight:600;margin-top:8px}
div[data-testid="stMetricValue"]{font-size:1.9rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    return Engine()


@st.cache_data
def load_test():
    return pd.read_csv(P("data", "test_predictions.csv.gz"))


def table(name, **kw):
    path = P("tables", f"{name}.csv")
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


eng = get_engine()
if "db" not in st.session_state:
    st.session_state.db = load_db()
db = st.session_state.db

STATE = {"go": ("#15803d", "✔"), "wait": ("#a16207", "⏸"), "stop": ("#b91c1c", "⛔"), "hold": ("#b91c1c", "⚠")}
REGN = {k: f"{d:g} mg every {t:g} h" for k, (d, t) in eng.reg.items()}


def now_local():
    tz = st.session_state.get("tz", "Africa/Lagos")
    return datetime.now(ZoneInfo(tz)).replace(tzinfo=None, second=0, microsecond=0)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M")


def fmt(s):
    return datetime.fromisoformat(s).strftime("%a %d %b, %H:%M")


# ---------- callbacks ----------
def add_patient():
    p = new_patient(now_local())
    db["patients"].append(p)
    db["last"] = p["id"]
    st.session_state["sel"] = p["id"]


def log_dose(p, when=None):
    p["doses"].append(when or iso(now_local()))


def delete_patient(pid):
    db["patients"] = [p for p in db["patients"] if p["id"] != pid]
    db["last"] = db["patients"][0]["id"] if db["patients"] else None
    st.session_state.pop("sel", None)
    st.session_state["confirm_del"] = False


# ---------- sidebar: choose or add a patient ----------
with st.sidebar:
    st.header("Patients")
    ids = [p["id"] for p in db["patients"]]
    if ids:
        names = {p["id"]: st.session_state.get(f"{p['id']}_name", p["name"]) or "Unnamed" for p in db["patients"]}
        if st.session_state.get("sel") not in ids:
            st.session_state["sel"] = db["last"] if db["last"] in ids else ids[0]
        st.selectbox("Patient", ids, format_func=names.get, key="sel")
        db["last"] = st.session_state["sel"]
    st.button("+ New patient", on_click=add_patient)
    st.selectbox("Time zone", ["Africa/Lagos", "UTC"], key="tz")
    st.caption("The last patient used opens automatically next time.")


def chart(o):
    tau = o["tau"]
    fig = go.Figure()
    fig.add_vrect(x0=eng.end - tau, x1=eng.end, fillcolor="#0f766e", opacity=0.08, line_width=0)
    fig.add_scatter(x=o["t"], y=o["c"], name="Amoxicillin in blood", line=dict(color="#0f766e", width=2.5))
    fig.add_scatter(x=o["t"], y=o["c"] * eng.fu, name="Active (free) part",
                    line=dict(color="#0f766e", dash="dash", width=1.5))
    fig.add_hline(y=o["mic"], line_color="#b45309", annotation_text=f"MIC {o['mic']:g}",
                  annotation_position="top right")
    if o["now_level"] is not None:
        x = eng.end - tau + min(o["e"], tau - 0.001)
        fig.add_scatter(x=[x], y=[o["now_level"]], mode="markers+text", text=["Now"], textposition="top center",
                        marker=dict(size=13, color="#b45309"), name="Now")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Hours", yaxis_title="mg/L",
                      legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, **STRETCH)


def monitor():
    pid = db["last"]
    p = next((x for x in db["patients"] if x["id"] == pid), None)
    if p is None:
        st.info("Add your first patient from the sidebar to check dosing and drug levels.")
        st.button("+ Add patient", on_click=add_patient, key="add_first")
        return
    now = now_local()
    k = lambda name: f"{pid}_{name}"

    # results are placed first so that they appear on top on a phone
    col_res, col_det = st.columns([1.7, 1], gap="large")

    # ----- patient details (right column on a computer) -----
    with col_det:
        st.subheader("Patient details")
        p["name"] = st.text_input("Name or ID", p["name"], key=k("name"))
        a, b = st.columns(2)
        p["age"] = a.number_input("Age (years)", 0.0, 120.0, float(p["age"]), 1.0, key=k("age"))
        p["sex"] = b.radio("Sex", [1, 0], format_func={1: "Male", 0: "Female"}.get,
                           index=0 if int(p["sex"]) == 1 else 1, horizontal=True, key=k("sex"))
        a, b = st.columns(2)
        p["wt"] = a.number_input("Weight (kg)", 1.0, 300.0, float(p["wt"]), 0.5, key=k("wt"))
        p["ht"] = b.number_input("Height (cm)", 30.0, 250.0, float(p["ht"]), 1.0, key=k("ht"))
        p["gfr"] = st.number_input("Kidney function, GFR (mL/min)", 0.0, 300.0, float(p["gfr"]), 1.0, key=k("gfr"))

        st.subheader("Treatment")
        regs = list(eng.reg)
        p["reg"] = st.selectbox("Prescribed regimen", regs, format_func=REGN.get,
                                index=regs.index(p["reg"]) if p["reg"] in regs else 0, key=k("reg"))
        a, b = st.columns(2)
        cur_mic = float(p["mic"]) if float(p["mic"]) in eng.mics else eng.primary_mic
        p["mic"] = a.selectbox("Germ MIC (mg/L)", eng.mics, index=eng.mics.index(cur_mic), key=k("mic"))
        p["days"] = b.number_input("Course length (days)", 1, 90, int(p["days"]), 1, key=k("days"))
        sd = datetime.fromisoformat(p["start"])
        a, b = st.columns(2)
        d1 = a.date_input("Start date", sd.date(), key=k("sdate"))
        t1 = b.time_input("Start time", sd.time(), key=k("stime"))
        p["start"] = iso(datetime.combine(d1, t1))

        st.subheader("Doses taken")
        a, b = st.columns(2)
        d2 = a.date_input("Date", now.date(), key=k("ddate"))
        t2 = b.time_input("Time", now.time(), key=k("dtime"))
        st.button("Add dose at this time", on_click=log_dose, args=(p, iso(datetime.combine(d2, t2))),
                  key=k("adddose"))
        for i, d in sorted(enumerate(p["doses"]), key=lambda x: x[1], reverse=True):
            x, y = st.columns([3, 1])
            x.write(fmt(d))
            y.button("Remove", key=f"{pid}_rd{i}", on_click=lambda i=i: p["doses"].pop(i))

        st.subheader("Lab result")
        st.caption("If a lab has measured this patient's amoxicillin level, enter it here.")
        a, b = st.columns(2)
        d3 = a.date_input("Sample date", now.date(), key=k("ldate"))
        t3 = b.time_input("Sample time", now.time(), key=k("ltime"))
        c3 = st.number_input("Level (mg/L)", 0.0, 500.0, 0.0, 0.1, key=k("lc"))
        st.button("Save lab result", key=k("addlab"),
                  on_click=lambda: p["levels"].append({"t": iso(datetime.combine(d3, t3)),
                                                       "c": float(st.session_state[k("lc")])}))
        for i, l in sorted(enumerate(p["levels"]), key=lambda x: x[1]["t"], reverse=True):
            x, y = st.columns([3, 1])
            x.write(f"{fmt(l['t'])}: {l['c']:g} mg/L")
            y.button("Remove", key=f"{pid}_rl{i}", on_click=lambda i=i: p["levels"].pop(i))

        st.divider()
        if not st.session_state.get("confirm_del"):
            st.button("Delete this patient", key=k("del"),
                      on_click=lambda: st.session_state.update(confirm_del=True))
        else:
            st.warning("Delete this patient and all saved doses and lab results?")
            x, y = st.columns(2)
            x.button("Yes, delete", on_click=delete_patient, args=(pid,), key=k("yes"))
            y.button("Cancel", on_click=lambda: st.session_state.update(confirm_del=False), key=k("no"))

    # ----- result (left column on a computer, top on a phone) -----
    with col_res:
        o = eng.decide(p, now)
        color, icon = STATE[o["state"]]
        nxt = ""
        if o["next"]:
            delta = o["next"] - now
            mins = int(abs(delta.total_seconds()) // 60)
            span = (f"{mins // 60} h " if mins >= 60 else "") + f"{mins % 60} min"
            nxt = (f"Next dose in {span} ({o['next']:%a %H:%M})" if delta.total_seconds() > 0
                   else f"Dose was due {span} ago")
        st.markdown(f'<div class="dec" style="border-color:{color}"><div class="t" style="color:{color}">'
                    f'{icon} {html.escape(o["title"])}</div><div>{html.escape(o["msg"])}</div>'
                    f'<div class="nx">{html.escape(nxt)}</div></div>', unsafe_allow_html=True)
        st.button("I took a dose now", type="primary", on_click=log_dose, args=(p,), key=k("dosenow"))
        for lvl, text in o["flags"]:
            (st.error if lvl == "r" else st.warning)(text)

        m1, m2, m3 = st.columns(3)
        with m1:
            if o["now_level"] is None:
                st.metric("Drug in blood now", "Not yet")
                st.caption("Log a dose to see this")
            else:
                free = o["now_level"] * eng.fu
                st.metric("Drug in blood now", f"{o['now_level']:.1f} mg/L")
                st.caption(f"Active part {free:.1f} mg/L: " + ("above the MIC, working" if free >= o["mic"]
                                                               else "below the MIC, not enough"))
        with m2:
            st.metric("Time above MIC in each interval", f"{o['ft']:.0f}%")
            st.progress(min(max(o["ft"], 0.0), 100.0) / 100.0)
            st.caption(f"Goal: at least {eng.target:.0f}%")
        with m3:
            pr = o["pr"] * 100
            word = "Very good" if pr >= 90 else "Good" if pr >= 60 else "Fair" if pr >= 30 else "Low"
            st.metric("Chance this regimen works", word)
            st.caption(f"{pr:.0f}% of similar patients reach the goal")

        if o["lv"]:
            q = o["lv"]["ratio"]
            verdict = "lower than expected" if q < 0.5 else "higher than expected" if q > 1.5 else "as expected"
            st.info(f"Lab check: result of {o['lv']['c']:g} mg/L, taken {o['lv']['e']:.1f} h after a dose. "
                    f"Expected about {o['lv']['exp']:.1f} mg/L. This is {verdict}.")

        st.subheader("Drug level over time")
        chart(o)
        st.caption("Solid: amoxicillin in blood. Dashed: active (free) part. Orange line: MIC, the level needed "
                   "to stop the germ. Shaded: the interval used for the time above MIC.")


def study():
    test = load_test()
    sub = st.tabs(["Overview", "Test individuals", "Prediction performance", "Target attainment",
                   "Hypotheses", "Figures and data"])
    with sub[0]:
        s = eng.settings
        st.markdown(f"""
**Aim:** evaluate whether linear regression and random forest regression can reproduce amoxicillin
concentration and time profiles from a documented PK reference model, and whether the predicted profiles
give the same pharmacodynamic target attainment as the reference.

| Setting | Value |
|---|---|
| PD target | {eng.target:.0f}% fT>MIC |
| Unbound fraction | {eng.fu} |
| Primary MIC | {eng.primary_mic:g} mg/L |
| MIC grid | {", ".join(f"{m:g}" for m in eng.mics)} mg/L |
| Predictors | {", ".join(eng.features)} |
| Regimens | {", ".join(REGN.values())} |
| Random forest settings | {s.get("rf_best_params")} |
| Test individuals | {test["id"].nunique()} (split by individual) |
""")
    with sub[1]:
        c1, c2, c3 = st.columns(3)
        reg = c1.selectbox("Regimen", list(eng.reg), format_func=REGN.get, key="s_reg")
        pid = c2.selectbox("Test individual", sorted(test["id"].unique()), key="s_id")
        mic = c3.select_slider("MIC (mg/L)", eng.mics, value=eng.primary_mic, key="s_mic")
        g = test[(test["id"] == pid) & (test["regimen"] == reg)].sort_values("time_h")
        st.caption(" | ".join(f"{c.split('|')[-1]}: {g[c].iloc[0]:.4g}" for c in eng.patient_cols if c in g))
        fig = go.Figure()
        for col, name, style in [("reference_mg_L", "Reference", dict(color="black", width=3)),
                                 ("pred_rf", "Random forest", dict(color="green", dash="dot")),
                                 ("pred_lr", "Linear regression", dict(color="royalblue", dash="dash"))]:
            fig.add_scatter(x=g["time_h"], y=g[col], name=name, line=style)
        fig.add_hline(y=mic / eng.fu, line_dash="dash", line_color="red",
                      annotation_text=f"Total concentration equal to MIC ({mic / eng.fu:.2f} mg/L)")
        fig.update_layout(template="plotly_white", xaxis_title="Time (h)", yaxis_title="Total plasma concentration (mg/L)",
                          hovermode="x unified", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, **STRETCH)
        cols = st.columns(3)
        for c, (col, name) in zip(cols, [("reference_mg_L", "Reference"), ("pred_rf", "Random forest"),
                                         ("pred_lr", "Linear regression")]):
            tau = eng.reg[reg][1]
            last = g[(g["time_h"] >= eng.end - tau - 1e-9) & (g["time_h"] < eng.end - 1e-9)]
            ft = pct_above(eng.fu * last[col].to_numpy(float)[None, :], eng.step, mic)[0]
            c.metric(f"{name} %fT>MIC", f"{ft:.1f}%", "attained" if ft >= eng.target else "not attained",
                     delta_color="normal" if ft >= eng.target else "inverse")
    with sub[2]:
        st.subheader("Test-set concentration prediction (mg/L)")
        st.dataframe(table("table_prediction_performance", index_col=0), **STRETCH)
        st.subheader("By regimen")
        st.dataframe(table("table_prediction_performance_by_regimen"), hide_index=True, **STRETCH)
        a, b = st.columns(2)
        a.subheader("Grouped CV RMSE (training)")
        a.dataframe(table("table_cv_rmse", index_col=0), **STRETCH)
        b.subheader("Random forest tuning grid")
        b.dataframe(table("table_rf_tuning"), hide_index=True, **STRETCH)
        imp = table("rf_feature_importances")
        if imp is not None:
            st.subheader("Random forest feature importance")
            st.bar_chart(imp.set_index("feature")["importance"])
    with sub[3]:
        pta = table("table_pta_agreement")
        reg3 = st.selectbox("Regimen", list(eng.reg), format_func=REGN.get, key="s_reg3")
        s3 = pta[pta["regimen"] == reg3]
        ref = s3[s3["model"] == "Linear regression"]
        fig = go.Figure()
        fig.add_scatter(x=ref["MIC"], y=ref["PTA_reference_%"], name="Reference", mode="lines+markers",
                        line=dict(color="black", width=3))
        for m, colr, dash in [("Random forest", "green", "dot"), ("Linear regression", "royalblue", "dash")]:
            q = s3[s3["model"] == m]
            fig.add_scatter(x=q["MIC"], y=q["PTA_predicted_%"], name=m, mode="lines+markers",
                            line=dict(color=colr, dash=dash))
        fig.add_hline(y=90, line_dash="dot", line_color="gray")
        fig.update_layout(template="plotly_white", xaxis_type="log", xaxis_title="MIC (mg/L)",
                          yaxis_title=f"PTA (%) for {eng.target:.0f}% fT>MIC",
                          xaxis=dict(tickvals=eng.mics, ticktext=[f"{x:g}" for x in eng.mics]),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, **STRETCH)
        st.dataframe(s3, hide_index=True, **STRETCH)
    with sub[4]:
        st.subheader("H01: linear regression versus random forest")
        st.caption("Wilcoxon signed-rank test on per-individual MAE, with a bootstrap 95% CI for the RMSE difference.")
        h01 = table("table_h01_test")
        st.dataframe(h01.T.rename(columns={0: "value"}).astype(str), **STRETCH)
        st.subheader("H02: predicted versus reference target attainment")
        st.caption(f"Exact McNemar test at MIC {eng.primary_mic:g} mg/L per regimen and model, Holm-corrected.")
        st.dataframe(table("table_h02_test"), hide_index=True, **STRETCH)
    with sub[5]:
        figdir = P("figures")
        for f in (sorted(x for x in os.listdir(figdir) if x.endswith(".png")) if os.path.isdir(figdir) else []):
            st.image(os.path.join(figdir, f), caption=f, **STRETCH)
        st.subheader("Data integrity checks")
        st.dataframe(table("table_integrity_checks", index_col=0), **STRETCH)
        st.download_button("Download test predictions (CSV)", test.to_csv(index=False),
                           "test_predictions.csv", "text/csv")


tab_monitor, tab_study = st.tabs(["Monitor", "Study results"])
with tab_monitor:
    monitor()
with tab_study:
    study()

# save patient records whenever something changed
snapshot = json.dumps(db, sort_keys=True)
if snapshot != st.session_state.get("_saved"):
    save_db(db)
    st.session_state["_saved"] = snapshot