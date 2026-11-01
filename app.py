import json, os
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Amoxicillin ML exposure & PTA", page_icon="💊", layout="wide")

ART, TAB, FIG = "artifacts", "outputs/tables", "outputs/figures"


# ---------- Loading (cached, nothing is trained) ----------
@st.cache_data
def load_meta():
    with open(f"{ART}/meta.json") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    return {
        "lr": joblib.load(f"{ART}/PLACEHOLDER_linear_regression.joblib"),
        "rf": joblib.load(f"{ART}/PLACEHOLDER_random_forest.joblib"),
    }

@st.cache_data
def load_test():
    return pd.read_csv(f"{ART}/test_predictions.csv")


@st.cache_data
def table(name, **kw):
    path = f"{TAB}/{PREFIX}{name}.csv"
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


meta = load_meta()
PREFIX, FU, TARGET = meta["out_prefix"], meta["fu"], meta["target"]
PD_TARGET, PRIMARY_MIC, MIC_GRID = meta["pd_target_pct"], meta["primary_mic"], meta["mic_grid"]
REGIMENS = meta["regimens"]
models = load_models()
test = load_test()


def pct_ft_above(conc, mic):
    """Same rule as notebook Cell 7: % of interval with FU*C > MIC."""
    return 100.0 * np.mean(FU * np.asarray(conc) > mic)


# ---------- Header ----------
st.title("Machine learning prediction of amoxicillin exposure and PD target attainment")
st.caption("B.Pharm research project, Nuhu Ibrahim Kankia (U19PH1070). "
           "Supervisor: Pharm Muhammad Sulaiman. Faculty of Pharmaceutical Sciences, ABU Zaria.")
if meta["use_placeholder"]:
    st.error("PLACEHOLDER DATA: results come from an illustrative model, not the published "
             "reference model. Do not report these numbers.")

tabs = st.tabs(["Overview", "Test individuals", "Try covariates",
                "Prediction performance", "Target attainment", "Hypotheses", "Figures & data"])

# ---------- Overview ----------
with tabs[0]:
    st.markdown(f"""
**Aim:** evaluate whether linear regression and random forest regression can reproduce amoxicillin
concentration–time profiles from a documented PK reference model, and whether the predicted profiles
give the same pharmacodynamic target attainment as the reference.

| Setting | Value |
|---|---|
| PD target | {PD_TARGET:.0f}% fT>MIC |
| Unbound fraction | {FU} |
| Primary MIC | {PRIMARY_MIC} mg/L |
| MIC grid | {", ".join(map(str, MIC_GRID))} mg/L |
| Predictors | {", ".join(meta["features"])} |
| Regimens | {", ".join(REGIMENS)} |
| Random forest settings | {meta["rf_best_params"]} |
| Test individuals | {test["id"].nunique()} (split by individual) |

This is a simulation study. It does not provide dosing recommendations and does not replace TDM.
""")

# ---------- Test individuals (stored reference vs stored predictions) ----------
with tabs[1]:
    st.write("Held-out individuals the models never saw. All curves are read from saved results.")
    c1, c2, c3 = st.columns(3)
    reg = c1.selectbox("Regimen", list(REGIMENS), key="r1")
    pid = c2.selectbox("Test individual", sorted(test["id"].unique()))
    mic = c3.select_slider("MIC (mg/L)", MIC_GRID, value=PRIMARY_MIC, key="m1")

    g = test[(test["id"] == pid) & (test["regimen"] == reg)].sort_values("time_h")
    st.caption(f"Weight {g.weight_kg.iloc[0]} kg · CrCl {g.crcl_ml_min.iloc[0]} mL/min")

    fig = go.Figure()
    for col, name, style in [(TARGET, "Reference", dict(color="black", width=3)),
                             ("pred_rf", "Random forest", dict(color="green", dash="dot")),
                             ("pred_lr", "Linear regression", dict(color="royalblue", dash="dash"))]:
        fig.add_scatter(x=g["time_h"], y=g[col], name=name, line=style)
    fig.add_hline(y=mic / FU, line_dash="dash", line_color="red",
                  annotation_text=f"Total conc. equivalent to MIC ({mic / FU:.2f} mg/L)")
    fig.update_layout(template="plotly_white", xaxis_title="Time after dose (h)",
                      yaxis_title="Total plasma concentration (mg/L)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for c, (col, name) in zip(cols, [(TARGET, "Reference"), ("pred_rf", "Random forest"),
                                     ("pred_lr", "Linear regression")]):
        ft = pct_ft_above(g[col], mic)
        c.metric(f"{name} %fT>MIC", f"{ft:.1f}%",
                 "attained" if ft >= PD_TARGET else "not attained",
                 delta_color="normal" if ft >= PD_TARGET else "inverse")

# ---------- Try covariates (prediction with saved models only) ----------
with tabs[2]:
    st.write("Predictions from the saved models for covariates you choose. No reference curve is "
             "shown because the reference model includes individual variability the predictors do not capture.")
    wlo, whi = meta["train_weight_range"]; clo, chi = meta["train_crcl_range"]
    c1, c2, c3, c4 = st.columns(4)
    w = c1.slider("Weight (kg)", float(wlo), float(whi), 70.0)
    crcl = c2.slider("CrCl (mL/min)", float(clo), float(chi), 100.0)
    reg2 = c3.selectbox("Regimen", list(REGIMENS), key="r2")
    mic2 = c4.select_slider("MIC (mg/L)", MIC_GRID, value=PRIMARY_MIC, key="m2")

    dose, tau = REGIMENS[reg2]
    t = np.arange(0.0, tau, meta["time_step_h"])
    X = pd.DataFrame({"time_h": t, "dose_mg": dose, "tau_h": tau,
                      "weight_kg": w, "crcl_ml_min": crcl})[meta["features"]]
    p_rf, p_lr = models["rf"].predict(X), models["lr"].predict(X)

    fig = go.Figure()
    fig.add_scatter(x=t, y=p_rf, name="Random forest", line=dict(color="green"))
    fig.add_scatter(x=t, y=p_lr, name="Linear regression", line=dict(color="royalblue", dash="dash"))
    fig.add_hline(y=mic2 / FU, line_dash="dash", line_color="red")
    fig.update_layout(template="plotly_white", xaxis_title="Time after dose (h)",
                      yaxis_title="Predicted total concentration (mg/L)")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Random forest %fT>MIC", f"{pct_ft_above(p_rf, mic2):.1f}%")
    b.metric("Linear regression %fT>MIC", f"{pct_ft_above(p_lr, mic2):.1f}%")
    st.caption("Sliders are limited to the covariate range seen in training to avoid extrapolation.")

# ---------- Prediction performance (notebook tables) ----------
with tabs[3]:
    st.subheader("Test-set concentration prediction (mg/L)")
    st.dataframe(table("table_prediction_performance", index_col=0), use_container_width=True)
    st.subheader("By regimen")
    st.dataframe(table("table_prediction_performance_by_regimen"), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("Grouped CV RMSE (training)")
    c1.dataframe(table("table_cv_rmse", index_col=0), use_container_width=True)
    c2.subheader("Random forest tuning grid")
    c2.dataframe(table("table_rf_tuning"), use_container_width=True, hide_index=True)

# ---------- Target attainment (notebook table) ----------
with tabs[4]:
    pta = table("table_pta_agreement")
    reg3 = st.selectbox("Regimen", list(REGIMENS), key="r3")
    s = pta[pta["regimen"] == reg3]
    ref = s[s["model"] == "Linear regression"]  # reference PTA is identical in both rows
    fig = go.Figure()
    fig.add_scatter(x=ref["MIC"], y=ref["PTA_reference_%"], name="Reference",
                    line=dict(color="black", width=3), mode="lines+markers")
    for m, colr, dash in [("Random forest", "green", "dot"), ("Linear regression", "royalblue", "dash")]:
        q = s[s["model"] == m]
        fig.add_scatter(x=q["MIC"], y=q["PTA_predicted_%"], name=m, mode="lines+markers",
                        line=dict(color=colr, dash=dash))
    fig.add_hline(y=90, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", xaxis_type="log", xaxis_title="MIC (mg/L)",
                      yaxis_title=f"PTA (%) for {PD_TARGET:.0f}% fT>MIC",
                      xaxis=dict(tickvals=MIC_GRID, ticktext=[str(x) for x in MIC_GRID]))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(s, use_container_width=True, hide_index=True)

# ---------- Hypotheses (notebook tables) ----------
with tabs[5]:
    st.subheader("H01: linear regression vs random forest")
    st.caption("Wilcoxon signed-rank test on per-individual MAE, with bootstrap 95% CI for RMSE difference.")
    st.dataframe(table("table_h01_test").T.rename(columns={0: "value"}), use_container_width=True)
    st.subheader("H02: predicted vs reference target attainment")
    st.caption(f"Exact McNemar test at MIC {PRIMARY_MIC} mg/L per regimen and model, Holm-corrected.")
    st.dataframe(table("table_h02_test"), use_container_width=True, hide_index=True)

# ---------- Figures & data ----------
with tabs[6]:
    figs = sorted(f for f in os.listdir(FIG) if f.endswith(".png")) if os.path.isdir(FIG) else []
    for f in figs:
        st.image(f"{FIG}/{f}", caption=f, use_container_width=True)
    st.subheader("Data integrity checks")
    st.dataframe(table("table_integrity_checks", index_col=0), use_container_width=True)
    st.download_button("Download test predictions (CSV)",
                       test.to_csv(index=False), "test_predictions.csv", "text/csv")

st.divider()
st.caption("Simulation-based research demonstration. Not for clinical use.")