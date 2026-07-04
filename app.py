"""
College Admissions Predictor - Web App
=======================================
Streamlit front-end for college_predictor.py. Designed for small-scale use
(a handful of users) - run locally or deploy free on Streamlit Community Cloud.

Run locally:
    pip install streamlit pandas matplotlib openpyxl
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from college_predictor import (
    act_to_sat, composite_score, sat_percentile_position,
    classify_school, normalize_sat, BAND_COLORS,
)

st.set_page_config(page_title="College Fit Predictor", layout="wide")
st.title("🎓 College Fit Predictor")
st.caption("Based on College Scorecard data. Composite score = 45% SAT/ACT + 35% GPA + 20% Extracurriculars.")

DATA_PATH = "Unidata.xlsx"  # bundle this file alongside app.py


@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    required = ["INSTNM", "ADM_RATE", "SATVR25", "SATVR75", "SATMT25", "SATMT75"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()
    return df


df = load_data(DATA_PATH)

with st.sidebar:
    st.header("Your academic information")
    score_type = st.radio("Test score type", ["SAT", "ACT"], horizontal=True)
    if score_type == "SAT":
        sat_input = st.number_input("SAT score", min_value=400, max_value=1600, value=1200, step=10)
        student_sat = sat_input
    else:
        act_input = st.number_input("ACT score", min_value=1, max_value=36, value=26, step=1)
        student_sat = act_to_sat(act_input)
        st.caption(f"Converted to SAT-equivalent: {student_sat}")

    gpa = st.number_input("Unweighted GPA (0.0-4.0)", min_value=0.0, max_value=4.0, value=3.6, step=0.05)
    ec = st.slider("Extracurricular score (1-10, self-assessed)", min_value=1.0, max_value=10.0, value=7.5, step=0.5)

    st.header("Colleges to check")
    school_choices = sorted(df["INSTNM"].dropna().unique().tolist())
    selected_schools = st.multiselect("Search and add colleges", school_choices, default=[])

    run = st.button("Calculate my chances", type="primary", use_container_width=True)


def generate_explanation(band, adm_rate, sat_pos, student_sat, low, high):
    """Plain-language reasoning behind the Reach/Match/Safety label."""
    if adm_rate is not None and pd.notna(adm_rate) and adm_rate < 0.15:
        base = (f"This school only admits {adm_rate*100:.1f}% of applicants, so it's "
                f"classified as a Reach regardless of your score — very few applicants "
                f"get in here even with top academics.")
        if low is not None and high is not None:
            if student_sat > high:
                base += " For reference, your SAT is still above their typical admitted range."
            elif student_sat < low:
                base += " Your SAT is also below their typical admitted range, adding to the gap."
        return base

    if sat_pos is None or low is None or high is None:
        return ("This school doesn't publicly report a full SAT range (often true for "
                "test-optional schools), so we can't precisely place your score here.")

    if student_sat < low:
        gap = round(low - student_sat)
        rate_note = f", and their admit rate is {adm_rate*100:.1f}%." if pd.notna(adm_rate) else "."
        return (f"Your SAT is about {gap} points below this school's 25th-percentile "
                f"admit score ({int(low)}){rate_note}")
    elif student_sat > high:
        gap = round(student_sat - high)
        return (f"Your SAT is about {gap} points above this school's 75th-percentile "
                f"admit score ({int(high)}), putting you above most admitted students.")
    else:
        return (f"Your SAT of {student_sat} falls within this school's typical admitted "
                f"range ({int(low)}–{int(high)}), at roughly the {round(sat_pos)}th "
                f"percentile of that range.")


def evaluate_schools(df, school_names, student_sat, comp_score):
    results = []
    for name in school_names:
        row = df[df["INSTNM"] == name].iloc[0]
        satvr25, satvr75 = row.get("SATVR25"), row.get("SATVR75")
        satmt25, satmt75 = row.get("SATMT25"), row.get("SATMT75")
        sat_pos = sat_percentile_position(student_sat, satvr25, satvr75, satmt25, satmt75)
        band = classify_school(row.get("ADM_RATE"), sat_pos)
        sat_avg = row.get("SAT_AVG")
        adm_rate = row.get("ADM_RATE")
        chart_x = normalize_sat(sat_avg) if pd.notna(sat_avg) else None
        chart_y = (1 - adm_rate) * 100 if pd.notna(adm_rate) else None

        low = high = None
        if pd.notna(satvr25) and pd.notna(satmt25) and pd.notna(satvr75) and pd.notna(satmt75):
            low = float(satvr25) + float(satmt25)
            high = float(satvr75) + float(satmt75)

        explanation = generate_explanation(band, adm_rate, sat_pos, student_sat, low, high)

        results.append({
            "school": row["INSTNM"], "adm_rate": adm_rate, "sat_avg_admitted": sat_avg,
            "student_sat_position_pct": sat_pos, "band": band,
            "chart_x": chart_x, "chart_y": chart_y, "explanation": explanation,
        })
    return results


def plot_chart(results, comp_score):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        if r.get("chart_x") is None or r.get("chart_y") is None:
            continue
        color = BAND_COLORS.get(r["band"], "#888888")
        ax.scatter(r["chart_x"], r["chart_y"], s=90, color=color, edgecolor="white", zorder=3)
        ax.annotate(r["school"], (r["chart_x"], r["chart_y"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="#333333")
    ax.axvline(comp_score, color="#6A5ACD", linestyle="--", linewidth=1.5, zorder=1)
    ax.text(comp_score, 102, "Your score", color="#6A5ACD", ha="center", fontsize=9, fontweight="bold")
    for band, color in BAND_COLORS.items():
        ax.scatter([], [], color=color, label=band)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("School's academic profile (avg admit SAT, normalized 0-100)")
    ax.set_ylabel("Selectivity (100 - admit rate)")
    ax.set_title("College Fit Chart")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


if run:
    if not selected_schools:
        st.warning("Add at least one college in the sidebar first.")
    else:
        comp_score = composite_score(student_sat, gpa, ec)

        col1, col2 = st.columns(2)
        col1.metric("Composite academic score", f"{comp_score}/100")

        results = evaluate_schools(df, selected_schools, student_sat, comp_score)

        st.subheader("College Fit Chart")
        fig = plot_chart(results, comp_score)
        st.pyplot(fig)

        st.subheader("Details")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r['school']}** — :{'green' if r['band']=='Safety' else 'violet' if r['band']=='Match' else 'red'}[{r['band']}]")
                c1, c2, c3 = st.columns(3)
                c1.write(f"Admit rate: {r['adm_rate']*100:.1f}%" if pd.notna(r['adm_rate']) else "Admit rate: N/A")
                c2.write(f"Avg admit SAT: {int(r['sat_avg_admitted'])}" if pd.notna(r['sat_avg_admitted']) else "Avg admit SAT: N/A")
                pos = r['student_sat_position_pct']
                c3.write(f"Your position: {pos} (relative to 25th-75th admitted range)" if pos is not None else "Your position: N/A")
                st.caption(f"💡 {r['explanation']}")
else:
    st.info("Fill in your info and pick a few colleges in the sidebar, then click **Calculate my chances**.")
