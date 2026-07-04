"""
College Admissions Predictor
============================
Computes a student's composite academic score (SAT/ACT + GPA + Extracurriculars)
and evaluates Safety / Match / Reach fit against a list of schools, using
College Scorecard data (ADM_RATE, SAT percentiles).

Usage:
    python college_predictor.py --data college_scorecard_relevant_columns.xlsx \
        --sat 1200 --gpa 3.6 --ec 7.5 \
        --schools "Harvard University" "Yale University" "University of Florida"

    # Or use ACT instead of SAT:
    python college_predictor.py --data scorecard.xlsx --act 26 --gpa 3.6 --ec 7.5 \
        --schools "Stanford University"
"""

import argparse
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Weights for the composite score. SAT/ACT is the most standardized input,
# GPA next, extracurriculars least (see rationale: EC scoring is inherently
# subjective and has no external benchmark dataset).
# ---------------------------------------------------------------------------
WEIGHTS = {"sat": 0.45, "gpa": 0.35, "ec": 0.20}

BAND_COLORS = {"Safety": "#2E8B57", "Match": "#6A5ACD", "Reach": "#DC3545"}

# Official College Board ACT->SAT concordance (abridged; extend as needed).
ACT_TO_SAT = {
    36: 1590, 35: 1540, 34: 1500, 33: 1460, 32: 1430, 31: 1400, 30: 1370,
    29: 1340, 28: 1310, 27: 1280, 26: 1240, 25: 1210, 24: 1180, 23: 1140,
    22: 1110, 21: 1080, 20: 1040, 19: 1010, 18: 980, 17: 940, 16: 910,
    15: 880, 14: 830, 13: 780, 12: 730, 11: 680, 10: 630,
}


def act_to_sat(act_score: int) -> int:
    if act_score in ACT_TO_SAT:
        return ACT_TO_SAT[act_score]
    closest = min(ACT_TO_SAT, key=lambda k: abs(k - act_score))
    return ACT_TO_SAT[closest]


def normalize_sat(sat_score: float) -> float:
    return max(0.0, min(100.0, (sat_score - 400) / (1600 - 400) * 100))


def normalize_gpa(gpa: float) -> float:
    return max(0.0, min(100.0, (gpa / 4.0) * 100))


def normalize_ec(ec_score: float) -> float:
    return max(0.0, min(100.0, (ec_score / 10) * 100))


def composite_score(sat_score: float, gpa: float, ec_score: float) -> float:
    sat_n = normalize_sat(sat_score)
    gpa_n = normalize_gpa(gpa)
    ec_n = normalize_ec(ec_score)
    return round(
        sat_n * WEIGHTS["sat"] + gpa_n * WEIGHTS["gpa"] + ec_n * WEIGHTS["ec"], 1
    )


def sat_percentile_position(student_sat: float, satvr25, satvr75, satmt25, satmt75) -> float | None:
    """Where the student's SAT falls within the school's admitted-range band (0-100)."""
    try:
        low = float(satvr25) + float(satmt25)
        high = float(satvr75) + float(satmt75)
        if high <= low:
            return None
        pos = (student_sat - low) / (high - low) * 100
        return round(pos, 1)
    except (TypeError, ValueError):
        return None


def classify_school(adm_rate, sat_position) -> str:
    """Reach/Match/Safety classification."""
    if adm_rate is not None and not pd.isna(adm_rate) and adm_rate < 0.15:
        return "Reach"  # elite low-admit-rate schools stay Reach regardless of score
    if sat_position is None:
        return "Unknown (missing SAT data for this school)"
    if sat_position >= 75:
        return "Safety"
    elif sat_position >= 25:
        return "Match"
    else:
        return "Reach"


def load_scorecard(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["INSTNM", "ADM_RATE", "SATVR25", "SATVR75", "SATMT25", "SATMT75"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Error: input file is missing required columns: {missing}")
    return df


def evaluate_schools(df: pd.DataFrame, school_names: list[str], student_sat: float, comp_score: float):
    results = []
    for name in school_names:
        match = df[df["INSTNM"].str.lower() == name.lower()]
        if match.empty:
            match = df[df["INSTNM"].str.lower().str.contains(name.lower(), na=False)]
        if match.empty:
            results.append({"school": name, "status": "NOT FOUND in dataset"})
            continue

        row = match.iloc[0]
        sat_pos = sat_percentile_position(
            student_sat, row.get("SATVR25"), row.get("SATVR75"),
            row.get("SATMT25"), row.get("SATMT75"),
        )
        band = classify_school(row.get("ADM_RATE"), sat_pos)

        sat_avg = row.get("SAT_AVG")
        adm_rate = row.get("ADM_RATE")
        # chart coordinates: x = school's academic profile (SAT_AVG normalized),
        # y = selectivity (100 - admit rate), i.e. more selective = higher up
        chart_x = normalize_sat(sat_avg) if pd.notna(sat_avg) else None
        chart_y = (1 - adm_rate) * 100 if pd.notna(adm_rate) else None

        results.append({
            "school": row["INSTNM"],
            "adm_rate": adm_rate,
            "sat_avg_admitted": sat_avg,
            "student_sat_position_pct": sat_pos,
            "student_composite_score": comp_score,
            "band": band,
            "chart_x": chart_x,
            "chart_y": chart_y,
        })
    return results


def plot_chart(results: list[dict], comp_score: float, output_path: str = "college_fit_chart.png"):
    """Scatter plot: school academic profile (x) vs selectivity (y), student's
    composite score marked as a vertical reference line."""
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted_any = False
    for r in results:
        if "status" in r or r.get("chart_x") is None or r.get("chart_y") is None:
            continue
        plotted_any = True
        color = BAND_COLORS.get(r["band"], "#888888")
        ax.scatter(r["chart_x"], r["chart_y"], s=90, color=color, edgecolor="white", zorder=3)
        ax.annotate(
            r["school"], (r["chart_x"], r["chart_y"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8, color="#333333",
        )

    if not plotted_any:
        print("Warning: no schools had enough data to plot.")

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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")


def print_report(results: list[dict]):
    print("\n" + "=" * 70)
    print("COLLEGE FIT REPORT")
    print("=" * 70)
    for r in results:
        print(f"\n{r['school']}")
        if "status" in r:
            print(f"  {r['status']}")
            continue
        adm_rate = r["adm_rate"]
        adm_str = f"{adm_rate*100:.1f}%" if pd.notna(adm_rate) else "N/A"
        sat_avg = r["sat_avg_admitted"]
        sat_avg_str = f"{int(sat_avg)}" if pd.notna(sat_avg) else "N/A"
        pos = r["student_sat_position_pct"]
        pos_str = f"{pos}th percentile of admitted SAT range" if pos is not None else "N/A (missing SAT data)"

        print(f"  Admit rate:              {adm_str}")
        print(f"  School's avg admit SAT:  {sat_avg_str}")
        print(f"  Your SAT position:       {pos_str}")
        print(f"  Your composite score:    {r['student_composite_score']}/100")
        print(f"  >> Classification:       {r['band']}")
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="College admissions predictor")
    parser.add_argument("--data", required=True, help="Path to cleaned College Scorecard xlsx")
    parser.add_argument("--sat", type=int, help="SAT score (400-1600)")
    parser.add_argument("--act", type=int, help="ACT score (1-36), used if --sat not given")
    parser.add_argument("--gpa", type=float, required=True, help="Unweighted GPA (0.0-4.0)")
    parser.add_argument("--ec", type=float, required=True, help="Extracurricular score (1-10)")
    parser.add_argument("--schools", nargs="+", required=True, help="List of school names")
    parser.add_argument("--chart", default="college_fit_chart.png", help="Output path for scatter chart PNG")
    args = parser.parse_args()

    if not args.sat and not args.act:
        sys.exit("Error: provide either --sat or --act")

    student_sat = args.sat if args.sat else act_to_sat(args.act)

    comp_score = composite_score(student_sat, args.gpa, args.ec)
    print(f"\nStudent composite academic score: {comp_score}/100")
    if args.act and not args.sat:
        print(f"(ACT {args.act} converted to SAT-equivalent: {student_sat})")

    df = load_scorecard(args.data)
    results = evaluate_schools(df, args.schools, student_sat, comp_score)
    print_report(results)
    plot_chart(results, comp_score, args.chart)


if __name__ == "__main__":
    main()
