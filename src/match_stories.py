"""
match_stories.py
================
Select 4 real IPL matches (2 successful chases, 2 failed) and narrate
their pressure trajectories analytically.

Selection criteria:
  - Dramatic: pressure changes significantly across the innings
  - Sufficient balls: at least 80 deliveries bowled
  - Clean data: no missing target or outcome

For each match, we plot pressure + RRR + wickets over balls bowled,
annotating wicket falls and phase boundaries.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")


def _select_matches(df: pd.DataFrame, n_each: int = 2) -> dict:
    """
    Pick n_each won + n_each lost chases with the most 'dramatic' pressure swings
    (measured by std of pressure within the innings).
    """
    agg = df.groupby("match_id").agg(
        won          = ("won",        "first"),
        n_balls      = ("balls_bowled", "count"),
        pressure_std = ("pressure",   "std"),
        pressure_max = ("pressure",   "max"),
    ).reset_index()

    # Must have enough balls for an interesting story
    agg = agg[agg["n_balls"] >= 80]

    won_ids  = (agg[agg["won"] == 1]
                .sort_values("pressure_std", ascending=False)
                .head(n_each)["match_id"].tolist())
    lost_ids = (agg[agg["won"] == 0]
                .sort_values("pressure_std", ascending=False)
                .head(n_each)["match_id"].tolist())

    return {"won": won_ids, "lost": lost_ids}


def _narrative_from_pressure(mdf: pd.DataFrame) -> str:
    """
    Auto-generate a 3-sentence narrative from the pressure curve.
    """
    pp     = mdf["pressure"]
    target = mdf["target_score"].iloc[0]
    final  = mdf["current_score"].iloc[-1]
    wkts   = mdf["wickets_lost"].iloc[-1]
    won    = mdf["won"].iloc[0]

    # Inflection detection
    pp_early = pp[mdf["over"] < 6].mean()
    pp_mid   = pp[(mdf["over"] >= 6) & (mdf["over"] < 15)].mean()
    pp_death = pp[mdf["over"] >= 15].mean()

    # Identify highest pressure over
    max_over = mdf.loc[pp.idxmax(), "over"]

    result_str = f"Chase {'successful' if won else 'unsuccessful'}: " \
                 f"{final}/{wkts} chasing {target}."

    trend = (
        "Pressure was manageable early before spiking in the middle overs."
        if pp_mid > pp_early * 1.3 else
        "The batting side maintained healthy run-rate control throughout the powerplay."
        if pp_early < 1.5 else
        "High pressure from the outset put the chase in doubt early."
    )

    death_note = (
        f"A dramatic death-overs recovery (pressure dropped from {pp_mid:.1f} to {pp_death:.1f}) "
        f"sealed the win." if won and pp_death < pp_mid * 0.8 else
        f"Death-overs pressure surged to {pp_death:.1f} — the required rate became unachievable."
        if not won and pp_death > pp_mid * 1.1 else
        f"Pressure peaked around over {max_over} and the innings followed accordingly."
    )

    return f"{result_str} {trend} {death_note}"


def plot_match_stories(df: pd.DataFrame, matches: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    selected = _select_matches(df)
    all_ids  = [("Won",  mid) for mid in selected["won"]] + \
               [("Lost", mid) for mid in selected["lost"]]

    fig, axes = plt.subplots(len(all_ids), 1, figsize=(14, 5 * len(all_ids)))
    if len(all_ids) == 1:
        axes = [axes]

    narratives = []

    for ax, (outcome, match_id) in zip(axes, all_ids):
        mdf = df[df["match_id"] == match_id].sort_values("balls_bowled").copy()
        if mdf.empty:
            continue

        color    = "#2ECC71" if outcome == "Won" else "#E74C3C"
        target   = mdf["target_score"].iloc[0]
        batting  = mdf["batting_team"].iloc[0]
        bowling  = mdf["bowling_team"].iloc[0]

        # Get match date
        match_meta = matches[matches["id"] == match_id]
        date_str   = match_meta["date"].values[0] if not match_meta.empty else "Unknown"
        season     = match_meta["season"].values[0] if not match_meta.empty else ""

        # ── Primary axis: pressure ───────────────────────────────────────
        ax2 = ax.twinx()

        ax.fill_between(mdf["balls_bowled"], mdf["pressure"],
                        alpha=0.15, color=color)
        ax.plot(mdf["balls_bowled"], mdf["pressure"],
                color=color, linewidth=2.5, label="Pressure", zorder=3)
        ax.axhline(1.0, ls="--", color="grey", alpha=0.5, linewidth=1, label="Balanced (P=1)")

        # ── Secondary axis: required run rate ────────────────────────────
        rrr_clipped = mdf["required_run_rate"].clip(upper=25)
        ax2.plot(mdf["balls_bowled"], rrr_clipped,
                 color="#8E44AD", linewidth=1.5, alpha=0.7, ls=":", label="RRR")
        ax2.plot(mdf["balls_bowled"], mdf["current_run_rate"],
                 color="#F39C12", linewidth=1.5, alpha=0.7, ls="-.", label="CRR")
        ax2.set_ylabel("Run Rate", color="#8E44AD", fontsize=9)
        ax2.set_ylim(0, 26)
        ax2.tick_params(axis="y", labelcolor="#8E44AD")

        # ── Wicket markers ───────────────────────────────────────────────
        if "is_wicket" in mdf.columns:
            wkt_balls = mdf[mdf["is_wicket"] == 1]
            ax.scatter(wkt_balls["balls_bowled"], wkt_balls["pressure"],
                       color="black", s=60, zorder=5, marker="v",
                       label=f"Wicket fall ({len(wkt_balls)})")

        # ── Phase shading ────────────────────────────────────────────────
        for lo, hi, label, c in [
            (0, 36,  "PP",  "#AED6F1"),
            (36, 90, "Mid", "#FAD7A0"),
            (90, 120,"Death","#F1948A"),
        ]:
            ax.axvspan(lo, hi, alpha=0.07, color=c, zorder=0)
            ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                    label, ha="center", fontsize=8, color="grey")

        # ── Labels & title ───────────────────────────────────────────────
        ax.set_xlabel("Balls Bowled")
        ax.set_ylabel("Pressure", color=color)
        ax.set_xlim(0, mdf["balls_bowled"].max() + 2)
        ax.set_ylim(bottom=0)

        narrative = _narrative_from_pressure(mdf)
        narratives.append({"match_id": match_id, "outcome": outcome, "narrative": narrative})

        title = (f"Match {match_id} ({season}, {date_str})  |  "
                 f"{batting} vs {bowling}  |  Target: {target}  —  Chase {outcome.upper()}")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)

        # Legend
        handles_ax  = ax.get_legend_handles_labels()
        handles_ax2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles_ax[0] + handles_ax2[0],
            handles_ax[1] + handles_ax2[1],
            loc="upper left", fontsize=8, ncol=3,
        )

        # Narrative text box
        ax.text(0.02, 0.04, narrative, transform=ax.transAxes,
                fontsize=8.5, color="#2C3E50", verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    fig.suptitle("Match Story Analysis — Pressure Trajectories in IPL Chases\n"
                 "(▼ = wicket | shaded: Powerplay / Middle / Death overs)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(PLOTS_DIR, "23_match_stories.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [match_stories] Saved → {path}")

    # Save narratives
    pd.DataFrame(narratives).to_csv(
        os.path.join(OUTPUTS_DIR, "match_narratives.csv"), index=False
    )
    print(f"  [match_stories] Narratives saved → {OUTPUTS_DIR}/match_narratives.csv")
    for n in narratives:
        print(f"\n  Match {n['match_id']} [{n['outcome']}]: {n['narrative']}")
