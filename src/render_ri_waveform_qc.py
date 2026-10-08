#!/usr/bin/env python3
"""Render deterministic waveform-level QC panels for corrected RI fiducials."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import vitaldb

from project_paths import RI_OUTPUT_DIR, VITAL_DIR

OUT = RI_OUTPUT_DIR
FS = 100


def select_cases(cases: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    valid = cases[cases["status"].eq("valid")].sort_values("ri_median_v14").copy()
    # Evenly sample the entire corrected RI distribution so low, central, and
    # high values all receive visual review.
    positions = np.linspace(0, len(valid) - 1, n).round().astype(int)
    return valid.iloc[positions].drop_duplicates("caseid").copy()


def case_panel(ax, case: pd.Series, beats: pd.DataFrame) -> dict:
    cid = int(case.caseid)
    group = beats[beats.caseid.eq(cid)].sort_values("on")
    # Display the densest run of six accepted beats within one continuous
    # segment.  This prevents a QC panel from visually joining accepted beats
    # that lie on opposite sides of a missing or rejected interval.
    best = None
    best_span = np.inf
    for _, segment in group.groupby("segment_index"):
        segment = segment.sort_values("on")
        if len(segment) <= 6:
            candidate = segment.copy()
            span = float(candidate["off"].max() - candidate["on"].min())
            if len(candidate) > 0 and (best is None or len(candidate) > len(best)):
                best, best_span = candidate, span
            continue
        for start in range(len(segment) - 5):
            candidate = segment.iloc[start:start + 6]
            span = float(candidate["off"].iloc[-1] - candidate["on"].iloc[0])
            if span < best_span:
                best, best_span = candidate.copy(), span
    shown = best.copy()
    path = VITAL_DIR / f"{cid:04d}.vital"
    vf = vitaldb.VitalFile(str(path))
    ppg = vf.to_numpy(["SNUADC/PLETH"], interval=1 / FS)[:, 0]
    lo = max(0, int(shown["on"].min()) - 20)
    hi = min(len(ppg), int(shown["off"].max()) + 20)
    time = (np.arange(lo, hi) - lo) / FS
    ax.plot(time, ppg[lo:hi], color="#333333", lw=0.9)
    styles = {
        "on": ("v", "#000000", "foot"),
        "sp": ("^", "#D55E00", "systolic"),
        "dn": ("x", "#E69F00", "notch"),
        "dp": ("o", "#0072B2", "diastolic"),
    }
    for key, (marker, colour, label) in styles.items():
        idx = shown[key].astype(int).to_numpy()
        ax.scatter((idx - lo) / FS, ppg[idx], marker=marker, color=colour,
                   s=28, zorder=3, label=label)
    ax.set_title(
        f"Case {cid}: RI={case.ri_median_v14:.3f}, valid beats={int(case.valid_ri_beats)}",
        fontsize=9,
    )
    ax.set_xlabel("Time within displayed segment (s)")
    ax.set_ylabel("Raw PPG (a.u.)")
    return {
        "caseid": cid,
        "ri_median_v14": float(case.ri_median_v14),
        "valid_ri_beats": int(case.valid_ri_beats),
        "displayed_beats": int(len(shown)),
        "strict_order_all_displayed": bool(
            ((shown["on"] < shown["sp"]) & (shown["sp"] < shown["dn"]) &
             (shown["dn"] < shown["dp"]) & (shown["dp"] < shown["off"])).all()
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(OUT / "ri_case_features_v14.parquet")
    beats = pd.read_parquet(OUT / "ri_beat_fiducials_v14.parquet")
    selected = select_cases(cases, 30)
    qc_rows = []
    pdf_path = OUT / "ri_waveform_qc_30cases_v14.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_number, start in enumerate(range(0, len(selected), 6), start=1):
            page = selected.iloc[start:start + 6]
            fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
            for ax, (_, case) in zip(axes.ravel(), page.iterrows()):
                qc_rows.append(case_panel(ax, case, beats))
            for ax in axes.ravel()[len(page):]:
                ax.axis("off")
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                ncol=4, frameon=False,
            )
            fig.suptitle(
                "v14 corrected Reflection Index waveform QC: same-pulse fiducials",
                y=0.995,
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.935))
            pdf.savefig(fig, dpi=200)
            fig.savefig(OUT / f"ri_waveform_qc_v14_page{page_number}.png", dpi=180)
            plt.close(fig)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT / "ri_waveform_qc_cases_v14.csv", index=False)
    summary = {
        "selection": "30 cases evenly spanning the corrected case-level RI distribution",
        "selected_cases_n": int(len(qc)),
        "displayed_beats_total": int(qc["displayed_beats"].sum()),
        "strict_temporal_order_cases_n": int(qc["strict_order_all_displayed"].sum()),
        "pdf": pdf_path.name,
    }
    (OUT / "ri_waveform_qc_summary_v14.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
