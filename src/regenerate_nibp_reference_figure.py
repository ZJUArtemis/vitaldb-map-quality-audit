#!/usr/bin/env python3
"""Regenerate the NIBP association figure with source-accurate wording."""

import json

import matplotlib.pyplot as plt

from project_paths import FIGURES_DIR, METRICS_DIR

OUTPUT = FIGURES_DIR / "fig9_nibp_state"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(
        (METRICS_DIR / "revision_nibp_state_association.json").read_text()
    )
    names = ["ART state", "Clinical", "Clinical +\nART state"]
    keys = ["state_only", "clinical_only", "clinical_plus_state"]
    means = [results[key]["mean"] for key in keys]
    lower = [results[key]["p2_5"] for key in keys]
    upper = [results[key]["p97_5"] for key in keys]
    errors = [
        [mean - low for mean, low in zip(means, lower)],
        [high - mean for mean, high in zip(means, upper)],
    ]

    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.errorbar(
        range(3), means, yerr=errors, fmt="o", capsize=5, color="#4472C4"
    )
    axis.axhline(0.5, linestyle="--", color="gray")
    axis.set_xticks(range(3), names)
    axis.set_ylim(0.45, 0.72)
    axis.set_ylabel("AUROC")
    axis.set_title("Exploratory association with the separately recorded NIBP-derived outcome")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT.with_suffix(".pdf"))
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
