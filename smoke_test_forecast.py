from __future__ import annotations

import numpy as np

from analyze_collapse_forecast import binary_metrics, first_collapse_index, fit_threshold


def main() -> None:
    collapse = first_collapse_index(np.array([0.20, 0.35, 0.30, 0.08]), 0.20, 0.10)
    assert collapse == 3

    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.2, 0.8, 1.0])
    auroc, auprc = binary_metrics(labels, scores)
    assert np.isclose(auroc, 1.0)
    assert np.isclose(auprc, 1.0)
    assert np.isclose(fit_threshold(labels, scores), 0.8)

    tied_auroc, tied_auprc = binary_metrics(labels, np.ones(4))
    assert np.isclose(tied_auroc, 0.5)
    assert np.isclose(tied_auprc, 0.5)
    print({"collapse_index": collapse, "perfect_auroc": auroc, "tied_auroc": tied_auroc})


if __name__ == "__main__":
    main()
