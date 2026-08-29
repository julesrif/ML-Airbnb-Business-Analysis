# Visualisation suite

Run the analysis from the repository root after installing `requirements.txt`:

```powershell
.venv\Scripts\python.exe generate_visualisations.py
```

The script downloads the public Airbnb, NYC Neighbourhood Tabulation Area, and
MTA entrance datasets when the ignored `dataset/` cache is absent. It preserves
the source CSV and applies the notebook's cleaning rules: official-NTA spatial
join, zero-price removal, 99th-percentile price cap, and a 365-day maximum
minimum stay.

Each run removes the previously generated PNG files and creates 39 standalone
PNG visualisations. Every PNG contains exactly one chart or map:

- `01`–`04`: data completeness, cleaning retention, listings by borough, and
  listings by room type;
- `05`–`08`: price distribution, price dispersion, median segment prices, and
  segment listing volumes;
- `09`–`10`: neighbourhood listing density and median-price maps;
- `11`–`14`: price relationships with subway access, visitor destinations,
  availability, and review volume;
- `15`–`18`: minimum-stay, recent-review-rate, host-portfolio, and host-band
  visualisations;
- `19`–`22`: selected-model predictions and calibration, plus held-out
  error and explained-variance comparisons across three regression families
  and the median baseline;
- `23`–`26`: selected-model permutation importance and three residual/error
  diagnostics;
- `27`–`33`: K-means cluster sizes, average prices, standardised profiles,
  room-type and borough composition, PCA projection, and geographic
  distribution; and
- `34`–`37`: model error by cluster, neighbourhood price uncertainty, the
  highest-error NTAs, and actual-versus-predicted median price by cluster; and
- `38`: model MAE by borough and room type, with listing counts displayed in
  each cell; and
- `39`: training-only cross-validated RMSE used to select the winning model
  family.

The clustering reproduces the merged notebook's stable three-cluster
segmentation. It standardises price, booking behaviour, review activity, host
activity, availability, subway distance, and visitor-destination distance
before fitting K-means with seed 85.

The three provisional business names are Mainstream market (Cluster 0),
Premium professional (Cluster 1), and Lower-price review-active (Cluster 2).
They are interpretations of the fitted groups, not externally supplied labels.

The cold-start comparison tunes Decision Tree, Random Forest, and Gradient
Boosting models using three-fold GridSearchCV on the training split. The
computationally bounded grids contain 24 combinations drawn from the merged
notebook's candidate values. The model family with the lowest cross-validated
RMSE is refitted on all training rows and evaluated once on the held-out test
set. Review fields remain excluded because a new listing has no review history.

The benchmark is a training-only median lookup by NTA and room type, with a
borough/room-type fallback for unseen combinations. Listing-level predictions
used in the uncertainty views are five-fold out-of-fold estimates from the
selected model. Their errors represent unobserved variation and model
uncertainty; they are not evidence that a host has set an incorrect price.

The directory also contains `metrics_summary.json`, `cluster_summary.csv`,
`cluster_volatility_metrics.csv`, `nta_volatility_metrics.csv`, and
`segment_volatility_metrics.csv`, together with
`model_comparison_metrics.csv` and `grid_search_results.csv` for reuse in
the report.

Each run also creates `report_visualisations/`, a nine-figure curated pack on
a consistent 1,800 × 1,260-pixel canvas. The pack includes descriptive
filenames, report captions, alternative text, a machine-readable figure
catalogue, and a reproducibility manifest containing the Git revision, code and
dataset hashes, seeds, environment versions, and generation time. Selection and
usage guidance is documented in `REPORT_VISUALISATIONS.md`.
