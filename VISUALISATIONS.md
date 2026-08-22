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

Each run removes the previously generated PNG files and creates 37 standalone
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
- `19`–`22`: held-out Random Forest predictions, calibration, error comparison,
  and explained variance against the median-price baseline;
- `23`–`26`: permutation importance and three residual/error diagnostics;
- `27`–`30`: pricing-anomaly prevalence, magnitude, distribution, and
  actual-price-band composition;
- `31`–`34`: neighbourhood hotspots and borough/room-type anomaly rates; and
- `35`–`37`: neighbourhood maps of overpriced rates, underpriced rates, and
  median price gaps.

Anomalies use five-fold out-of-fold predictions. A listing is classified as
overpriced when its observed price exceeds its estimated price by more than
50%, and underpriced when it falls more than 50% below its estimate. This avoids
the notebook's mixed in-sample/out-of-sample anomaly predictions.

The directory also contains `metrics_summary.json`,
`nta_opportunity_metrics.csv`, and `segment_opportunity_metrics.csv` for reuse
in the report.
