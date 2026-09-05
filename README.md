# NYC Airbnb Business Analysis — Track 1: Classical ML

**Module:** ML_PCOM7E — Development Team Project

**University of Essex Online**

**Team:** Team Lorem Ipsum (Group 2)

A spatial pricing and market-segmentation analysis of NYC Airbnb listings, built for **Track 1 (Classical ML: Regression & Clustering)** of the Development Team Project. The project develops a cold-start Automated Valuation Model (AVM) that prices new listings from multiple features, benchmarks it against a non-ML baseline, and uses K-Means clustering to identify the market segments where that pricing is least reliable.

---

## Purpose

This is the practical development deliverable supporting the team's 1,000-word analytical report (Task 3 of the assignment brief), due Unit 6. It implements Task 1 (business question) and Task 2 (Track 1 data analysis: regression for price prediction, clustering for customer segmentation) of the brief, producing the visualisations and metrics the report draws on.

---

## Business Question

> How can we develop an Automated Valuation Model (AVM) to provide a cold-start asking price for new Airbnb listings based on their features, and which geographic segments present the highest pricing volatility?

"Cold-start" here means the model only uses information available before a listing has any booking history: location, room type, and host-set fields such as minimum nights.

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **AB_NYC_2019** | Kaggle, `dgomonov/new-york-city-airbnb-open-data` | Core listing-level data: location, price, room type, host activity, availability |
| **2020 Neighborhood Tabulation Areas (NTAs)** | NYC Open Data, `data.cityofnewyork.us` (dataset id `9nt8-h7nd`) | Neighbourhood and borough boundaries, used to assign each listing a named geographic segment via spatial join |
| **MTA Subway Entrances and Exits** | NY State Open Data, `data.ny.gov` (dataset id `i9wp-a4ja`) | Nearest-subway-distance feature |
| **Points of interest (curated)** | Hand-picked list of 12 major NYC tourist landmarks and transit hubs (coordinates hardcoded in the notebook) | Nearest-POI-distance feature |

The listing data is enriched by spatial join with the NTA boundaries, then joined against the two proximity datasets by nearest-neighbour search (haversine `BallTree`), not by key, since none of the sources share a common identifier.

---

## Pipeline Overview

The notebook (`main.ipynb`) runs top to bottom as a single reproducible pipeline:

### 1. Data Acquisition
Downloads AB_NYC_2019 from Kaggle (cached locally after first run).

### 2. Geospatial Enrichment
Spatial-joins listings to NYC NTAs to attach a neighbourhood and borough to every row, and maps listing density and volume by neighbourhood.

### 3. Exploratory Data Audit
Structural check of the raw data (shape, dtypes, missing values) before any cleaning.

### 4. Data Cleaning & Feature Selection
Drops listings outside NYC boundaries, treats missing `reviews_per_month` as zero, removes price outliers (>99th percentile) and properties demanding over 365 `minimum_nights`, and log-transforms price.

### 5. No-ML Baseline
Median log-price per borough and room type, used as the benchmark the ML approach has to beat.

### 6. Feature Engineering
Nearest-subway and nearest-POI distance, computed once via haversine BallTree and reused across downstream steps.

### 7. Clustering Analysis (customer segmentation)
K-Means over scaled price, demand, host-activity, availability and proximity features. Cluster count (k=3) selected via the Elbow Method and silhouette score; clusters are profiled by price, room type mix, borough mix, and visualised both in PCA space and geographically.

### 8. Predictive Modelling
Comprehensive model evaluation using `GridSearchCV` to compare Decision Trees, Random Forests, and Gradient Boosting regressors. The best-performing model is selected based on RMSE (converted back to USD), using features available at listing time to preserve the cold-start framing.

### 9. Model Evaluation & Spatial Diagnostics
5-fold out-of-fold cross-validated predictions, aggregated into mean absolute error per neighbourhood to map where the model is least reliable, i.e. where unobserved listing quality (not location) is driving price.

### 10. Report Visualisations
Five final notebook cells create and save the report's model-comparison, feature-importance,
cluster-distribution, listing-density and neighbourhood-uncertainty figures under
report_visualisations/.

---

## Evaluation Metrics

The regression models are evaluated using **MAE_USD**, **RMSE_USD**, **R2_USD**, and **R2_Log_Price**, comparing the best-performing `GridSearchCV` estimator against the no-ML median baseline (Step 5) and reporting out-of-fold spatial error (Step 9).

---

## Team & Roles

* **Zee Mehmood:** Editor, Research & Reference Support, Business Question Owner
* **John-Jon Steyn:** Data Analytics & Visualisation
* **Džemal Jukić:** Technical Integration, Model Evaluation & Business Insight Lead
* **Julio Espinosa Rifel:** Data Engineering
* **Patrick Tosto:** Modelling

---

## Workflow & Reproducibility

* **Code & Data:** The original dataset remains unchanged. All data preprocessing, structural wrangling, and cleaning are performed programmatically in `main.ipynb`, so the pipeline can be re-run end to end from raw source data.
* **Version Control:** GitHub acts as the source of truth for all machine learning code, scripts, and experimental artefacts.
* **Collaboration:** Google Drive is used for report drafting, while Telegram serves as the primary communication channel.
* **Quality Assurance:** All analytical contributions include code, outputs, and methodological explanations. No model result or visualisation is merged without peer review.

---

## Technology Stack

| Library | Purpose |
|---|---|
| **kaggle** | Programmatic download of AB_NYC_2019 |
| **pandas / numpy** | Data manipulation, numerical computation |
| **geopandas** | Spatial join between listings and NTA boundaries, choropleth mapping |
| **matplotlib** | Visualisation (density maps, elbow/silhouette plots, cluster plots) |
| **python-dotenv** | Environment variable loading (Kaggle credentials) |
| **scikit-learn** | `BallTree`, `KMeans`, `StandardScaler`, `PCA`, `RandomForestRegressor`, `GradientBoostingRegressor`, `DecisionTreeRegressor`, `GridSearchCV`, `train_test_split`, `cross_val_predict`, `r2_score`, `mean_absolute_error`, `mean_squared_error`, `make_scorer`, `silhouette_score` |

*Exact pinned versions live in `requirements.txt`; add them here once that file is finalised so the table matches what's actually installed, rather than guessing at numbers.*

---

## Getting Started

### Prerequisites
- Python 3.x (matching the notebook's kernel)
- A Kaggle account with API credentials (`kaggle.json`), for automatic dataset download
- Internet access at first run, to fetch the NTA, MTA, and AB_NYC_2019 datasets

### 1. Set up the environment
```bash
python -m venv .venv
.venv\scripts\activate      # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

### 2. Configure credentials
Place your Kaggle API token where the `kaggle` package expects it (`~/.kaggle/kaggle.json`, or per Kaggle's documentation for your OS), and create a `.env` file for any other required variables.

### 3. Run the notebook
```bash
jupyter notebook main.ipynb
```
On first run, the notebook downloads AB_NYC_2019 from Kaggle and reads the NTA and MTA datasets directly from their public URLs; the Kaggle CSV is cached locally under `dataset/` so subsequent runs skip the download.

---

## Use of Generative AI — Acknowledgement

In accordance with the University of Essex Online guidelines on academic integrity and the transparent use of generative AI tools in assessed work, the following declaration is made:

**Tool:** Claude Sonnet 5 (Anthropic, https://www.anthropic.com/claude)

**Context of use:** Claude Sonnet 5 was used as a supporting development agent during the implementation phase of this project. Its role was limited to reorganising and cleaning the existing analysis notebook (consolidating imports, removing a duplicated computation, adding section-level structure) and drafting this README's documentation and formatting.

**Author responsibility:** All architectural decisions, design choices, modelling strategies, evaluation methodology, and critical business interpretation presented in this submission are solely the intellectual work of the team. Every output produced by the AI tool was critically reviewed, validated, and where necessary revised or rejected by the authors before incorporation. The team retains full responsibility for the correctness, originality, and academic integrity of the submitted work.

**Scope of AI assistance:**
- Code reorganisation and structural clean-up of an existing, author-written notebook
- Documentation drafting and formatting

**Not AI-generated:** The business question, track/approach selection, feature engineering choices (subway and POI proximity), cluster interpretation, model selection, and all business/ethical implications discussed in the accompanying report are the team's original intellectual contributions.

---

## References

- Gomonov, D. (2019) *New York City Airbnb open data* [Dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data (Accessed: 16 August 2026).
- City of New York (2020) *2020 Neighborhood Tabulation Areas (NTAs) – Mapped* [Dataset]. Available at: https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-Mapped/4hft-v355 (Accessed: 15 August 2026).
- New York State Open Data (n.d.) *MTA Subway Entrances and Exits*. Available at: https://data.ny.gov/Transportation/MTA-Subway-Entrances-and-Exits/i9wp-a4ja (Accessed: 16 August 2026).
