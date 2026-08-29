"""Generate standalone market, clustering, and model visualisations for NYC Airbnb."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas
import matplotlib

matplotlib.use("Agg")
import numpy
import pandas
import seaborn
import sklearn
from matplotlib import pyplot
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

REPOSITORY_ROOT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "dataset"
VISUALISATION_OUTPUT_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "visualisations"
REPORT_VISUALISATION_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "report_visualisations"
AIRBNB_SOURCE_CSV_PATH = DATA_DIRECTORY / "AB_NYC_2019.csv"
AIRBNB_SOURCE_ARCHIVE_PATH = DATA_DIRECTORY / "new-york-city-airbnb-open-data.zip"
NEIGHBOURHOOD_TABULATION_AREA_GEOJSON_PATH = DATA_DIRECTORY / "nyc_nta_2020.geojson"
SUBWAY_ENTRANCE_CSV_PATH = DATA_DIRECTORY / "mta_subway_entrances.csv"
AIRBNB_DATASET_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/dgomonov/new-york-city-airbnb-open-data"
NEIGHBOURHOOD_TABULATION_AREA_DOWNLOAD_URL = (
    "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson"
)
SUBWAY_ENTRANCE_DOWNLOAD_URL = (
    "https://data.ny.gov/api/views/i9wp-a4ja/rows.csv?accessType=DOWNLOAD"
)
CLUSTER_RANDOM_STATE = 85
MODEL_RANDOM_STATE = 85
EARTH_RADIUS_METRES = 6371000
EXPECTED_VISUALISATION_COUNT = 39
EXPECTED_REPORT_VISUALISATION_COUNT = 9
REPORT_FIGURE_SIZE_INCHES = (7.2, 5.04)
REPORT_FIGURE_DPI = 250
MODEL_SELECTION_CV_FOLDS = 3
MODEL_ESTIMATION_CV_FOLDS = 5
MODEL_FAMILY_ORDER = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]
MODEL_COLOURS = {
    "Decision Tree": "#4c78a8",
    "Random Forest": "#f2a541",
    "Gradient Boosting": "#157f68",
    "Neighbourhood median baseline": "#7a7a7a",
}
MODEL_PARAMETER_GRIDS = {
    "Decision Tree": {
        "max_depth": [6, 10, 15],
        "min_samples_split": [20, 50],
        "min_samples_leaf": [10, 20],
    },
    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [15, None],
        "min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [10, 20],
        "subsample": [0.8],
    },
}
CLUSTER_ORDER = [0, 1, 2]
CLUSTER_COLOURS = {
    0: "#0072B2",
    1: "#E69F00",
    2: "#CC79A7",
}
CLUSTER_NAMES = {
    0: "Mainstream market",
    1: "Premium professional",
    2: "Lower-price review-active",
}
CLUSTERING_FEATURES = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "dist_meters_to_subway",
    "dist_min_poi_meters",
]


def ensure_source_files() -> None:
    """Download public source files when the ignored cache is absent."""
    DATA_DIRECTORY.mkdir(exist_ok=True)
    if not AIRBNB_SOURCE_CSV_PATH.exists():
        urllib.request.urlretrieve(
            AIRBNB_DATASET_DOWNLOAD_URL, AIRBNB_SOURCE_ARCHIVE_PATH
        )
        with zipfile.ZipFile(AIRBNB_SOURCE_ARCHIVE_PATH) as archive:
            archive.extract("AB_NYC_2019.csv", DATA_DIRECTORY)
    if not NEIGHBOURHOOD_TABULATION_AREA_GEOJSON_PATH.exists():
        urllib.request.urlretrieve(
            NEIGHBOURHOOD_TABULATION_AREA_DOWNLOAD_URL,
            NEIGHBOURHOOD_TABULATION_AREA_GEOJSON_PATH,
        )
    if not SUBWAY_ENTRANCE_CSV_PATH.exists():
        urllib.request.urlretrieve(
            SUBWAY_ENTRANCE_DOWNLOAD_URL, SUBWAY_ENTRANCE_CSV_PATH
        )


def set_plot_style() -> None:
    """Configure the shared Seaborn and Matplotlib report style."""
    seaborn.set_theme(style="whitegrid", context="notebook")
    pyplot.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(visualisation_figure: pyplot.Figure, output_filename: str) -> None:
    """Save and close one visualisation in the output directory."""
    VISUALISATION_OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    visualisation_figure.savefig(
        VISUALISATION_OUTPUT_DIRECTORY / output_filename,
        bbox_inches="tight",
        facecolor="white",
    )
    pyplot.close(visualisation_figure)


def create_standalone_figures(
    number_of_rows: int,
    number_of_columns: int,
    *,
    figure_size: tuple[float, float] = (10, 7),
) -> tuple[list[pyplot.Figure], numpy.ndarray]:
    """Create an independent figure for every position in a logical grid."""
    visualisation_figures = []
    subplot_axes_grid = numpy.empty((number_of_rows, number_of_columns), dtype=object)
    for figure_row_index in range(number_of_rows):
        for figure_column_index in range(number_of_columns):
            visualisation_figure, subplot_axis = pyplot.subplots(figsize=figure_size)
            visualisation_figures.append(visualisation_figure)
            subplot_axes_grid[figure_row_index, figure_column_index] = subplot_axis
    return visualisation_figures, subplot_axes_grid.squeeze()


def save_standalone_figures(
    visualisation_figures: list[pyplot.Figure], output_filenames: list[str]
) -> None:
    """Save each independent figure under its corresponding PNG filename."""
    if len(visualisation_figures) != len(output_filenames):
        raise ValueError(
            "The number of visualisation figures must match the number of filenames."
        )
    for visualisation_figure, output_filename in zip(
        visualisation_figures, output_filenames
    ):
        seaborn.despine(visualisation_figure)
        visualisation_figure.tight_layout(pad=1.2)
        save_figure(visualisation_figure, output_filename)


def clear_generated_outputs() -> None:
    """Remove files previously generated by this script."""
    VISUALISATION_OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for existing_visualisation_path in VISUALISATION_OUTPUT_DIRECTORY.glob("*.png"):
        existing_visualisation_path.unlink()
    for generated_table_name in [
        "metrics_summary.json",
        "cluster_summary.csv",
        "cluster_volatility_metrics.csv",
        "nta_volatility_metrics.csv",
        "segment_volatility_metrics.csv",
        "model_comparison_metrics.csv",
        "grid_search_results.csv",
        "nta_opportunity_metrics.csv",
        "segment_opportunity_metrics.csv",
    ]:
        generated_table_path = VISUALISATION_OUTPUT_DIRECTORY / generated_table_name
        if generated_table_path.exists():
            generated_table_path.unlink()


def label_bars(
    subplot_axis: pyplot.Axes, *, suffix: str = "", decimals: int = 0
) -> None:
    """Label every bar container using consistent numeric formatting."""
    for bar_container in subplot_axis.containers:
        if not hasattr(bar_container, "datavalues"):
            continue
        formatted_bar_labels = [
            f"{value:,.{decimals}f}{suffix}" for value in bar_container.datavalues
        ]
        subplot_axis.bar_label(
            bar_container, labels=formatted_bar_labels, padding=3, fontsize=9
        )


def label_signed_horizontal_bars(
    subplot_axis: pyplot.Axes, *, suffix: str = "", decimals: int = 0
) -> None:
    """Label signed horizontal bars while preserving space around both ends."""
    horizontal_axis_minimum, horizontal_axis_maximum = subplot_axis.get_xlim()
    horizontal_axis_span = horizontal_axis_maximum - horizontal_axis_minimum
    all_bar_values = [
        float(value)
        for bar_container in subplot_axis.containers
        for value in bar_container.datavalues
        if numpy.isfinite(value)
    ]
    if not all_bar_values:
        return
    label_offset = 0.015 * horizontal_axis_span
    label_margin = 0.18 * horizontal_axis_span
    subplot_axis.set_xlim(
        min(horizontal_axis_minimum, min(all_bar_values) - label_margin),
        max(horizontal_axis_maximum, max(all_bar_values) + label_margin),
    )
    for bar_container in subplot_axis.containers:
        for bar_patch, value in zip(bar_container.patches, bar_container.datavalues):
            if not numpy.isfinite(value):
                continue
            subplot_axis.text(
                value + label_offset if value >= 0 else value - label_offset,
                bar_patch.get_y() + bar_patch.get_height() / 2,
                f"{value:,.{decimals}f}{suffix}",
                horizontalalignment="left" if value >= 0 else "right",
                verticalalignment="center",
                fontsize=9,
            )


def cluster_display_label(cluster: int, *, multiline: bool = False) -> str:
    """Return a stable report label for one provisional cluster name."""
    separator = "\n" if multiline else " — "
    return f"Cluster {cluster}{separator}{CLUSTER_NAMES[cluster]}"


def nearest_distance_metres(
    origin_listing_locations_dataframe: pandas.DataFrame,
    destination_coordinates_degrees: numpy.ndarray,
) -> numpy.ndarray:
    """Return each listing's haversine distance to its nearest destination."""
    origin_coordinates_radians = numpy.radians(
        origin_listing_locations_dataframe[["latitude", "longitude"]].to_numpy()
    )
    nearest_neighbour_tree = BallTree(
        numpy.radians(destination_coordinates_degrees), metric="haversine"
    )
    nearest_distances_radians, nearest_destination_indices = (
        nearest_neighbour_tree.query(origin_coordinates_radians, k=1)
    )
    return nearest_distances_radians.ravel() * EARTH_RADIUS_METRES


def prepare_data() -> (
    tuple[pandas.DataFrame, geopandas.GeoDataFrame, dict[str, int | float]]
):
    """Clean and spatially enrich the listings, returning retention metrics."""
    source_listings_dataframe = pandas.read_csv(AIRBNB_SOURCE_CSV_PATH)
    neighbourhood_tabulation_areas_geodataframe = geopandas.read_file(
        NEIGHBOURHOOD_TABULATION_AREA_GEOJSON_PATH
    ).to_crs("EPSG:4326")
    listing_points_geodataframe = geopandas.GeoDataFrame(
        source_listings_dataframe,
        geometry=geopandas.points_from_xy(
            source_listings_dataframe["longitude"],
            source_listings_dataframe["latitude"],
        ),
        crs="EPSG:4326",
    )
    geographically_enriched_listings_geodataframe = geopandas.sjoin(
        listing_points_geodataframe,
        neighbourhood_tabulation_areas_geodataframe[
            ["ntaname", "boroname", "geometry"]
        ],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])
    listings_inside_official_neighbourhoods_dataframe = (
        geographically_enriched_listings_geodataframe.dropna(subset=["ntaname"]).copy()
    )
    listings_inside_official_neighbourhoods_dataframe["reviews_per_month"] = (
        listings_inside_official_neighbourhoods_dataframe["reviews_per_month"].fillna(0)
    )
    price_cap = float(
        listings_inside_official_neighbourhoods_dataframe["price"].quantile(0.99)
    )
    listings_with_valid_prices_dataframe = (
        listings_inside_official_neighbourhoods_dataframe[
            (listings_inside_official_neighbourhoods_dataframe["price"] > 0)
            & (listings_inside_official_neighbourhoods_dataframe["price"] <= price_cap)
        ].copy()
    )
    cleaned_listings_dataframe = listings_with_valid_prices_dataframe[
        listings_with_valid_prices_dataframe["minimum_nights"] <= 365
    ].copy()
    subway_entrances_dataframe = pandas.read_csv(SUBWAY_ENTRANCE_CSV_PATH)
    subway_entrance_latitude_column = (
        "Entrance Latitude"
        if "Entrance Latitude" in subway_entrances_dataframe
        else "Latitude"
    )
    subway_entrance_longitude_column = (
        "Entrance Longitude"
        if "Entrance Longitude" in subway_entrances_dataframe
        else "Longitude"
    )
    subway_entrances_dataframe = subway_entrances_dataframe.dropna(
        subset=[subway_entrance_latitude_column, subway_entrance_longitude_column]
    )
    cleaned_listings_dataframe["dist_meters_to_subway"] = nearest_distance_metres(
        cleaned_listings_dataframe,
        subway_entrances_dataframe[
            [subway_entrance_latitude_column, subway_entrance_longitude_column]
        ].to_numpy(),
    )
    points_of_interest = numpy.array(
        [
            [40.758, -73.9855],
            [40.7644, -73.973],
            [40.7794, -73.9632],
            [40.7484, -73.9857],
            [40.7308, -73.9973],
            [40.7233, -73.9988],
            [40.7074, -74.0113],
            [40.7505, -73.9934],
            [40.7527, -73.9772],
            [40.716, -73.9587],
            [40.6826, -73.9754],
            [40.7465, -73.9455],
        ]
    )
    cleaned_listings_dataframe["dist_min_poi_meters"] = nearest_distance_metres(
        cleaned_listings_dataframe, points_of_interest
    )
    cleaned_listings_dataframe["log_price"] = numpy.log1p(
        cleaned_listings_dataframe["price"]
    )
    data_retention_metrics = {
        "Raw listings": len(source_listings_dataframe),
        "Inside an official NTA": len(
            listings_inside_official_neighbourhoods_dataframe
        ),
        "Valid price after 99th-percentile cap": len(
            listings_with_valid_prices_dataframe
        ),
        "Minimum stay of 365 days or fewer": len(cleaned_listings_dataframe),
        "price_cap": price_cap,
    }
    return (
        cleaned_listings_dataframe,
        neighbourhood_tabulation_areas_geodataframe,
        data_retention_metrics,
    )


def assign_listing_clusters(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Assign the notebook's stable three-cluster segmentation."""
    clustering_features_scaled = StandardScaler().fit_transform(
        cleaned_listings_dataframe[CLUSTERING_FEATURES]
    )
    clustering_model = KMeans(
        n_clusters=len(CLUSTER_ORDER),
        random_state=CLUSTER_RANDOM_STATE,
        n_init=10,
    )
    clustered_listings_dataframe = cleaned_listings_dataframe.copy()
    clustered_listings_dataframe["cluster"] = clustering_model.fit_predict(
        clustering_features_scaled
    )
    clustered_listings_dataframe["cluster_name"] = clustered_listings_dataframe[
        "cluster"
    ].map(CLUSTER_NAMES)
    principal_component_coordinates = PCA(n_components=2).fit_transform(
        clustering_features_scaled
    )
    clustered_listings_dataframe["pca_1"] = principal_component_coordinates[:, 0]
    clustered_listings_dataframe["pca_2"] = principal_component_coordinates[:, 1]
    listings_grouped_by_cluster = clustered_listings_dataframe.groupby(
        "cluster", observed=True
    )
    cluster_summary_dataframe = listings_grouped_by_cluster[CLUSTERING_FEATURES].mean()
    cluster_summary_dataframe.insert(
        0, "number_of_listings", listings_grouped_by_cluster.size()
    )
    cluster_summary_dataframe = cluster_summary_dataframe.reset_index()
    cluster_summary_dataframe.insert(
        1, "cluster_name", cluster_summary_dataframe["cluster"].map(CLUSTER_NAMES)
    )
    return clustered_listings_dataframe, cluster_summary_dataframe


def build_model_data(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.Series]:
    """Encode model features and return them with the log-price target."""
    model_input_columns = [
        "boroname",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "log_price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "dist_meters_to_subway",
        "dist_min_poi_meters",
    ]
    encoded_model_dataframe = pandas.get_dummies(
        cleaned_listings_dataframe[model_input_columns],
        columns=["room_type", "boroname"],
        drop_first=True,
        dtype=float,
    )
    return (
        encoded_model_dataframe.drop(columns=["price", "log_price"]),
        encoded_model_dataframe["log_price"],
    )


def rmse_usd_from_log_prices(
    actual_log_prices: pandas.Series | numpy.ndarray,
    predicted_log_prices: pandas.Series | numpy.ndarray,
) -> float:
    """Calculate RMSE after converting log prices back to US dollars."""
    return float(
        mean_squared_error(
            numpy.expm1(actual_log_prices),
            numpy.expm1(predicted_log_prices),
        )
        ** 0.5
    )


def calculate_prediction_metrics(
    actual_log_prices: pandas.Series,
    predicted_log_prices: numpy.ndarray,
) -> dict[str, float]:
    """Calculate comparable evaluation metrics in log and dollar space."""
    actual_prices = numpy.expm1(actual_log_prices)
    predicted_prices = numpy.expm1(predicted_log_prices)
    absolute_errors = numpy.abs(actual_prices - predicted_prices)
    return {
        "r2_log_price": float(r2_score(actual_log_prices, predicted_log_prices)),
        "r2_price_usd": float(r2_score(actual_prices, predicted_prices)),
        "mae_usd": float(mean_absolute_error(actual_prices, predicted_prices)),
        "rmse_usd": float(mean_squared_error(actual_prices, predicted_prices) ** 0.5),
        "median_absolute_error_usd": float(numpy.median(absolute_errors)),
        "wape_percent": float(
            100 * numpy.sum(absolute_errors) / numpy.sum(actual_prices)
        ),
    }


def build_median_baseline_predictions(
    cleaned_listings_dataframe: pandas.DataFrame,
    training_indices: pandas.Index,
    testing_indices: pandas.Index,
    training_log_price_target_series: pandas.Series,
) -> numpy.ndarray:
    """Build a training-only NTA/room median with defensible fallbacks."""
    training_baseline_dataframe = cleaned_listings_dataframe.loc[
        training_indices,
        ["ntaname", "boroname", "room_type"],
    ].assign(log_price=training_log_price_target_series)
    neighbourhood_room_lookup = (
        training_baseline_dataframe.groupby(
            ["ntaname", "room_type"],
            observed=True,
        )["log_price"]
        .median()
        .rename("neighbourhood_room_log_price")
    )
    borough_room_lookup = (
        training_baseline_dataframe.groupby(
            ["boroname", "room_type"],
            observed=True,
        )["log_price"]
        .median()
        .rename("borough_room_log_price")
    )
    testing_baseline_dataframe = cleaned_listings_dataframe.loc[
        testing_indices,
        ["ntaname", "boroname", "room_type"],
    ]
    testing_baseline_dataframe = testing_baseline_dataframe.join(
        neighbourhood_room_lookup,
        on=["ntaname", "room_type"],
    ).join(
        borough_room_lookup,
        on=["boroname", "room_type"],
    )
    baseline_predictions = testing_baseline_dataframe[
        "neighbourhood_room_log_price"
    ].fillna(testing_baseline_dataframe["borough_room_log_price"])
    return baseline_predictions.fillna(
        training_log_price_target_series.median()
    ).to_numpy()


def train_and_score(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.Series, dict[str, object]]:
    """Select and evaluate a cold-start model, then attach out-of-fold errors."""
    model_features_dataframe, log_price_target_series = build_model_data(
        cleaned_listings_dataframe
    )
    (
        training_features_dataframe,
        testing_features_dataframe,
        training_log_price_target_series,
        testing_log_price_target_series,
    ) = train_test_split(
        model_features_dataframe,
        log_price_target_series,
        test_size=0.2,
        random_state=MODEL_RANDOM_STATE,
    )
    model_estimators = {
        "Decision Tree": DecisionTreeRegressor(
            random_state=MODEL_RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            random_state=MODEL_RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=MODEL_RANDOM_STATE,
        ),
    }
    rmse_usd_scorer = make_scorer(
        rmse_usd_from_log_prices,
        greater_is_better=False,
    )
    selection_cross_validation = KFold(
        n_splits=MODEL_SELECTION_CV_FOLDS,
        shuffle=True,
        random_state=MODEL_RANDOM_STATE,
    )
    fitted_estimators: dict[str, object] = {}
    testing_predictions_by_model: dict[str, numpy.ndarray] = {}
    model_comparison_records: list[dict[str, object]] = []
    grid_search_result_frames: list[pandas.DataFrame] = []

    for model_family in MODEL_FAMILY_ORDER:
        parameter_grid = MODEL_PARAMETER_GRIDS[model_family]
        parameter_combination_count = int(
            numpy.prod([len(values) for values in parameter_grid.values()])
        )
        print(
            f"Grid searching {model_family}: "
            f"{parameter_combination_count} parameter combinations"
        )
        grid_search = GridSearchCV(
            estimator=model_estimators[model_family],
            param_grid=parameter_grid,
            scoring=rmse_usd_scorer,
            cv=selection_cross_validation,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid_search.fit(
            training_features_dataframe,
            training_log_price_target_series,
        )
        fitted_estimator = clone(grid_search.best_estimator_)
        fitted_estimator.fit(
            training_features_dataframe,
            training_log_price_target_series,
        )
        predicted_log_prices = fitted_estimator.predict(testing_features_dataframe)
        testing_predictions_by_model[model_family] = predicted_log_prices
        fitted_estimators[model_family] = fitted_estimator
        test_metrics = calculate_prediction_metrics(
            testing_log_price_target_series,
            predicted_log_prices,
        )
        best_grid_index = int(grid_search.best_index_)
        model_comparison_records.append(
            {
                "model_family": model_family,
                "best_cv_rmse_usd": float(-grid_search.best_score_),
                "best_cv_rmse_std_usd": float(
                    grid_search.cv_results_["std_test_score"][best_grid_index]
                ),
                "test_mae_usd": test_metrics["mae_usd"],
                "test_rmse_usd": test_metrics["rmse_usd"],
                "test_median_absolute_error_usd": test_metrics[
                    "median_absolute_error_usd"
                ],
                "test_r2_log_price": test_metrics["r2_log_price"],
                "test_r2_price_usd": test_metrics["r2_price_usd"],
                "test_wape_percent": test_metrics["wape_percent"],
                "best_parameters": json.dumps(
                    grid_search.best_params_,
                    sort_keys=True,
                ),
            }
        )
        family_grid_results_dataframe = pandas.DataFrame(grid_search.cv_results_)
        family_grid_results_dataframe["model_family"] = model_family
        family_grid_results_dataframe["cv_rmse_usd"] = -family_grid_results_dataframe[
            "mean_test_score"
        ]
        family_grid_results_dataframe["cv_rmse_std_usd"] = (
            family_grid_results_dataframe["std_test_score"]
        )
        family_grid_results_dataframe["parameters"] = family_grid_results_dataframe[
            "params"
        ].map(lambda parameters: json.dumps(parameters, sort_keys=True))
        family_grid_results_dataframe["is_family_best"] = (
            family_grid_results_dataframe["rank_test_score"] == 1
        )
        grid_search_result_frames.append(
            family_grid_results_dataframe[
                [
                    "model_family",
                    "rank_test_score",
                    "cv_rmse_usd",
                    "cv_rmse_std_usd",
                    "mean_fit_time",
                    "parameters",
                    "is_family_best",
                ]
            ]
        )

    model_comparison_results_dataframe = pandas.DataFrame(model_comparison_records)
    selected_model_record = model_comparison_results_dataframe.loc[
        model_comparison_results_dataframe["best_cv_rmse_usd"].idxmin()
    ]
    selected_model_family = str(selected_model_record["model_family"])
    selected_model = fitted_estimators[selected_model_family]
    selected_predicted_log_prices = testing_predictions_by_model[selected_model_family]
    selected_test_metrics = calculate_prediction_metrics(
        testing_log_price_target_series,
        selected_predicted_log_prices,
    )
    model_comparison_results_dataframe["is_selected"] = (
        model_comparison_results_dataframe["model_family"] == selected_model_family
    )

    median_baseline_prediction_log = build_median_baseline_predictions(
        cleaned_listings_dataframe,
        training_features_dataframe.index,
        testing_features_dataframe.index,
        training_log_price_target_series,
    )
    median_baseline_metrics = calculate_prediction_metrics(
        testing_log_price_target_series,
        median_baseline_prediction_log,
    )
    model_comparison_results_dataframe = pandas.concat(
        [
            model_comparison_results_dataframe,
            pandas.DataFrame(
                [
                    {
                        "model_family": "Neighbourhood median baseline",
                        "best_cv_rmse_usd": numpy.nan,
                        "best_cv_rmse_std_usd": numpy.nan,
                        "test_mae_usd": median_baseline_metrics["mae_usd"],
                        "test_rmse_usd": median_baseline_metrics["rmse_usd"],
                        "test_median_absolute_error_usd": (
                            median_baseline_metrics["median_absolute_error_usd"]
                        ),
                        "test_r2_log_price": median_baseline_metrics["r2_log_price"],
                        "test_r2_price_usd": median_baseline_metrics["r2_price_usd"],
                        "test_wape_percent": median_baseline_metrics["wape_percent"],
                        "best_parameters": json.dumps(
                            {
                                "primary_grouping": "ntaname + room_type",
                                "fallback_grouping": "boroname + room_type",
                            },
                            sort_keys=True,
                        ),
                        "is_selected": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    display_order = {
        model_family: order
        for order, model_family in enumerate(
            MODEL_FAMILY_ORDER + ["Neighbourhood median baseline"]
        )
    }
    model_comparison_results_dataframe["display_order"] = (
        model_comparison_results_dataframe["model_family"].map(display_order)
    )
    model_comparison_results_dataframe = model_comparison_results_dataframe.sort_values(
        "display_order"
    ).reset_index(drop=True)

    testing_predictions_dataframe = pandas.DataFrame(
        {
            "actual_price": numpy.expm1(testing_log_price_target_series),
            "predicted_price": numpy.expm1(selected_predicted_log_prices),
        },
        index=testing_features_dataframe.index,
    )
    testing_predictions_dataframe["residual"] = (
        testing_predictions_dataframe["actual_price"]
        - testing_predictions_dataframe["predicted_price"]
    )
    testing_predictions_dataframe["absolute_error"] = testing_predictions_dataframe[
        "residual"
    ].abs()

    model_comparison_export = {}
    for comparison_record in model_comparison_results_dataframe.to_dict(
        orient="records"
    ):
        model_family = str(comparison_record["model_family"])
        best_cv_rmse_usd = comparison_record["best_cv_rmse_usd"]
        best_cv_rmse_std_usd = comparison_record["best_cv_rmse_std_usd"]
        model_comparison_export[model_family] = {
            "best_cv_rmse_usd": (
                None if pandas.isna(best_cv_rmse_usd) else float(best_cv_rmse_usd)
            ),
            "best_cv_rmse_std_usd": (
                None
                if pandas.isna(best_cv_rmse_std_usd)
                else float(best_cv_rmse_std_usd)
            ),
            "test_mae_usd": float(comparison_record["test_mae_usd"]),
            "test_rmse_usd": float(comparison_record["test_rmse_usd"]),
            "test_median_absolute_error_usd": float(
                comparison_record["test_median_absolute_error_usd"]
            ),
            "test_r2_log_price": float(comparison_record["test_r2_log_price"]),
            "test_r2_price_usd": float(comparison_record["test_r2_price_usd"]),
            "test_wape_percent": float(comparison_record["test_wape_percent"]),
            "best_parameters": json.loads(str(comparison_record["best_parameters"])),
            "is_selected": bool(comparison_record["is_selected"]),
        }

    model_evaluation_metrics: dict[str, object] = {
        "winner_model_family": selected_model_family,
        "winner_best_parameters": json.loads(
            str(selected_model_record["best_parameters"])
        ),
        "winner_cv_rmse_usd": float(selected_model_record["best_cv_rmse_usd"]),
        "r2_log_price": selected_test_metrics["r2_log_price"],
        "r2_price_usd": selected_test_metrics["r2_price_usd"],
        "mae_usd": selected_test_metrics["mae_usd"],
        "rmse_usd": selected_test_metrics["rmse_usd"],
        "median_absolute_error_usd": selected_test_metrics["median_absolute_error_usd"],
        "wape_percent": selected_test_metrics["wape_percent"],
        "test_rows": len(testing_features_dataframe),
        "median_baseline_r2_log_price": median_baseline_metrics["r2_log_price"],
        "median_baseline_r2_price_usd": median_baseline_metrics["r2_price_usd"],
        "median_baseline_mae_usd": median_baseline_metrics["mae_usd"],
        "median_baseline_rmse_usd": median_baseline_metrics["rmse_usd"],
        "median_baseline_median_absolute_error_usd": (
            median_baseline_metrics["median_absolute_error_usd"]
        ),
        "median_baseline_wape_percent": median_baseline_metrics["wape_percent"],
        "model_selection_cv_folds": MODEL_SELECTION_CV_FOLDS,
        "out_of_fold_estimation_cv_folds": MODEL_ESTIMATION_CV_FOLDS,
        "review_history_excluded": True,
        "model_comparison": model_comparison_export,
    }

    permutation_importance_sample_dataframe = testing_features_dataframe.sample(
        n=min(5000, len(testing_features_dataframe)),
        random_state=MODEL_RANDOM_STATE,
    )
    permutation_importance_result = permutation_importance(
        selected_model,
        permutation_importance_sample_dataframe,
        testing_log_price_target_series.loc[
            permutation_importance_sample_dataframe.index
        ],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance_series = pandas.Series(
        permutation_importance_result.importances_mean,
        index=model_features_dataframe.columns,
    ).sort_values(ascending=False)

    print(
        f"Generating {MODEL_ESTIMATION_CV_FOLDS}-fold out-of-fold estimates "
        f"with {selected_model_family}"
    )
    out_of_fold_predicted_log_prices = cross_val_predict(
        clone(selected_model),
        model_features_dataframe,
        log_price_target_series,
        cv=KFold(
            n_splits=MODEL_ESTIMATION_CV_FOLDS,
            shuffle=True,
            random_state=MODEL_RANDOM_STATE,
        ),
        n_jobs=1,
    )
    cleaned_listings_dataframe = cleaned_listings_dataframe.copy()
    cleaned_listings_dataframe["predicted_price"] = numpy.expm1(
        out_of_fold_predicted_log_prices
    )
    cleaned_listings_dataframe["residual_usd"] = (
        cleaned_listings_dataframe["price"]
        - cleaned_listings_dataframe["predicted_price"]
    )
    cleaned_listings_dataframe["absolute_error"] = cleaned_listings_dataframe[
        "residual_usd"
    ].abs()

    model_evaluation_metrics["_test_results"] = testing_predictions_dataframe
    model_evaluation_metrics["_model_comparison_results"] = (
        model_comparison_results_dataframe
    )
    model_evaluation_metrics["_grid_search_results"] = pandas.concat(
        grid_search_result_frames,
        ignore_index=True,
    )
    return (
        cleaned_listings_dataframe,
        feature_importance_series,
        model_evaluation_metrics,
    )


def plot_data_and_market(
    source_listings_dataframe: pandas.DataFrame,
    cleaned_listings_dataframe: pandas.DataFrame,
    data_retention_metrics: dict[str, int | float],
) -> None:
    """Create standalone data-quality and market-composition visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    missing_value_percentages = 100 * source_listings_dataframe.isna().mean()
    missing_value_percentages = missing_value_percentages[
        missing_value_percentages > 0
    ].sort_values()
    subplot_axes[0, 0].barh(
        missing_value_percentages.index,
        missing_value_percentages.values,
        color="#4c78a8",
    )
    subplot_axes[0, 0].set_title("Missing values in the source data")
    subplot_axes[0, 0].set_xlabel("Missing observations (%)")
    label_bars(subplot_axes[0, 0], suffix="%", decimals=2)
    data_retention_step_names = [
        dictionary_key
        for dictionary_key in data_retention_metrics
        if dictionary_key != "price_cap"
    ]
    data_retention_listing_counts = [
        float(data_retention_metrics[dictionary_key])
        for dictionary_key in data_retention_step_names
    ]
    subplot_axes[0, 1].barh(
        data_retention_step_names[::-1],
        data_retention_listing_counts[::-1],
        color="#5b8e7d",
    )
    subplot_axes[0, 1].set_title("Cleaning and geographic-join retention")
    subplot_axes[0, 1].set_xlabel("Listings retained")
    for bar_patch, value in zip(
        subplot_axes[0, 1].patches, data_retention_listing_counts[::-1]
    ):
        subplot_axes[0, 1].text(
            value,
            bar_patch.get_y() + bar_patch.get_height() / 2,
            f"  {value:,.0f} ({100 * value / data_retention_listing_counts[0]:.2f}%)",
            va="center",
            fontsize=9,
        )
    listing_counts_by_borough = (
        cleaned_listings_dataframe["boroname"].value_counts().sort_values()
    )
    subplot_axes[1, 0].barh(
        listing_counts_by_borough.index,
        listing_counts_by_borough.values,
        color="#f2a65a",
    )
    subplot_axes[1, 0].set_title("Listings by borough")
    subplot_axes[1, 0].set_xlabel("Clean listings")
    label_bars(subplot_axes[1, 0])
    listing_counts_by_room_type = (
        cleaned_listings_dataframe["room_type"].value_counts().sort_values()
    )
    subplot_axes[1, 1].barh(
        listing_counts_by_room_type.index,
        listing_counts_by_room_type.values,
        color="#8f6bb3",
    )
    subplot_axes[1, 1].set_title("Listings by room type")
    subplot_axes[1, 1].set_xlabel("Clean listings")
    label_bars(subplot_axes[1, 1])
    save_standalone_figures(
        visualisation_figures,
        [
            "01_source_missing_values.png",
            "02_cleaning_and_geographic_join_retention.png",
            "03_listings_by_borough.png",
            "04_listings_by_room_type.png",
        ],
    )


def plot_price_segments(cleaned_listings_dataframe: pandas.DataFrame) -> None:
    """Create standalone price-distribution and market-segment visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    room_types_ordered_by_median_price = (
        cleaned_listings_dataframe.groupby("room_type", observed=True)["price"]
        .median()
        .sort_values()
        .index
    )
    seaborn.histplot(
        data=cleaned_listings_dataframe,
        x="price",
        hue="room_type",
        hue_order=room_types_ordered_by_median_price,
        bins=50,
        element="step",
        stat="density",
        common_norm=False,
        ax=subplot_axes[0, 0],
    )
    subplot_axes[0, 0].set_title("Price distribution by room type")
    subplot_axes[0, 0].set_xlabel("Nightly price (USD; capped at the 99th percentile)")
    subplot_axes[0, 0].set_ylabel("Density")
    seaborn.boxplot(
        data=cleaned_listings_dataframe,
        x="room_type",
        y="price",
        order=room_types_ordered_by_median_price,
        showfliers=False,
        color="#72a0c1",
        ax=subplot_axes[0, 1],
    )
    subplot_axes[0, 1].set_title("Price dispersion by room type")
    subplot_axes[0, 1].set_xlabel("")
    subplot_axes[0, 1].set_ylabel("Nightly price (USD)")
    subplot_axes[0, 1].tick_params(axis="x", rotation=12)
    price_segment_summary_dataframe = cleaned_listings_dataframe.pivot_table(
        index="boroname",
        columns="room_type",
        values="price",
        aggfunc=["median", "count"],
        observed=True,
    )
    seaborn.heatmap(
        price_segment_summary_dataframe["median"].reindex(
            columns=room_types_ordered_by_median_price
        ),
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Median nightly price (USD)"},
        ax=subplot_axes[1, 0],
    )
    subplot_axes[1, 0].set_title("Median price by borough and room type")
    subplot_axes[1, 0].set_xlabel("")
    subplot_axes[1, 0].set_ylabel("")
    seaborn.heatmap(
        price_segment_summary_dataframe["count"].reindex(
            columns=room_types_ordered_by_median_price
        ),
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar_kws={"label": "Listings"},
        ax=subplot_axes[1, 1],
    )
    subplot_axes[1, 1].set_title("Listing volume by borough and room type")
    subplot_axes[1, 1].set_xlabel("")
    subplot_axes[1, 1].set_ylabel("")
    save_standalone_figures(
        visualisation_figures,
        [
            "05_price_distribution_by_room_type.png",
            "06_price_dispersion_by_room_type.png",
            "07_median_price_by_borough_and_room_type.png",
            "08_listing_volume_by_borough_and_room_type.png",
        ],
    )


def plot_spatial_market(
    cleaned_listings_dataframe: pandas.DataFrame,
    neighbourhood_tabulation_areas_geodataframe: geopandas.GeoDataFrame,
) -> None:
    """Create maps of listing supply and median price by neighbourhood."""
    market_metrics_by_neighbourhood_dataframe = (
        cleaned_listings_dataframe.groupby("ntaname", observed=True)
        .agg(listings=("id", "size"), median_price=("price", "median"))
        .reset_index()
    )
    spatial_market_metrics_geodataframe = (
        neighbourhood_tabulation_areas_geodataframe.merge(
            market_metrics_by_neighbourhood_dataframe, on="ntaname", how="left"
        )
    )
    spatial_market_metrics_geodataframe["listings"] = (
        spatial_market_metrics_geodataframe["listings"].fillna(0)
    )
    visualisation_figures, subplot_axes = create_standalone_figures(
        1, 2, figure_size=(10, 8)
    )
    neighbourhoods_with_listings_geodataframe = spatial_market_metrics_geodataframe[
        spatial_market_metrics_geodataframe["listings"] > 0
    ]
    listing_density_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "listing_density",
        ["#9ECAE1", "#4292C6", "#2171B5", "#08306B"],
    )
    listing_density_normalisation = matplotlib.colors.PowerNorm(
        gamma=0.45,
        vmin=1,
        vmax=float(neighbourhoods_with_listings_geodataframe["listings"].max()),
    )
    spatial_market_metrics_geodataframe.plot(
        color="#D9D9D9",
        linewidth=0.25,
        edgecolor="white",
        ax=subplot_axes[0],
    )
    neighbourhoods_with_listings_geodataframe.plot(
        column="listings",
        cmap=listing_density_colormap,
        norm=listing_density_normalisation,
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Listings per NTA", "shrink": 0.72},
        ax=subplot_axes[0],
    )
    subplot_axes[0].legend(
        handles=[
            Patch(
                facecolor="#D9D9D9",
                edgecolor="#8C8C8C",
                label="No listings",
            )
        ],
        loc="lower left",
        frameon=False,
    )
    subplot_axes[0].set_title("Listing density by Neighbourhood Tabulation Area")
    subplot_axes[0].set_axis_off()
    neighbourhoods_with_sufficient_price_data_geodataframe = (
        spatial_market_metrics_geodataframe[
            spatial_market_metrics_geodataframe["listings"] >= 20
        ]
    )
    spatial_market_metrics_geodataframe.plot(
        color="#e5e5e5", edgecolor="white", linewidth=0.25, ax=subplot_axes[1]
    )
    neighbourhoods_with_sufficient_price_data_geodataframe.plot(
        column="median_price",
        cmap="viridis",
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Median nightly price (USD)", "shrink": 0.72},
        ax=subplot_axes[1],
    )
    subplot_axes[1].set_title("Median price by NTA (at least 20 listings)")
    subplot_axes[1].set_axis_off()
    save_standalone_figures(
        visualisation_figures,
        [
            "09_listing_density_by_nta.png",
            "10_median_price_by_nta.png",
        ],
    )


def plot_binned_relationship(
    subplot_axis: pyplot.Axes,
    listings_dataframe: pandas.DataFrame,
    relationship_metric_column: str,
    subplot_title: str,
    horizontal_axis_label: str,
) -> None:
    """Plot decile-level median price against a selected listing metric."""
    binned_relationship_source_dataframe = (
        listings_dataframe[[relationship_metric_column, "price"]].dropna().copy()
    )
    binned_relationship_source_dataframe["bin"] = pandas.qcut(
        binned_relationship_source_dataframe[relationship_metric_column],
        q=10,
        duplicates="drop",
    )
    binned_relationship_summary_dataframe = (
        binned_relationship_source_dataframe.groupby("bin", observed=True)
        .agg(x=(relationship_metric_column, "median"), median_price=("price", "median"))
        .reset_index(drop=True)
    )
    subplot_axis.plot(
        binned_relationship_summary_dataframe["x"],
        binned_relationship_summary_dataframe["median_price"],
        marker="o",
        color="#2f6f9f",
    )
    subplot_axis.set_title(subplot_title)
    subplot_axis.set_xlabel(horizontal_axis_label)
    subplot_axis.set_ylabel("Median nightly price (USD)")


def plot_operational_relationships(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> None:
    """Create standalone access and operating-characteristic visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    binned_relationship_specifications = [
        (
            "dist_meters_to_subway",
            "Distance to the nearest subway entrance",
            "Median distance within decile (metres)",
        ),
        (
            "dist_min_poi_meters",
            "Distance to the nearest selected visitor destination",
            "Median distance within decile (metres)",
        ),
        (
            "availability_365",
            "Annual availability",
            "Median availability within decile (days)",
        ),
        ("number_of_reviews", "Review volume", "Median reviews within decile"),
    ]
    for subplot_axis, binned_relationship_specification in zip(
        subplot_axes.ravel(), binned_relationship_specifications
    ):
        plot_binned_relationship(
            subplot_axis, cleaned_listings_dataframe, *binned_relationship_specification
        )
    save_standalone_figures(
        visualisation_figures,
        [
            "11_price_by_subway_distance.png",
            "12_price_by_visitor_destination_distance.png",
            "13_price_by_annual_availability.png",
            "14_price_by_review_volume.png",
        ],
    )


def plot_host_and_booking_policy_relationships(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> None:
    """Create standalone host-scale, policy and review-rate visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    relationship_specifications = [
        (
            "minimum_nights",
            "Minimum-stay requirement",
            "Median minimum stay within decile (nights)",
        ),
        (
            "reviews_per_month",
            "Recent review rate",
            "Median reviews per month within decile",
        ),
        (
            "calculated_host_listings_count",
            "Host portfolio size",
            "Median host listing count within decile",
        ),
    ]
    for subplot_axis, relationship_specification in zip(
        subplot_axes.ravel()[:3], relationship_specifications
    ):
        plot_binned_relationship(
            subplot_axis, cleaned_listings_dataframe, *relationship_specification
        )
    host_portfolio_band = pandas.cut(
        cleaned_listings_dataframe["calculated_host_listings_count"],
        bins=[0, 1, 2, 5, 10, numpy.inf],
        labels=[
            "1 listing",
            "2 listings",
            "3–5 listings",
            "6–10 listings",
            "11+ listings",
        ],
        include_lowest=True,
    )
    host_portfolio_listing_counts = host_portfolio_band.value_counts().sort_index()
    subplot_axes[1, 1].barh(
        host_portfolio_listing_counts.index.astype(str),
        host_portfolio_listing_counts.values,
        color="#8f6bb3",
    )
    subplot_axes[1, 1].set_title("Listings by host portfolio band")
    subplot_axes[1, 1].set_xlabel("Listings")
    label_bars(subplot_axes[1, 1])
    save_standalone_figures(
        visualisation_figures,
        [
            "15_price_by_minimum_stay.png",
            "16_price_by_recent_review_rate.png",
            "17_price_by_host_portfolio_size.png",
            "18_listings_by_host_portfolio_band.png",
        ],
    )


def plot_model_performance(
    testing_predictions_dataframe: pandas.DataFrame,
    model_evaluation_metrics: dict[str, object],
    model_comparison_results_dataframe: pandas.DataFrame,
) -> None:
    """Create winner diagnostics and held-out model-comparison charts."""
    selected_model_family = str(model_evaluation_metrics["winner_model_family"])
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    comparison_axis_upper_limit = float(
        max(
            testing_predictions_dataframe["actual_price"].quantile(0.995),
            testing_predictions_dataframe["predicted_price"].quantile(0.995),
        )
    )
    hexagonal_density_plot = subplot_axes[0, 0].hexbin(
        testing_predictions_dataframe["actual_price"],
        testing_predictions_dataframe["predicted_price"],
        gridsize=42,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    subplot_axes[0, 0].plot(
        [0, comparison_axis_upper_limit],
        [0, comparison_axis_upper_limit],
        linestyle="--",
        color="#c94845",
    )
    subplot_axes[0, 0].set(
        xlim=(0, comparison_axis_upper_limit),
        ylim=(0, comparison_axis_upper_limit),
    )
    subplot_axes[0, 0].set_title(
        f"Actual versus predicted price: {selected_model_family}"
    )
    subplot_axes[0, 0].set_xlabel("Actual price (USD)")
    subplot_axes[0, 0].set_ylabel("Predicted price (USD)")
    visualisation_figures[0].colorbar(
        hexagonal_density_plot,
        ax=subplot_axes[0, 0],
        label="Log listing density",
    )

    calibration_by_prediction_decile_dataframe = testing_predictions_dataframe.copy()
    calibration_by_prediction_decile_dataframe["decile"] = pandas.qcut(
        calibration_by_prediction_decile_dataframe["predicted_price"],
        q=10,
        duplicates="drop",
    )
    calibration_by_prediction_decile_dataframe = (
        calibration_by_prediction_decile_dataframe.groupby(
            "decile",
            observed=True,
        )
        .agg(
            predicted=("predicted_price", "mean"),
            actual=("actual_price", "mean"),
        )
        .reset_index(drop=True)
    )
    subplot_axes[0, 1].plot(
        calibration_by_prediction_decile_dataframe["predicted"],
        calibration_by_prediction_decile_dataframe["actual"],
        marker="o",
        color=MODEL_COLOURS[selected_model_family],
        label="Observed",
    )
    calibration_axis_upper_limit = max(
        calibration_by_prediction_decile_dataframe["predicted"].max(),
        calibration_by_prediction_decile_dataframe["actual"].max(),
    )
    subplot_axes[0, 1].plot(
        [0, calibration_axis_upper_limit],
        [0, calibration_axis_upper_limit],
        linestyle="--",
        color="#7a7a7a",
        label="Perfect calibration",
    )
    subplot_axes[0, 1].set_title(
        f"Calibration by predicted-price decile: {selected_model_family}"
    )
    subplot_axes[0, 1].set_xlabel("Mean predicted price (USD)")
    subplot_axes[0, 1].set_ylabel("Mean actual price (USD)")
    subplot_axes[0, 1].legend()

    ordered_model_comparison_dataframe = model_comparison_results_dataframe.sort_values(
        "display_order"
    ).set_index("model_family")
    model_display_order = list(ordered_model_comparison_dataframe.index)
    model_colours = [
        MODEL_COLOURS[model_family] for model_family in model_display_order
    ]
    error_metric_comparison = (
        ordered_model_comparison_dataframe[
            [
                "test_median_absolute_error_usd",
                "test_mae_usd",
                "test_rmse_usd",
            ]
        ]
        .transpose()
        .rename(
            index={
                "test_median_absolute_error_usd": "Median absolute error",
                "test_mae_usd": "Mean absolute error",
                "test_rmse_usd": "RMSE",
            }
        )
    )
    error_metric_comparison.plot(
        kind="barh",
        ax=subplot_axes[1, 0],
        color=model_colours,
    )
    subplot_axes[1, 0].set_title("Held-out error across regression models")
    subplot_axes[1, 0].set_xlabel("Prediction error (USD per night)")
    subplot_axes[1, 0].set_ylabel("")
    subplot_axes[1, 0].legend(
        title="Model",
        fontsize=8,
    )

    coefficient_of_determination_comparison = (
        ordered_model_comparison_dataframe[["test_r2_log_price", "test_r2_price_usd"]]
        .transpose()
        .rename(
            index={
                "test_r2_log_price": "Log price",
                "test_r2_price_usd": "Price in USD",
            }
        )
    )
    coefficient_of_determination_comparison.plot(
        kind="barh",
        ax=subplot_axes[1, 1],
        color=model_colours,
    )
    coefficient_of_determination_lower_limit = min(
        0.0,
        coefficient_of_determination_comparison.min().min() - 0.05,
    )
    subplot_axes[1, 1].set_xlim(
        coefficient_of_determination_lower_limit,
        1,
    )
    subplot_axes[1, 1].set_title(
        f"Held-out explained variance; selected: {selected_model_family}"
    )
    subplot_axes[1, 1].set_xlabel("R²")
    subplot_axes[1, 1].set_ylabel("")
    subplot_axes[1, 1].legend(
        title="Model",
        fontsize=8,
    )
    save_standalone_figures(
        visualisation_figures,
        [
            "19_actual_versus_predicted_price.png",
            "20_predicted_price_calibration.png",
            "21_prediction_error_model_comparison.png",
            "22_explained_variance_model_comparison.png",
        ],
    )


def plot_model_selection(
    model_comparison_results_dataframe: pandas.DataFrame,
) -> None:
    """Plot the training-only cross-validated model-selection result."""
    search_results_dataframe = (
        model_comparison_results_dataframe[
            model_comparison_results_dataframe["model_family"].isin(MODEL_FAMILY_ORDER)
        ]
        .set_index("model_family")
        .reindex(MODEL_FAMILY_ORDER)
        .reset_index()
    )
    model_colours = [
        MODEL_COLOURS[model_family]
        for model_family in search_results_dataframe["model_family"]
    ]
    visualisation_figure, subplot_axis = pyplot.subplots(figsize=(10, 7))
    bars = subplot_axis.bar(
        search_results_dataframe["model_family"],
        search_results_dataframe["best_cv_rmse_usd"],
        yerr=search_results_dataframe["best_cv_rmse_std_usd"],
        capsize=5,
        color=model_colours,
    )
    selected_model_index = int(
        search_results_dataframe["is_selected"].to_numpy().nonzero()[0][0]
    )
    bars[selected_model_index].set_edgecolor("#c94845")
    bars[selected_model_index].set_linewidth(2.5)
    subplot_axis.set_title("GridSearchCV model selection on training data")
    subplot_axis.set_ylabel("Best cross-validated RMSE (USD)")
    subplot_axis.set_xlabel("")
    label_bars(subplot_axis, suffix=" USD", decimals=1)
    subplot_axis.text(
        selected_model_index,
        search_results_dataframe.loc[
            selected_model_index,
            "best_cv_rmse_usd",
        ]
        + search_results_dataframe.loc[
            selected_model_index,
            "best_cv_rmse_std_usd",
        ]
        + 2,
        "Selected",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=10,
        fontweight="bold",
        color="#c94845",
    )
    seaborn.despine(visualisation_figure)
    visualisation_figure.tight_layout(pad=1.2)
    save_figure(
        visualisation_figure,
        "39_cross_validated_rmse_by_model_family.png",
    )


def readable_feature_name(feature_name: str) -> str:
    """Convert an encoded model-feature name into a report-ready label."""
    feature_name_replacements = {
        "dist_meters_to_subway": "Distance to subway",
        "dist_min_poi_meters": "Distance to visitor destination",
        "calculated_host_listings_count": "Host listing count",
        "reviews_per_month": "Reviews per month",
        "number_of_reviews": "Review count",
        "minimum_nights": "Minimum nights",
        "availability_365": "Annual availability",
        "room_type_": "Room: ",
        "boroname_": "Borough: ",
        "latitude": "Latitude",
        "longitude": "Longitude",
    }
    for source, target in feature_name_replacements.items():
        if feature_name == source or feature_name.startswith(source):
            return feature_name.replace(source, target, 1)
    return feature_name.replace("_", " ").title()


def plot_model_diagnostics(
    testing_predictions_dataframe: pandas.DataFrame,
    feature_importance_series: pandas.Series,
    selected_model_family: str,
) -> None:
    """Create standalone feature-importance and diagnostic visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    top_feature_importances_series = feature_importance_series.head(14).sort_values()
    subplot_axes[0, 0].barh(
        [
            readable_feature_name(feature_name)
            for feature_name in top_feature_importances_series.index
        ],
        top_feature_importances_series.values,
        color="#4c78a8",
    )
    subplot_axes[0, 0].set_title(
        f"Permutation importance on held-out data: {selected_model_family}"
    )
    subplot_axes[0, 0].set_xlabel("Increase in log-price MAE after permutation")
    residual_limit = float(
        testing_predictions_dataframe["residual"].abs().quantile(0.99)
    )
    testing_predictions_with_central_residuals_dataframe = (
        testing_predictions_dataframe[
            testing_predictions_dataframe["residual"].abs() <= residual_limit
        ]
    )
    seaborn.histplot(
        testing_predictions_with_central_residuals_dataframe["residual"],
        bins=60,
        color="#5b8e7d",
        ax=subplot_axes[0, 1],
    )
    subplot_axes[0, 1].axvline(0, linestyle="--", color="#7a7a7a")
    subplot_axes[0, 1].set_title("Residual distribution (central 98%)")
    subplot_axes[0, 1].set_xlabel("Actual minus predicted price (USD)")
    prediction_decile_dataframe = testing_predictions_dataframe.copy()
    prediction_decile_dataframe["decile"] = pandas.qcut(
        prediction_decile_dataframe["predicted_price"], q=10, duplicates="drop"
    )
    residuals_by_prediction_decile_dataframe = (
        prediction_decile_dataframe.groupby("decile", observed=True)
        .agg(predicted=("predicted_price", "median"), residual=("residual", "median"))
        .reset_index(drop=True)
    )
    subplot_axes[1, 0].plot(
        residuals_by_prediction_decile_dataframe["predicted"],
        residuals_by_prediction_decile_dataframe["residual"],
        marker="o",
        color="#c94845",
    )
    subplot_axes[1, 0].axhline(0, linestyle="--", color="#7a7a7a")
    subplot_axes[1, 0].set_title("Median residual by predicted-price decile")
    subplot_axes[1, 0].set_xlabel("Median predicted price (USD)")
    subplot_axes[1, 0].set_ylabel("Median residual (USD)")
    actual_price_band_dataframe = testing_predictions_dataframe.copy()
    actual_price_band_dataframe["band"] = pandas.qcut(
        actual_price_band_dataframe["actual_price"], q=5, duplicates="drop"
    )
    actual_price_band_dataframe = (
        actual_price_band_dataframe.groupby("band", observed=True)
        .agg(actual=("actual_price", "median"), mae=("absolute_error", "mean"))
        .reset_index(drop=True)
    )
    subplot_axes[1, 1].plot(
        actual_price_band_dataframe["actual"],
        actual_price_band_dataframe["mae"],
        marker="o",
        color="#f2a65a",
    )
    subplot_axes[1, 1].set_title("Mean absolute error by actual-price quintile")
    subplot_axes[1, 1].set_xlabel("Median actual price in quintile (USD)")
    subplot_axes[1, 1].set_ylabel("Mean absolute error (USD)")
    save_standalone_figures(
        visualisation_figures,
        [
            "23_permutation_feature_importance.png",
            "24_residual_distribution.png",
            "25_median_residual_by_predicted_price_decile.png",
            "26_mean_absolute_error_by_actual_price_quintile.png",
        ],
    )


def plot_cluster_profiles(
    cleaned_listings_dataframe: pandas.DataFrame,
    cluster_summary_dataframe: pandas.DataFrame,
    neighbourhood_tabulation_areas_geodataframe: geopandas.GeoDataFrame,
) -> None:
    """Create the stable clustering visualisations used by the report."""
    visualisation_figures, subplot_axes = create_standalone_figures(
        1, 7, figure_size=(10, 7)
    )
    ordered_cluster_summary_dataframe = cluster_summary_dataframe.set_index(
        "cluster"
    ).reindex(CLUSTER_ORDER)
    cluster_labels = [
        cluster_display_label(cluster, multiline=True) for cluster in CLUSTER_ORDER
    ]
    cluster_colours = [CLUSTER_COLOURS[cluster] for cluster in CLUSTER_ORDER]

    subplot_axes[0].bar(
        cluster_labels,
        ordered_cluster_summary_dataframe["number_of_listings"],
        color=cluster_colours,
    )
    subplot_axes[0].set_title("Listings in each K-means cluster")
    subplot_axes[0].set_ylabel("Listings")
    label_bars(subplot_axes[0])

    subplot_axes[1].bar(
        cluster_labels,
        ordered_cluster_summary_dataframe["price"],
        color=cluster_colours,
    )
    subplot_axes[1].set_title("Average nightly price by cluster")
    subplot_axes[1].set_ylabel("Average price (USD)")
    label_bars(subplot_axes[1], suffix=" USD", decimals=1)

    cluster_profile_dataframe = ordered_cluster_summary_dataframe[CLUSTERING_FEATURES]
    standardised_cluster_profile_dataframe = (
        cluster_profile_dataframe - cluster_profile_dataframe.mean()
    ) / cluster_profile_dataframe.std(ddof=0).replace(0, numpy.nan)
    standardised_cluster_profile_dataframe.index = cluster_labels
    standardised_cluster_profile_dataframe.columns = [
        readable_feature_name(feature_name)
        for feature_name in standardised_cluster_profile_dataframe.columns
    ]
    seaborn.heatmap(
        standardised_cluster_profile_dataframe,
        annot=True,
        fmt=".1f",
        cmap="vlag",
        center=0,
        cbar_kws={"label": "Standard deviations from the feature mean"},
        ax=subplot_axes[2],
    )
    subplot_axes[2].set_title("Standardised cluster profiles")
    subplot_axes[2].set_xlabel("")
    subplot_axes[2].set_ylabel("")
    subplot_axes[2].tick_params(axis="y", labelrotation=0)

    room_type_composition_dataframe = 100 * pandas.crosstab(
        cleaned_listings_dataframe["cluster"],
        cleaned_listings_dataframe["room_type"],
        normalize="index",
    ).reindex(CLUSTER_ORDER).fillna(0)
    room_type_composition_dataframe.index = cluster_labels
    room_type_composition_dataframe.plot(
        kind="bar",
        stacked=True,
        color=seaborn.color_palette(
            "Set2", n_colors=len(room_type_composition_dataframe.columns)
        ),
        ax=subplot_axes[3],
    )
    subplot_axes[3].set_title("Room-type composition by cluster")
    subplot_axes[3].set_xlabel("")
    subplot_axes[3].set_ylabel("Listings (%)")
    subplot_axes[3].tick_params(axis="x", rotation=0)
    subplot_axes[3].legend(
        title="Room type",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )

    borough_composition_dataframe = 100 * pandas.crosstab(
        cleaned_listings_dataframe["cluster"],
        cleaned_listings_dataframe["boroname"],
        normalize="index",
    ).reindex(CLUSTER_ORDER).fillna(0)
    borough_composition_dataframe.index = cluster_labels
    borough_composition_dataframe.plot(
        kind="bar",
        stacked=True,
        color=seaborn.color_palette(
            "tab10", n_colors=len(borough_composition_dataframe.columns)
        ),
        ax=subplot_axes[4],
    )
    subplot_axes[4].set_title("Borough composition by cluster")
    subplot_axes[4].set_xlabel("")
    subplot_axes[4].set_ylabel("Listings (%)")
    subplot_axes[4].tick_params(axis="x", rotation=0)
    subplot_axes[4].legend(
        title="Borough",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )

    plotting_sample_dataframe = cleaned_listings_dataframe.sample(
        n=min(15000, len(cleaned_listings_dataframe)),
        random_state=CLUSTER_RANDOM_STATE,
    )
    for cluster in CLUSTER_ORDER:
        cluster_sample_dataframe = plotting_sample_dataframe[
            plotting_sample_dataframe["cluster"] == cluster
        ]
        subplot_axes[5].scatter(
            cluster_sample_dataframe["pca_1"],
            cluster_sample_dataframe["pca_2"],
            s=9,
            alpha=0.35,
            color=CLUSTER_COLOURS[cluster],
            label=cluster_display_label(cluster),
            linewidths=0,
        )
    subplot_axes[5].set_title("Two-dimensional projection of cluster membership")
    subplot_axes[5].set_xlabel("Principal component 1")
    subplot_axes[5].set_ylabel("Principal component 2")
    subplot_axes[5].legend(title="Cluster")

    neighbourhood_tabulation_areas_geodataframe.boundary.plot(
        color="#bdbdbd", linewidth=0.4, ax=subplot_axes[6]
    )
    for cluster in CLUSTER_ORDER:
        cluster_sample_dataframe = plotting_sample_dataframe[
            plotting_sample_dataframe["cluster"] == cluster
        ]
        subplot_axes[6].scatter(
            cluster_sample_dataframe["longitude"],
            cluster_sample_dataframe["latitude"],
            s=5,
            alpha=0.4,
            color=CLUSTER_COLOURS[cluster],
            label=cluster_display_label(cluster),
            linewidths=0,
        )
    subplot_axes[6].set_title("Geographic distribution of listing clusters")
    subplot_axes[6].set_axis_off()
    subplot_axes[6].legend(title="Cluster", markerscale=2)

    save_standalone_figures(
        visualisation_figures,
        [
            "27_cluster_listing_counts.png",
            "28_average_price_by_cluster.png",
            "29_standardised_cluster_profiles.png",
            "30_room_type_composition_by_cluster.png",
            "31_borough_composition_by_cluster.png",
            "32_pca_cluster_projection.png",
            "33_geographic_cluster_distribution.png",
        ],
    )


def build_volatility_metrics(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """Summarise out-of-fold prediction error by NTA and cluster."""
    neighbourhood_volatility_metrics_dataframe = (
        cleaned_listings_dataframe.groupby(["boroname", "ntaname"], observed=True)
        .agg(
            listings=("id", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            median_actual_price=("price", "median"),
            median_predicted_price=("predicted_price", "median"),
        )
        .reset_index()
    )
    cluster_volatility_metrics_dataframe = (
        cleaned_listings_dataframe.groupby("cluster", observed=True)
        .agg(
            listings=("id", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            median_actual_price=("price", "median"),
            median_predicted_price=("predicted_price", "median"),
        )
        .reindex(CLUSTER_ORDER)
        .reset_index()
    )
    cluster_volatility_metrics_dataframe.insert(
        1,
        "cluster_name",
        cluster_volatility_metrics_dataframe["cluster"].map(CLUSTER_NAMES),
    )
    market_segment_volatility_metrics_dataframe = (
        cleaned_listings_dataframe.groupby(["boroname", "room_type"], observed=True)
        .agg(
            listings=("id", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
        )
        .reset_index()
    )
    return (
        neighbourhood_volatility_metrics_dataframe,
        cluster_volatility_metrics_dataframe,
        market_segment_volatility_metrics_dataframe,
    )


def plot_model_volatility(
    neighbourhood_tabulation_areas_geodataframe: geopandas.GeoDataFrame,
    neighbourhood_volatility_metrics_dataframe: pandas.DataFrame,
    cluster_volatility_metrics_dataframe: pandas.DataFrame,
) -> None:
    """Create cluster and neighbourhood views of model uncertainty."""
    visualisation_figures, subplot_axes = create_standalone_figures(
        1, 4, figure_size=(11, 8)
    )
    cluster_labels = [
        cluster_display_label(int(cluster), multiline=True)
        for cluster in cluster_volatility_metrics_dataframe["cluster"]
    ]
    cluster_colours = [
        CLUSTER_COLOURS[int(cluster)]
        for cluster in cluster_volatility_metrics_dataframe["cluster"]
    ]
    subplot_axes[0].bar(
        cluster_labels,
        cluster_volatility_metrics_dataframe["mean_absolute_error"],
        color=cluster_colours,
    )
    subplot_axes[0].set_title("Cold-start model error by listing cluster")
    subplot_axes[0].set_ylabel("Mean absolute error (USD)")
    label_bars(subplot_axes[0], suffix=" USD", decimals=1)

    spatial_volatility_metrics_geodataframe = (
        neighbourhood_tabulation_areas_geodataframe.merge(
            neighbourhood_volatility_metrics_dataframe,
            on=["boroname", "ntaname"],
            how="left",
        )
    )
    reliable_spatial_volatility_metrics_geodataframe = (
        spatial_volatility_metrics_geodataframe[
            spatial_volatility_metrics_geodataframe["listings"] >= 50
        ]
    )
    spatial_volatility_metrics_geodataframe.plot(
        color="#e5e5e5",
        edgecolor="white",
        linewidth=0.25,
        ax=subplot_axes[1],
    )
    reliable_spatial_volatility_metrics_geodataframe.plot(
        column="mean_absolute_error",
        cmap="YlOrRd",
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Mean absolute error (USD)", "shrink": 0.7},
        ax=subplot_axes[1],
    )
    subplot_axes[1].set_title("Cold-start price uncertainty by NTA")
    subplot_axes[1].set_axis_off()

    highest_volatility_neighbourhoods_dataframe = (
        neighbourhood_volatility_metrics_dataframe[
            neighbourhood_volatility_metrics_dataframe["listings"] >= 100
        ]
        .nlargest(12, "mean_absolute_error")
        .sort_values("mean_absolute_error")
    )
    subplot_axes[2].barh(
        highest_volatility_neighbourhoods_dataframe["ntaname"],
        highest_volatility_neighbourhoods_dataframe["mean_absolute_error"],
        color="#c94845",
    )
    subplot_axes[2].set_title("Highest-error NTAs with at least 100 listings")
    subplot_axes[2].set_xlabel("Mean absolute error (USD)")
    label_bars(subplot_axes[2], suffix=" USD", decimals=1)

    cluster_price_comparison_dataframe = (
        cluster_volatility_metrics_dataframe.set_index("cluster")[
            ["median_actual_price", "median_predicted_price"]
        ]
        .rename(
            columns={
                "median_actual_price": "Actual median",
                "median_predicted_price": "Predicted median",
            }
        )
        .reindex(CLUSTER_ORDER)
    )
    cluster_price_comparison_dataframe.index = [
        cluster_display_label(cluster, multiline=True) for cluster in CLUSTER_ORDER
    ]
    cluster_price_comparison_dataframe.plot(
        kind="bar",
        color=["#4c78a8", "#f2a541"],
        ax=subplot_axes[3],
    )
    subplot_axes[3].set_title("Actual and predicted median price by cluster")
    subplot_axes[3].set_xlabel("")
    subplot_axes[3].set_ylabel("Nightly price (USD)")
    subplot_axes[3].tick_params(axis="x", rotation=0)
    subplot_axes[3].legend(title="")
    label_bars(subplot_axes[3], suffix=" USD", decimals=1)

    save_standalone_figures(
        visualisation_figures,
        [
            "34_model_mae_by_cluster.png",
            "35_model_uncertainty_by_nta.png",
            "36_highest_error_ntas.png",
            "37_actual_vs_predicted_price_by_cluster.png",
        ],
    )


def plot_model_error_by_market_segment(
    market_segment_volatility_metrics_dataframe: pandas.DataFrame,
) -> None:
    """Plot cold-start model error by borough and room type."""
    room_type_order = ["Entire home/apt", "Private room", "Shared room"]
    mean_absolute_error_matrix = (
        market_segment_volatility_metrics_dataframe.pivot(
            index="boroname",
            columns="room_type",
            values="mean_absolute_error",
        )
        .reindex(columns=room_type_order)
        .sort_index()
    )
    listing_count_matrix = market_segment_volatility_metrics_dataframe.pivot(
        index="boroname",
        columns="room_type",
        values="listings",
    ).reindex(
        index=mean_absolute_error_matrix.index,
        columns=mean_absolute_error_matrix.columns,
    )
    annotation_matrix = pandas.DataFrame(
        "",
        index=mean_absolute_error_matrix.index,
        columns=mean_absolute_error_matrix.columns,
    )
    for borough_name in mean_absolute_error_matrix.index:
        for room_type in mean_absolute_error_matrix.columns:
            mean_absolute_error_value = mean_absolute_error_matrix.loc[
                borough_name, room_type
            ]
            listing_count = listing_count_matrix.loc[borough_name, room_type]
            if pandas.notna(mean_absolute_error_value):
                annotation_matrix.loc[borough_name, room_type] = (
                    f"${mean_absolute_error_value:,.1f}\n" f"(n={listing_count:,.0f})"
                )
    visualisation_figure, subplot_axis = pyplot.subplots(figsize=(10, 7))
    seaborn.heatmap(
        mean_absolute_error_matrix,
        annot=annotation_matrix,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean absolute error (USD)"},
        ax=subplot_axis,
    )
    subplot_axis.set_title("Cold-start model error by borough and room type")
    subplot_axis.set_xlabel("")
    subplot_axis.set_ylabel("")
    visualisation_figure.tight_layout(pad=1.2)
    save_figure(
        visualisation_figure,
        "38_model_mae_by_borough_and_room_type.png",
    )


def export_metrics(
    data_retention_metrics: dict[str, int | float],
    model_evaluation_metrics: dict[str, object],
    model_comparison_results_dataframe: pandas.DataFrame,
    grid_search_results_dataframe: pandas.DataFrame,
    cluster_summary_dataframe: pandas.DataFrame,
    neighbourhood_volatility_metrics_dataframe: pandas.DataFrame,
    cluster_volatility_metrics_dataframe: pandas.DataFrame,
    market_segment_volatility_metrics_dataframe: pandas.DataFrame,
) -> None:
    """Export summary metrics and supporting tables beside the PNG reports."""
    combined_cluster_metrics_dataframe = cluster_summary_dataframe.merge(
        cluster_volatility_metrics_dataframe[
            ["cluster", "mean_absolute_error", "median_absolute_error"]
        ],
        on="cluster",
        how="left",
    )
    exported_metrics_dictionary = {
        "data": {
            dictionary_key: (
                int(value) if dictionary_key != "price_cap" else float(value)
            )
            for dictionary_key, value in data_retention_metrics.items()
        },
        "model": {
            dictionary_key: value
            for dictionary_key, value in model_evaluation_metrics.items()
            if not dictionary_key.startswith("_")
        },
        "clustering": {
            str(int(cluster_record["cluster"])): {
                "name": str(cluster_record["cluster_name"]),
                "listings": int(cluster_record["number_of_listings"]),
                "average_price_usd": float(cluster_record["price"]),
                "mean_absolute_error_usd": float(cluster_record["mean_absolute_error"]),
                "median_absolute_error_usd": float(
                    cluster_record["median_absolute_error"]
                ),
            }
            for cluster_record in combined_cluster_metrics_dataframe.to_dict(
                orient="records"
            )
        },
    }
    (VISUALISATION_OUTPUT_DIRECTORY / "metrics_summary.json").write_text(
        json.dumps(exported_metrics_dictionary, indent=2), encoding="utf-8"
    )
    cluster_summary_dataframe.sort_values("cluster").to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "cluster_summary.csv", index=False
    )
    cluster_volatility_metrics_dataframe.sort_values("cluster").to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "cluster_volatility_metrics.csv",
        index=False,
    )
    neighbourhood_volatility_metrics_dataframe.sort_values(
        "listings", ascending=False
    ).to_csv(VISUALISATION_OUTPUT_DIRECTORY / "nta_volatility_metrics.csv", index=False)
    market_segment_volatility_metrics_dataframe.sort_values(
        ["boroname", "room_type"]
    ).to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "segment_volatility_metrics.csv",
        index=False,
    )
    model_comparison_results_dataframe.to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "model_comparison_metrics.csv",
        index=False,
    )
    grid_search_results_dataframe.sort_values(
        ["model_family", "rank_test_score"]
    ).to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "grid_search_results.csv",
        index=False,
    )


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate a reproducible SHA-256 digest for one file."""
    sha256_digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for file_chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            sha256_digest.update(file_chunk)
    return sha256_digest.hexdigest()


def current_git_state() -> dict[str, str | bool]:
    """Return the current Git revision and whether tracked work is uncommitted."""
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "commit": revision_result.stdout.strip(),
        "tracked_worktree_dirty": bool(status_result.stdout.strip()),
    }


def build_report_figure_specifications(
    data_retention_metrics: dict[str, int | float],
    model_evaluation_metrics: dict[str, object],
    cluster_summary_dataframe: pandas.DataFrame,
) -> list[dict[str, object]]:
    """Define the selected report figures, captions, and alternative text."""
    raw_listing_count = int(data_retention_metrics["Raw listings"])
    final_listing_count = int(
        data_retention_metrics["Minimum stay of 365 days or fewer"]
    )
    retained_listing_percentage = 100 * final_listing_count / raw_listing_count
    price_cap = float(data_retention_metrics["price_cap"])
    selected_model_family = str(model_evaluation_metrics["winner_model_family"])
    selected_model_mae = float(model_evaluation_metrics["mae_usd"])
    median_baseline_mae = float(model_evaluation_metrics["median_baseline_mae_usd"])
    mae_improvement_percentage = (
        100 * (median_baseline_mae - selected_model_mae) / median_baseline_mae
    )
    ordered_cluster_summary_dataframe = cluster_summary_dataframe.set_index(
        "cluster"
    ).reindex(CLUSTER_ORDER)
    cluster_counts = {
        cluster: int(
            ordered_cluster_summary_dataframe.loc[cluster, "number_of_listings"]
        )
        for cluster in CLUSTER_ORDER
    }
    cluster_shares = {
        cluster: 100 * cluster_counts[cluster] / final_listing_count
        for cluster in CLUSTER_ORDER
    }
    return [
        {
            "figure_number": 1,
            "section": "Data and methods",
            "source_filename": "02_cleaning_and_geographic_join_retention.png",
            "report_filename": "figure_01_data_retention.png",
            "title": "Listings retained through data preparation",
            "caption": (
                f"The chart traces listing retention through the geographic join "
                f"and cleaning rules. The final analytical sample contains "
                f"{final_listing_count:,} of {raw_listing_count:,} source listings "
                f"({retained_listing_percentage:.1f}%), showing that the rules remove "
                f"few observations while addressing invalid and extreme values. "
                f"This establishes the coverage of the evidence used later in the "
                f"analysis. Retention depends on the official NTA boundaries, the "
                f"USD {price_cap:,.0f} 99th-percentile price cap, and the 365-night "
                f"minimum-stay ceiling."
            ),
            "alt_text": (
                f"Horizontal bars show {raw_listing_count:,} raw listings declining "
                f"slightly to {final_listing_count:,} cleaned listings after the NTA "
                f"join, price filtering, and minimum-stay filtering."
            ),
        },
        {
            "figure_number": 2,
            "section": "Market overview",
            "source_filename": "07_median_price_by_borough_and_room_type.png",
            "report_filename": "figure_02_segment_median_prices.png",
            "title": "Median nightly price by borough and room type",
            "caption": (
                "The chart compares median nightly prices across borough and "
                "room-type combinations. Entire homes or apartments command the "
                "highest median price in every borough, with Manhattan forming the "
                "highest-priced broad market. This supports location and room type "
                "as essential controls in the pricing model and as practical inputs "
                "for new hosts. These are unadjusted medians and therefore do not "
                "separate location effects from correlated listing characteristics."
            ),
            "alt_text": (
                "Heatmap rows represent New York City boroughs and columns represent "
                "shared rooms, private rooms, and entire homes or apartments. "
                "Manhattan entire homes have the darkest, highest-price cell."
            ),
        },
        {
            "figure_number": 3,
            "section": "Market overview",
            "source_filename": "10_median_price_by_nta.png",
            "report_filename": "figure_03_neighbourhood_median_prices.png",
            "title": "Median nightly price by neighbourhood",
            "caption": (
                "The choropleth maps median nightly price across official NTAs. "
                "Higher-price areas concentrate in Manhattan and selected nearby "
                "districts, demonstrating substantial price variation within as "
                "well as between boroughs. This supports using granular location "
                "information in the cold-start model. Only NTAs with at least 20 "
                "listings are coloured; the map is descriptive and does not control "
                "for room type or listing quality."
            ),
            "alt_text": (
                "Map of New York City neighbourhoods shaded by median Airbnb price, "
                "with darker high-price areas concentrated mainly in Manhattan and "
                "low-volume neighbourhoods shown in grey."
            ),
        },
        {
            "figure_number": 4,
            "section": "Model evaluation",
            "source_filename": "21_prediction_error_model_comparison.png",
            "report_filename": "figure_04_model_vs_baseline.png",
            "title": "Held-out error across regression models",
            "caption": (
                f"The chart compares held-out prediction errors for three tuned "
                f"regression families and the neighbourhood/room-type median baseline. "
                f"{selected_model_family} was selected using the lowest three-fold "
                f"cross-validated RMSE on the training data. It "
                f"reduces MAE from USD {median_baseline_mae:.2f} to USD "
                f"{selected_model_mae:.2f}, an improvement of "
                f"{mae_improvement_percentage:.1f}%. This demonstrates whether the "
                f"selected model adds predictive value beyond a transparent market "
                f"lookup. The held-out evaluation cannot measure the effect of "
                f"unobserved amenities, photographs, or listing quality."
            ),
            "alt_text": (
                f"Grouped horizontal bars compare median absolute error, mean absolute "
                f"error, and RMSE for Decision Tree, Random Forest, Gradient Boosting, "
                f"and a neighbourhood median baseline. The selected "
                f"{selected_model_family} has held-out MAE USD "
                f"{selected_model_mae:.2f}, compared with baseline MAE USD "
                f"{median_baseline_mae:.2f}."
            ),
        },
        {
            "figure_number": 5,
            "section": "Market segmentation",
            "source_filename": "27_cluster_listing_counts.png",
            "report_filename": "figure_05_cluster_sizes.png",
            "title": "Listing counts by market cluster",
            "caption": (
                f"The chart shows the size of the three K-means segments. "
                f"{cluster_display_label(0)} contains {cluster_counts[0]:,} listings "
                f"({cluster_shares[0]:.1f}%), {cluster_display_label(1)} contains "
                f"{cluster_counts[1]:,} ({cluster_shares[1]:.1f}%), and "
                f"{cluster_display_label(2)} contains {cluster_counts[2]:,} "
                f"({cluster_shares[2]:.1f}%). The imbalance shows that the premium "
                f"professional segment is distinctive but small, which matters when "
                f"interpreting aggregate market findings. Cluster names are provisional "
                f"interpretations of an unsupervised model rather than known market labels."
            ),
            "alt_text": (
                f"Three bars show {cluster_counts[0]:,} mainstream listings, "
                f"{cluster_counts[1]:,} premium professional listings, and "
                f"{cluster_counts[2]:,} lower-price review-active listings."
            ),
        },
        {
            "figure_number": 6,
            "section": "Market segmentation",
            "source_filename": "29_standardised_cluster_profiles.png",
            "report_filename": "figure_06_cluster_profiles.png",
            "title": "Standardised characteristics of the market clusters",
            "caption": (
                "The heatmap compares each cluster with the overall feature means. "
                "The premium professional cluster combines higher price, longer "
                "minimum stays, larger host portfolios, and high availability, while "
                "the lower-price review-active cluster has stronger review activity "
                "and lower prices. This converts the numerical K-means output into "
                "business-facing segment profiles. Standardised values show relative "
                "differences, not causal effects, and extreme variables can influence "
                "the fitted centroids."
            ),
            "alt_text": (
                "Heatmap with three cluster rows and eight feature columns; red cells "
                "indicate above-average values and blue cells below-average values. "
                "The premium professional row is high on price and host activity, "
                "while the lower-price review-active row is high on reviews."
            ),
        },
        {
            "figure_number": 7,
            "section": "Market segmentation",
            "source_filename": "30_room_type_composition_by_cluster.png",
            "report_filename": "figure_07_cluster_room_types.png",
            "title": "Room-type composition of each market cluster",
            "caption": (
                "The stacked bars show how room types are distributed within each "
                "cluster. The premium professional segment is dominated by entire "
                "homes or apartments, whereas the two larger segments contain more "
                "balanced mixes of entire homes and private rooms. This helps translate "
                "cluster membership into listing formats that hosts and analysts can "
                "recognise. Percentages describe composition only and do not establish "
                "that room type caused a listing to enter a cluster."
            ),
            "alt_text": (
                "Three stacked bars show room-type percentages by cluster. The premium "
                "professional cluster is mostly entire homes or apartments; the other "
                "clusters contain larger private-room shares."
            ),
        },
        {
            "figure_number": 8,
            "section": "Market segmentation",
            "source_filename": "33_geographic_cluster_distribution.png",
            "report_filename": "figure_08_cluster_geography.png",
            "title": "Geographic distribution of the market clusters",
            "caption": (
                "The point map plots a reproducible sample of listings by cluster. "
                "The premium professional segment is concentrated in Manhattan, while "
                "the mainstream and lower-price review-active segments extend more "
                "widely across the city. This demonstrates that the segmentation adds "
                "geographic structure without being defined by borough alone. The map "
                "uses a 15,000-listing display sample to control overplotting and should "
                "not be used to infer precise local prevalence."
            ),
            "alt_text": (
                "New York City map with sampled listing points coloured and labelled "
                "by cluster; premium professional points concentrate in Manhattan, "
                "while the other clusters are more geographically dispersed."
            ),
        },
        {
            "figure_number": 9,
            "section": "Model evaluation",
            "source_filename": "35_model_uncertainty_by_nta.png",
            "report_filename": "figure_09_neighbourhood_model_uncertainty.png",
            "title": "Cold-start model uncertainty by neighbourhood",
            "caption": (
                "The choropleth maps mean absolute out-of-fold prediction error by "
                "NTA. Error is highest in several central, high-price neighbourhoods, "
                "indicating where omitted qualitative information may matter most. "
                "This provides a geographic risk indicator for using the cold-start "
                "valuation tool. Only NTAs with at least 50 listings are coloured, and "
                "high error represents model uncertainty rather than incorrect host pricing."
            ),
            "alt_text": (
                "Map of New York City neighbourhoods shaded from light yellow to dark "
                "red by mean absolute prediction error; several central Manhattan "
                "neighbourhoods are darkest, and low-volume areas are grey."
            ),
        },
    ]


def render_report_visualisation_pack(
    data_retention_metrics: dict[str, int | float],
    model_evaluation_metrics: dict[str, object],
    cluster_summary_dataframe: pandas.DataFrame,
) -> None:
    """Render the selected figures and their reproducibility documentation."""
    report_figure_specifications = build_report_figure_specifications(
        data_retention_metrics,
        model_evaluation_metrics,
        cluster_summary_dataframe,
    )
    REPORT_VISUALISATION_DIRECTORY.mkdir(exist_ok=True)
    generated_metadata_filenames = [
        "captions_and_alt_text.md",
        "figure_catalogue.json",
        "reproducibility_manifest.json",
    ]
    for report_figure_specification in report_figure_specifications:
        report_output_path = REPORT_VISUALISATION_DIRECTORY / str(
            report_figure_specification["report_filename"]
        )
        if report_output_path.exists():
            report_output_path.unlink()
    for generated_metadata_filename in generated_metadata_filenames:
        generated_metadata_path = (
            REPORT_VISUALISATION_DIRECTORY / generated_metadata_filename
        )
        if generated_metadata_path.exists():
            generated_metadata_path.unlink()

    output_figure_records = []
    for report_figure_specification in report_figure_specifications:
        source_visualisation_path = VISUALISATION_OUTPUT_DIRECTORY / str(
            report_figure_specification["source_filename"]
        )
        report_output_path = REPORT_VISUALISATION_DIRECTORY / str(
            report_figure_specification["report_filename"]
        )
        if not source_visualisation_path.exists():
            raise FileNotFoundError(
                f"Required report figure is missing: {source_visualisation_path}"
            )
        source_image = pyplot.imread(source_visualisation_path)
        report_figure = pyplot.figure(
            figsize=REPORT_FIGURE_SIZE_INCHES,
            dpi=REPORT_FIGURE_DPI,
            facecolor="white",
        )
        report_axis = report_figure.add_axes([0, 0, 1, 1])
        report_axis.imshow(source_image)
        report_axis.set_axis_off()
        report_figure.savefig(
            report_output_path,
            dpi=REPORT_FIGURE_DPI,
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
        )
        pyplot.close(report_figure)
        report_image = pyplot.imread(report_output_path)
        output_figure_records.append(
            {
                **report_figure_specification,
                "width_pixels": int(report_image.shape[1]),
                "height_pixels": int(report_image.shape[0]),
                "sha256": calculate_file_sha256(report_output_path),
            }
        )

    catalogue_path = REPORT_VISUALISATION_DIRECTORY / "figure_catalogue.json"
    catalogue_path.write_text(
        json.dumps(
            {
                "figure_count": len(output_figure_records),
                "canvas_pixels": [
                    int(REPORT_FIGURE_SIZE_INCHES[0] * REPORT_FIGURE_DPI),
                    int(REPORT_FIGURE_SIZE_INCHES[1] * REPORT_FIGURE_DPI),
                ],
                "figures": output_figure_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    caption_lines = [
        "# Report visualisations",
        "",
        (
            "The cluster names are provisional business descriptions of the "
            "three-cluster K-means solution."
        ),
        "",
    ]
    for figure_record in output_figure_records:
        caption_lines.extend(
            [
                (
                    f"## Figure {figure_record['figure_number']}: "
                    f"{figure_record['title']}"
                ),
                "",
                f"File: {figure_record['report_filename']}",
                "",
                f"Caption: {figure_record['caption']}",
                "",
                f"Alternative text: {figure_record['alt_text']}",
                "",
            ]
        )
    (REPORT_VISUALISATION_DIRECTORY / "captions_and_alt_text.md").write_text(
        "\n".join(caption_lines) + "\n", encoding="utf-8"
    )

    source_data_files = [
        AIRBNB_SOURCE_CSV_PATH,
        NEIGHBOURHOOD_TABULATION_AREA_GEOJSON_PATH,
        SUBWAY_ENTRANCE_CSV_PATH,
    ]
    reproducibility_manifest = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "git": current_git_state(),
        "code": {
            "generator": {
                "path": "generate_visualisations.py",
                "sha256": calculate_file_sha256(Path(__file__).resolve()),
            },
            "notebook": {
                "path": "main.ipynb",
                "sha256": calculate_file_sha256(
                    REPOSITORY_ROOT_DIRECTORY / "main.ipynb"
                ),
            },
        },
        "source_data": {
            source_data_path.name: {
                "path": str(source_data_path.relative_to(REPOSITORY_ROOT_DIRECTORY)),
                "bytes": source_data_path.stat().st_size,
                "sha256": calculate_file_sha256(source_data_path),
            }
            for source_data_path in source_data_files
        },
        "configuration": {
            "cluster_count": len(CLUSTER_ORDER),
            "cluster_random_seed": CLUSTER_RANDOM_STATE,
            "cluster_names": CLUSTER_NAMES,
            "clustering_features": CLUSTERING_FEATURES,
            "model_random_seed": MODEL_RANDOM_STATE,
            "model_families": MODEL_FAMILY_ORDER,
            "model_parameter_grids": MODEL_PARAMETER_GRIDS,
            "model_selection_cv_folds": MODEL_SELECTION_CV_FOLDS,
            "out_of_fold_estimation_cv_folds": MODEL_ESTIMATION_CV_FOLDS,
            "selected_model_family": model_evaluation_metrics["winner_model_family"],
            "selected_model_parameters": model_evaluation_metrics[
                "winner_best_parameters"
            ],
            "review_history_excluded_from_pricing_model": True,
            "report_canvas_inches": REPORT_FIGURE_SIZE_INCHES,
            "report_dpi": REPORT_FIGURE_DPI,
        },
        "environment": {
            "python": platform.python_version(),
            "geopandas": geopandas.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "seaborn": seaborn.__version__,
        },
        "report_outputs": [
            {
                "figure_number": figure_record["figure_number"],
                "filename": figure_record["report_filename"],
                "source_filename": figure_record["source_filename"],
                "width_pixels": figure_record["width_pixels"],
                "height_pixels": figure_record["height_pixels"],
                "sha256": figure_record["sha256"],
            }
            for figure_record in output_figure_records
        ],
    }
    (REPORT_VISUALISATION_DIRECTORY / "reproducibility_manifest.json").write_text(
        json.dumps(reproducibility_manifest, indent=2),
        encoding="utf-8",
    )

    generated_report_visualisation_count = len(
        list(REPORT_VISUALISATION_DIRECTORY.glob("*.png"))
    )
    if generated_report_visualisation_count != EXPECTED_REPORT_VISUALISATION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REPORT_VISUALISATION_COUNT} report PNGs but "
            f"generated {generated_report_visualisation_count}."
        )


def main() -> None:
    """Generate all reports and machine-readable metric files."""
    ensure_source_files()
    set_plot_style()
    VISUALISATION_OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    clear_generated_outputs()
    source_listings_dataframe = pandas.read_csv(AIRBNB_SOURCE_CSV_PATH)
    (
        cleaned_listings_dataframe,
        neighbourhood_tabulation_areas_geodataframe,
        data_retention_metrics,
    ) = prepare_data()
    cleaned_listings_dataframe, cluster_summary_dataframe = assign_listing_clusters(
        cleaned_listings_dataframe
    )
    cleaned_listings_dataframe, feature_importance_series, model_evaluation_metrics = (
        train_and_score(cleaned_listings_dataframe)
    )
    testing_predictions_dataframe = model_evaluation_metrics.pop("_test_results")
    model_comparison_results_dataframe = model_evaluation_metrics.pop(
        "_model_comparison_results"
    )
    grid_search_results_dataframe = model_evaluation_metrics.pop("_grid_search_results")
    (
        neighbourhood_volatility_metrics_dataframe,
        cluster_volatility_metrics_dataframe,
        market_segment_volatility_metrics_dataframe,
    ) = build_volatility_metrics(cleaned_listings_dataframe)
    plot_data_and_market(
        source_listings_dataframe, cleaned_listings_dataframe, data_retention_metrics
    )
    plot_price_segments(cleaned_listings_dataframe)
    plot_spatial_market(
        cleaned_listings_dataframe, neighbourhood_tabulation_areas_geodataframe
    )
    plot_operational_relationships(cleaned_listings_dataframe)
    plot_host_and_booking_policy_relationships(cleaned_listings_dataframe)
    plot_model_performance(
        testing_predictions_dataframe,
        model_evaluation_metrics,
        model_comparison_results_dataframe,
    )
    plot_model_diagnostics(
        testing_predictions_dataframe,
        feature_importance_series,
        str(model_evaluation_metrics["winner_model_family"]),
    )
    plot_model_selection(model_comparison_results_dataframe)
    plot_cluster_profiles(
        cleaned_listings_dataframe,
        cluster_summary_dataframe,
        neighbourhood_tabulation_areas_geodataframe,
    )
    plot_model_volatility(
        neighbourhood_tabulation_areas_geodataframe,
        neighbourhood_volatility_metrics_dataframe,
        cluster_volatility_metrics_dataframe,
    )
    plot_model_error_by_market_segment(market_segment_volatility_metrics_dataframe)
    export_metrics(
        data_retention_metrics,
        model_evaluation_metrics,
        model_comparison_results_dataframe,
        grid_search_results_dataframe,
        cluster_summary_dataframe,
        neighbourhood_volatility_metrics_dataframe,
        cluster_volatility_metrics_dataframe,
        market_segment_volatility_metrics_dataframe,
    )
    render_report_visualisation_pack(
        data_retention_metrics,
        model_evaluation_metrics,
        cluster_summary_dataframe,
    )
    generated_visualisation_count = len(
        list(VISUALISATION_OUTPUT_DIRECTORY.glob("*.png"))
    )
    if generated_visualisation_count != EXPECTED_VISUALISATION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_VISUALISATION_COUNT} PNGs but generated "
            f"{generated_visualisation_count}."
        )
    print(
        f"Generated {EXPECTED_VISUALISATION_COUNT} standalone PNG visualisations in "
        f"{VISUALISATION_OUTPUT_DIRECTORY}"
    )
    print(
        f"Generated {EXPECTED_REPORT_VISUALISATION_COUNT} report-ready PNG "
        f"visualisations in {REPORT_VISUALISATION_DIRECTORY}"
    )
    print(json.dumps(model_evaluation_metrics, indent=2))


if __name__ == "__main__":
    main()
