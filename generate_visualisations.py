"""Generate standalone market, spatial, pricing-anomaly, and machine-learning visualisations for the NYC Airbnb dataset."""

from __future__ import annotations
import json
import urllib.request
import zipfile
from pathlib import Path
import geopandas
import matplotlib

matplotlib.use("Agg")
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import BallTree

REPOSITORY_ROOT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "dataset"
VISUALISATION_OUTPUT_DIRECTORY = REPOSITORY_ROOT_DIRECTORY / "visualisations"
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
RANDOM_STATE = 11
EARTH_RADIUS_METRES = 6371000
EXPECTED_VISUALISATION_COUNT = 37
STATUS_ORDER = ["underpriced", "normal", "overpriced"]
STATUS_COLOURS = {
    "underpriced": "#157f68",
    "normal": "#7a7a7a",
    "overpriced": "#c94845",
}


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


def clear_generated_png_files() -> None:
    """Remove prior generated PNG files before writing the current suite."""
    VISUALISATION_OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for existing_visualisation_path in VISUALISATION_OUTPUT_DIRECTORY.glob("*.png"):
        existing_visualisation_path.unlink()


def label_bars(
    subplot_axis: pyplot.Axes, *, suffix: str = "", decimals: int = 0
) -> None:
    """Label every bar container using consistent numeric formatting."""
    for bar_container in subplot_axis.containers:
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
        "number_of_reviews",
        "reviews_per_month",
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


def train_and_score(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.Series, dict[str, object]]:
    """Evaluate the model and attach out-of-fold pricing-anomaly estimates."""
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
        random_state=RANDOM_STATE,
    )
    random_forest_regressor = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
    )
    random_forest_regressor.fit(
        training_features_dataframe, training_log_price_target_series
    )
    predicted_log_prices = random_forest_regressor.predict(testing_features_dataframe)
    testing_predictions_dataframe = pandas.DataFrame(
        {
            "actual_price": numpy.expm1(testing_log_price_target_series),
            "predicted_price": numpy.expm1(predicted_log_prices),
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
    median_baseline_model = DummyRegressor(strategy="median")
    median_baseline_model.fit(
        training_features_dataframe, training_log_price_target_series
    )
    median_baseline_prediction_log = median_baseline_model.predict(
        testing_features_dataframe
    )
    median_baseline_predicted_price = numpy.expm1(median_baseline_prediction_log)
    median_baseline_absolute_error = (
        testing_predictions_dataframe["actual_price"] - median_baseline_predicted_price
    ).abs()
    model_evaluation_metrics: dict[str, object] = {
        "r2_log_price": float(
            r2_score(testing_log_price_target_series, predicted_log_prices)
        ),
        "r2_price_usd": float(
            r2_score(
                testing_predictions_dataframe["actual_price"],
                testing_predictions_dataframe["predicted_price"],
            )
        ),
        "mae_usd": float(
            mean_absolute_error(
                testing_predictions_dataframe["actual_price"],
                testing_predictions_dataframe["predicted_price"],
            )
        ),
        "rmse_usd": float(
            mean_squared_error(
                testing_predictions_dataframe["actual_price"],
                testing_predictions_dataframe["predicted_price"],
            )
            ** 0.5
        ),
        "median_absolute_error_usd": float(
            testing_predictions_dataframe["absolute_error"].median()
        ),
        "wape_percent": float(
            100
            * testing_predictions_dataframe["absolute_error"].sum()
            / testing_predictions_dataframe["actual_price"].sum()
        ),
        "test_rows": len(testing_features_dataframe),
        "median_baseline_r2_log_price": float(
            r2_score(testing_log_price_target_series, median_baseline_prediction_log)
        ),
        "median_baseline_r2_price_usd": float(
            r2_score(
                testing_predictions_dataframe["actual_price"],
                median_baseline_predicted_price,
            )
        ),
        "median_baseline_mae_usd": float(median_baseline_absolute_error.mean()),
        "median_baseline_rmse_usd": float(
            mean_squared_error(
                testing_predictions_dataframe["actual_price"],
                median_baseline_predicted_price,
            )
            ** 0.5
        ),
        "median_baseline_median_absolute_error_usd": float(
            median_baseline_absolute_error.median()
        ),
        "median_baseline_wape_percent": float(
            100
            * median_baseline_absolute_error.sum()
            / testing_predictions_dataframe["actual_price"].sum()
        ),
    }
    permutation_importance_sample_dataframe = testing_features_dataframe.sample(
        n=min(5000, len(testing_features_dataframe)), random_state=RANDOM_STATE
    )
    permutation_importance_result = permutation_importance(
        random_forest_regressor,
        permutation_importance_sample_dataframe,
        testing_log_price_target_series.loc[
            permutation_importance_sample_dataframe.index
        ],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance_series = pandas.Series(
        permutation_importance_result.importances_mean,
        index=model_features_dataframe.columns,
    ).sort_values(ascending=False)
    out_of_fold_predicted_log_prices = cross_val_predict(
        RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        model_features_dataframe,
        log_price_target_series,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
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
    cleaned_listings_dataframe["price_gap_percent"] = (
        100
        * cleaned_listings_dataframe["residual_usd"]
        / cleaned_listings_dataframe["predicted_price"]
    )
    cleaned_listings_dataframe["status"] = "normal"
    cleaned_listings_dataframe.loc[
        cleaned_listings_dataframe["price_gap_percent"] > 50, "status"
    ] = "overpriced"
    cleaned_listings_dataframe.loc[
        cleaned_listings_dataframe["price_gap_percent"] < -50, "status"
    ] = "underpriced"
    cleaned_listings_dataframe["status"] = pandas.Categorical(
        cleaned_listings_dataframe["status"], categories=STATUS_ORDER, ordered=True
    )
    model_evaluation_metrics["_test_results"] = testing_predictions_dataframe
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
    spatial_market_metrics_geodataframe.plot(
        column="listings",
        cmap="OrRd",
        linewidth=0.25,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": "Listings per NTA", "shrink": 0.72},
        ax=subplot_axes[0],
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
) -> None:
    """Create standalone model-performance and baseline visualisations."""
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
        xlim=(0, comparison_axis_upper_limit), ylim=(0, comparison_axis_upper_limit)
    )
    subplot_axes[0, 0].set_title("Actual versus predicted nightly price")
    subplot_axes[0, 0].set_xlabel("Actual price (USD)")
    subplot_axes[0, 0].set_ylabel("Predicted price (USD)")
    visualisation_figures[0].colorbar(
        hexagonal_density_plot, ax=subplot_axes[0, 0], label="Log listing density"
    )
    calibration_by_prediction_decile_dataframe = testing_predictions_dataframe.copy()
    calibration_by_prediction_decile_dataframe["decile"] = pandas.qcut(
        calibration_by_prediction_decile_dataframe["predicted_price"],
        q=10,
        duplicates="drop",
    )
    calibration_by_prediction_decile_dataframe = (
        calibration_by_prediction_decile_dataframe.groupby("decile", observed=True)
        .agg(predicted=("predicted_price", "mean"), actual=("actual_price", "mean"))
        .reset_index(drop=True)
    )
    subplot_axes[0, 1].plot(
        calibration_by_prediction_decile_dataframe["predicted"],
        calibration_by_prediction_decile_dataframe["actual"],
        marker="o",
        color="#157f68",
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
    subplot_axes[0, 1].set_title("Mean price by predicted-price decile")
    subplot_axes[0, 1].set_xlabel("Mean predicted price (USD)")
    subplot_axes[0, 1].set_ylabel("Mean actual price (USD)")
    subplot_axes[0, 1].legend()
    error_metric_comparison = pandas.DataFrame(
        {
            "Random Forest": [
                float(model_evaluation_metrics["median_absolute_error_usd"]),
                float(model_evaluation_metrics["mae_usd"]),
                float(model_evaluation_metrics["rmse_usd"]),
            ],
            "Median baseline": [
                float(
                    model_evaluation_metrics[
                        "median_baseline_median_absolute_error_usd"
                    ]
                ),
                float(model_evaluation_metrics["median_baseline_mae_usd"]),
                float(model_evaluation_metrics["median_baseline_rmse_usd"]),
            ],
        },
        index=["Median absolute error", "Mean absolute error", "RMSE"],
    )
    error_metric_comparison.plot(
        kind="barh", ax=subplot_axes[1, 0], color=["#f2a65a", "#7a7a7a"]
    )
    subplot_axes[1, 0].set_title("Random Forest versus median-price baseline")
    subplot_axes[1, 0].set_xlabel("Prediction error (USD per night)")
    subplot_axes[1, 0].set_ylabel("")
    coefficient_of_determination_comparison = pandas.DataFrame(
        {
            "Random Forest": [
                float(model_evaluation_metrics["r2_log_price"]),
                float(model_evaluation_metrics["r2_price_usd"]),
            ],
            "Median baseline": [
                float(model_evaluation_metrics["median_baseline_r2_log_price"]),
                float(model_evaluation_metrics["median_baseline_r2_price_usd"]),
            ],
        },
        index=["Log price", "Price in USD"],
    )
    coefficient_of_determination_comparison.plot(
        kind="barh", ax=subplot_axes[1, 1], color=["#8f6bb3", "#7a7a7a"]
    )
    coefficient_of_determination_lower_limit = min(
        0.0, coefficient_of_determination_comparison.min().min() - 0.05
    )
    subplot_axes[1, 1].set_xlim(coefficient_of_determination_lower_limit, 1)
    subplot_axes[1, 1].set_title(
        f"Explained variance; Random Forest WAPE = {float(model_evaluation_metrics['wape_percent']):.1f}%"
    )
    subplot_axes[1, 1].set_xlabel("R²")
    subplot_axes[1, 1].set_ylabel("")
    save_standalone_figures(
        visualisation_figures,
        [
            "19_actual_versus_predicted_price.png",
            "20_predicted_price_calibration.png",
            "21_prediction_error_against_median_baseline.png",
            "22_explained_variance_against_median_baseline.png",
        ],
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
    subplot_axes[0, 0].set_title("Permutation importance on held-out data")
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


def plot_anomaly_overview(cleaned_listings_dataframe: pandas.DataFrame) -> None:
    """Create standalone pricing-anomaly overview visualisations."""
    visualisation_figures, subplot_axes = create_standalone_figures(2, 2)
    pricing_status_counts_series = (
        cleaned_listings_dataframe["status"].value_counts().reindex(STATUS_ORDER)
    )
    pricing_status_shares_series = (
        100 * pricing_status_counts_series / pricing_status_counts_series.sum()
    )
    subplot_axes[0, 0].barh(
        STATUS_ORDER,
        pricing_status_shares_series.values,
        color=[STATUS_COLOURS[status] for status in STATUS_ORDER],
    )
    subplot_axes[0, 0].set_title("Share of clean listings")
    subplot_axes[0, 0].set_xlabel("Listings (%)")
    label_bars(subplot_axes[0, 0], suffix="%", decimals=1)
    median_residual_by_status_series = (
        cleaned_listings_dataframe.groupby("status", observed=True)["residual_usd"]
        .median()
        .reindex(STATUS_ORDER)
    )
    subplot_axes[0, 1].barh(
        STATUS_ORDER,
        median_residual_by_status_series.values,
        color=[STATUS_COLOURS[status] for status in STATUS_ORDER],
    )
    subplot_axes[0, 1].axvline(0, color="#7a7a7a", linewidth=1)
    subplot_axes[0, 1].set_title("Median actual-minus-estimated price gap")
    subplot_axes[0, 1].set_xlabel("USD per night")
    label_signed_horizontal_bars(subplot_axes[0, 1], suffix=" USD", decimals=1)
    percentage_gap_limit = float(
        cleaned_listings_dataframe["price_gap_percent"].abs().quantile(0.99)
    )
    central_percentage_gaps = cleaned_listings_dataframe[
        cleaned_listings_dataframe["price_gap_percent"].between(
            -percentage_gap_limit, percentage_gap_limit
        )
    ]
    seaborn.histplot(
        data=central_percentage_gaps,
        x="price_gap_percent",
        hue="status",
        hue_order=STATUS_ORDER,
        palette=STATUS_COLOURS,
        bins=80,
        element="step",
        common_norm=False,
        ax=subplot_axes[1, 0],
    )
    subplot_axes[1, 0].axvline(-50, linestyle="--", color="#157f68")
    subplot_axes[1, 0].axvline(50, linestyle="--", color="#c94845")
    subplot_axes[1, 0].set_title(
        "Price-gap distribution with ±50% classification thresholds"
    )
    subplot_axes[1, 0].set_xlabel("Actual minus estimated price (%)")
    subplot_axes[1, 0].set_ylabel("Listings")
    price_band_dataframe = cleaned_listings_dataframe[["price", "status"]].copy()
    price_band_dataframe["price_band"] = pandas.qcut(
        price_band_dataframe["price"], q=6, duplicates="drop"
    )
    status_share_by_price_band = 100 * pandas.crosstab(
        price_band_dataframe["price_band"],
        price_band_dataframe["status"],
        normalize="index",
    ).reindex(columns=STATUS_ORDER)
    status_share_by_price_band.index = [
        f"${price_interval.left:,.0f}–${price_interval.right:,.0f}"
        for price_interval in status_share_by_price_band.index
    ]
    status_share_by_price_band.plot(
        kind="bar",
        stacked=True,
        color=[STATUS_COLOURS[status] for status in STATUS_ORDER],
        ax=subplot_axes[1, 1],
    )
    subplot_axes[1, 1].set_title("Pricing-status composition by actual-price band")
    subplot_axes[1, 1].set_xlabel("Actual nightly-price band")
    subplot_axes[1, 1].set_ylabel("Listings (%)")
    subplot_axes[1, 1].tick_params(axis="x", rotation=20)
    subplot_axes[1, 1].legend(
        title="Pricing status",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    save_standalone_figures(
        visualisation_figures,
        [
            "27_pricing_anomaly_shares.png",
            "28_median_price_gap_by_pricing_status.png",
            "29_price_gap_distribution.png",
            "30_pricing_status_by_actual_price_band.png",
        ],
    )


def plot_anomaly_hotspots(
    cleaned_listings_dataframe: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Create hotspot reports and return neighbourhood and segment metrics."""
    neighbourhood_opportunity_metrics_dataframe = (
        cleaned_listings_dataframe.assign(
            is_overpriced=cleaned_listings_dataframe["status"].eq("overpriced"),
            is_underpriced=cleaned_listings_dataframe["status"].eq("underpriced"),
        )
        .groupby(["boroname", "ntaname"], observed=True)
        .agg(
            listings=("id", "size"),
            median_actual_price=("price", "median"),
            median_estimated_price=("predicted_price", "median"),
            median_gap_percent=("price_gap_percent", "median"),
            overpriced_listings=("is_overpriced", "sum"),
            underpriced_listings=("is_underpriced", "sum"),
            overpriced_rate=("is_overpriced", "mean"),
            underpriced_rate=("is_underpriced", "mean"),
        )
        .reset_index()
    )
    neighbourhood_opportunity_metrics_dataframe[
        ["overpriced_rate", "underpriced_rate"]
    ] *= 100
    neighbourhoods_with_sufficient_listing_volume_dataframe = (
        neighbourhood_opportunity_metrics_dataframe[
            neighbourhood_opportunity_metrics_dataframe["listings"] >= 100
        ]
    )
    highest_overpriced_rate_neighbourhoods_dataframe = (
        neighbourhoods_with_sufficient_listing_volume_dataframe.nlargest(
            12, "overpriced_rate"
        ).sort_values("overpriced_rate")
    )
    highest_underpriced_rate_neighbourhoods_dataframe = (
        neighbourhoods_with_sufficient_listing_volume_dataframe.nlargest(
            12, "underpriced_rate"
        ).sort_values("underpriced_rate")
    )
    market_segment_opportunity_metrics_dataframe = (
        cleaned_listings_dataframe.assign(
            is_overpriced=cleaned_listings_dataframe["status"].eq("overpriced"),
            is_underpriced=cleaned_listings_dataframe["status"].eq("underpriced"),
        )
        .groupby(["boroname", "room_type"], observed=True)
        .agg(
            listings=("id", "size"),
            median_actual_price=("price", "median"),
            median_estimated_price=("predicted_price", "median"),
            overpriced_rate=("is_overpriced", "mean"),
            underpriced_rate=("is_underpriced", "mean"),
        )
        .reset_index()
    )
    market_segment_opportunity_metrics_dataframe[
        ["overpriced_rate", "underpriced_rate"]
    ] *= 100
    visualisation_figures, subplot_axes = create_standalone_figures(
        2, 2, figure_size=(12, 8)
    )
    subplot_axes[0, 0].barh(
        highest_overpriced_rate_neighbourhoods_dataframe["ntaname"],
        highest_overpriced_rate_neighbourhoods_dataframe["overpriced_rate"],
        color="#c94845",
    )
    subplot_axes[0, 0].set_title(
        "Highest overpriced share by NTA (at least 100 listings)"
    )
    subplot_axes[0, 0].set_xlabel("Overpriced listings (%)")
    label_bars(subplot_axes[0, 0], suffix="%", decimals=1)
    subplot_axes[0, 1].barh(
        highest_underpriced_rate_neighbourhoods_dataframe["ntaname"],
        highest_underpriced_rate_neighbourhoods_dataframe["underpriced_rate"],
        color="#157f68",
    )
    subplot_axes[0, 1].set_title(
        "Highest underpriced share by NTA (at least 100 listings)"
    )
    subplot_axes[0, 1].set_xlabel("Underpriced listings (%)")
    label_bars(subplot_axes[0, 1], suffix="%", decimals=1)
    for subplot_axis, value, colour, label in [
        (subplot_axes[1, 0], "overpriced_rate", "Reds", "Overpriced listings (%)"),
        (subplot_axes[1, 1], "underpriced_rate", "Greens", "Underpriced listings (%)"),
    ]:
        status_rate_by_market_segment_matrix = (
            market_segment_opportunity_metrics_dataframe.pivot(
                index="boroname", columns="room_type", values=value
            )
        )
        seaborn.heatmap(
            status_rate_by_market_segment_matrix,
            annot=True,
            fmt=".1f",
            cmap=colour,
            cbar_kws={"label": label},
            ax=subplot_axis,
        )
        subplot_axis.set_title(f"{label.split()[0]} share by borough and room type")
        subplot_axis.set_xlabel("")
        subplot_axis.set_ylabel("")
    save_standalone_figures(
        visualisation_figures,
        [
            "31_highest_overpriced_share_by_nta.png",
            "32_highest_underpriced_share_by_nta.png",
            "33_overpriced_share_by_borough_and_room_type.png",
            "34_underpriced_share_by_borough_and_room_type.png",
        ],
    )
    return (
        neighbourhood_opportunity_metrics_dataframe,
        market_segment_opportunity_metrics_dataframe,
    )


def plot_spatial_anomaly_rates(
    cleaned_listings_dataframe: pandas.DataFrame,
    neighbourhood_tabulation_areas_geodataframe: geopandas.GeoDataFrame,
) -> None:
    """Create neighbourhood maps of reliable pricing-anomaly rates."""
    anomaly_metrics_by_neighbourhood = (
        cleaned_listings_dataframe.assign(
            is_overpriced=cleaned_listings_dataframe["status"].eq("overpriced"),
            is_underpriced=cleaned_listings_dataframe["status"].eq("underpriced"),
        )
        .groupby("ntaname", observed=True)
        .agg(
            listings=("id", "size"),
            overpriced_rate=("is_overpriced", "mean"),
            underpriced_rate=("is_underpriced", "mean"),
            median_price_gap_percent=("price_gap_percent", "median"),
        )
        .reset_index()
    )
    anomaly_metrics_by_neighbourhood[["overpriced_rate", "underpriced_rate"]] *= 100
    spatial_anomaly_metrics = neighbourhood_tabulation_areas_geodataframe.merge(
        anomaly_metrics_by_neighbourhood, on="ntaname", how="left"
    )
    reliable_spatial_anomaly_metrics = spatial_anomaly_metrics[
        spatial_anomaly_metrics["listings"] >= 50
    ]
    visualisation_figures, subplot_axes = create_standalone_figures(
        1, 3, figure_size=(10, 8)
    )
    maximum_overpriced_rate = float(
        reliable_spatial_anomaly_metrics["overpriced_rate"].max()
    )
    maximum_underpriced_rate = float(
        reliable_spatial_anomaly_metrics["underpriced_rate"].max()
    )
    maximum_absolute_median_price_gap = float(
        reliable_spatial_anomaly_metrics["median_price_gap_percent"].abs().max()
    )
    map_specifications = [
        (
            "overpriced_rate",
            "Overpriced listings (%)",
            "Reds",
            0.0,
            maximum_overpriced_rate,
        ),
        (
            "underpriced_rate",
            "Underpriced listings (%)",
            "Greens",
            0.0,
            maximum_underpriced_rate,
        ),
        (
            "median_price_gap_percent",
            "Median price gap (%)",
            "coolwarm",
            -maximum_absolute_median_price_gap,
            maximum_absolute_median_price_gap,
        ),
    ]
    for (
        subplot_axis,
        metric_column,
        map_title,
        colour_map,
        colour_scale_minimum,
        colour_scale_maximum,
    ) in [
        (subplot_axes[0], *map_specifications[0]),
        (subplot_axes[1], *map_specifications[1]),
        (subplot_axes[2], *map_specifications[2]),
    ]:
        spatial_anomaly_metrics.plot(
            color="#e5e5e5", edgecolor="white", linewidth=0.25, ax=subplot_axis
        )
        reliable_spatial_anomaly_metrics.plot(
            column=metric_column,
            cmap=colour_map,
            vmin=colour_scale_minimum,
            vmax=colour_scale_maximum,
            linewidth=0.25,
            edgecolor="white",
            legend=True,
            legend_kwds={"label": map_title, "shrink": 0.7},
            ax=subplot_axis,
        )
        subplot_axis.set_title(f"{map_title}; at least 50 listings")
        subplot_axis.set_axis_off()
    save_standalone_figures(
        visualisation_figures,
        [
            "35_overpriced_rate_by_nta.png",
            "36_underpriced_rate_by_nta.png",
            "37_median_price_gap_by_nta.png",
        ],
    )


def export_metrics(
    cleaned_listings_dataframe: pandas.DataFrame,
    data_retention_metrics: dict[str, int | float],
    model_evaluation_metrics: dict[str, object],
    neighbourhood_opportunity_metrics_dataframe: pandas.DataFrame,
    market_segment_opportunity_metrics_dataframe: pandas.DataFrame,
) -> None:
    """Export summary metrics and opportunity tables beside the PNG reports."""
    pricing_status_counts_series = (
        cleaned_listings_dataframe["status"].value_counts().reindex(STATUS_ORDER)
    )
    pricing_status_shares_series = (
        100 * pricing_status_counts_series / len(cleaned_listings_dataframe)
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
        "pricing_status": {
            status: {
                "listings": int(pricing_status_counts_series[status]),
                "share_percent": float(pricing_status_shares_series[status]),
            }
            for status in STATUS_ORDER
        },
    }
    (VISUALISATION_OUTPUT_DIRECTORY / "metrics_summary.json").write_text(
        json.dumps(exported_metrics_dictionary, indent=2), encoding="utf-8"
    )
    neighbourhood_opportunity_metrics_dataframe.sort_values(
        "listings", ascending=False
    ).to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "nta_opportunity_metrics.csv", index=False
    )
    market_segment_opportunity_metrics_dataframe.sort_values(
        ["boroname", "room_type"]
    ).to_csv(
        VISUALISATION_OUTPUT_DIRECTORY / "segment_opportunity_metrics.csv", index=False
    )


def main() -> None:
    """Generate all reports and machine-readable metric files."""
    ensure_source_files()
    set_plot_style()
    VISUALISATION_OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    clear_generated_png_files()
    source_listings_dataframe = pandas.read_csv(AIRBNB_SOURCE_CSV_PATH)
    (
        cleaned_listings_dataframe,
        neighbourhood_tabulation_areas_geodataframe,
        data_retention_metrics,
    ) = prepare_data()
    cleaned_listings_dataframe, feature_importance_series, model_evaluation_metrics = (
        train_and_score(cleaned_listings_dataframe)
    )
    testing_predictions_dataframe = model_evaluation_metrics.pop("_test_results")
    plot_data_and_market(
        source_listings_dataframe, cleaned_listings_dataframe, data_retention_metrics
    )
    plot_price_segments(cleaned_listings_dataframe)
    plot_spatial_market(
        cleaned_listings_dataframe, neighbourhood_tabulation_areas_geodataframe
    )
    plot_operational_relationships(cleaned_listings_dataframe)
    plot_host_and_booking_policy_relationships(cleaned_listings_dataframe)
    plot_model_performance(testing_predictions_dataframe, model_evaluation_metrics)
    plot_model_diagnostics(testing_predictions_dataframe, feature_importance_series)
    plot_anomaly_overview(cleaned_listings_dataframe)
    (
        neighbourhood_opportunity_metrics_dataframe,
        market_segment_opportunity_metrics_dataframe,
    ) = plot_anomaly_hotspots(cleaned_listings_dataframe)
    plot_spatial_anomaly_rates(
        cleaned_listings_dataframe, neighbourhood_tabulation_areas_geodataframe
    )
    export_metrics(
        cleaned_listings_dataframe,
        data_retention_metrics,
        model_evaluation_metrics,
        neighbourhood_opportunity_metrics_dataframe,
        market_segment_opportunity_metrics_dataframe,
    )
    print(
        f"Generated {EXPECTED_VISUALISATION_COUNT} standalone PNG visualisations in "
        f"{VISUALISATION_OUTPUT_DIRECTORY}"
    )
    print(json.dumps(model_evaluation_metrics, indent=2))


if __name__ == "__main__":
    main()
