from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import multiprocessing as mp


def process_single_mmsi_stops(args_tuple):
    """
    Processes AIS data for a single MMSI to find significant stop events.
    Expected tuple: (mmsi, ship_data_df, min_stop_duration_thresh, max_time_diff_thresh)
    """
    (
        mmsi,
        ship_data_df,
        min_stop_duration_threshold,
        max_time_diff_within_stop_threshold,
    ) = args_tuple

    processed_stops_list = []
    if ship_data_df.empty:
        return pd.DataFrame(processed_stops_list)

    ship_data = ship_data_df.copy()
    ship_data = ship_data.sort_values(by="Timestamp").reset_index(drop=True)

    ship_data["time_diff"] = ship_data["Timestamp"].diff()
    ship_data["stop_segment_id"] = (
        ship_data["time_diff"].isna()
        | (ship_data["time_diff"] > max_time_diff_within_stop_threshold)
    ).cumsum()

    grouped_by_segment = ship_data.groupby("stop_segment_id")

    for _segment_id, segment_data in grouped_by_segment:
        if segment_data.empty:
            continue

        start_time = segment_data["Timestamp"].min()
        end_time = segment_data["Timestamp"].max()
        duration = timedelta(0) if len(segment_data) == 1 else end_time - start_time

        if duration >= min_stop_duration_threshold:
            mean_latitude = segment_data["Latitude"].mean()
            mean_longitude = segment_data["Longitude"].mean()

            ship_type = "N/A"
            if (
                "Ship type" in segment_data.columns
                and not segment_data["Ship type"].empty
            ):
                # Check if all values are NaN or if there's at least one valid string
                first_valid_ship_type = (
                    segment_data["Ship type"].dropna().astype(str).unique()
                )
                if len(first_valid_ship_type) > 0:
                    ship_type = first_valid_ship_type[0]

            nav_status = "N/A"
            if (
                "Navigational status" in segment_data.columns
                and not segment_data["Navigational status"].empty
            ):
                first_valid_nav_status = (
                    segment_data["Navigational status"].dropna().astype(str).unique()
                )
                if len(first_valid_nav_status) > 0:
                    nav_status = first_valid_nav_status[0]

            processed_stops_list.append(
                {
                    "MMSI": mmsi,
                    "stop_latitude": mean_latitude,
                    "stop_longitude": mean_longitude,
                    "stop_start_time": start_time,
                    "stop_end_time": end_time,
                    "stop_duration_hours": duration.total_seconds() / 3600,
                    "num_pings_in_stop": len(segment_data),
                    "Ship type": ship_type,
                    "Navigational status": nav_status,
                }
            )
    return pd.DataFrame(processed_stops_list)


def get_significant_stops_parallel(
    df: pd.DataFrame,
    min_stop_duration_threshold: timedelta,
    max_time_diff_within_stop_threshold: timedelta,
    num_processes: int = None,
) -> pd.DataFrame:
    """
    Identifies significant stop events in parallel by processing each MMSI's data separately.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    required_columns = [
        "MMSI",
        "Timestamp",
        "Latitude",
        "Longitude",
        "Ship type",
        "Navigational status",
    ]
    if not all(col_name in df.columns for col_name in required_columns):
        missing_cols = [
            col_name for col_name in required_columns if col_name not in df.columns
        ]
        print(
            f"Error: DataFrame for stop detection must contain required columns. Missing: {missing_cols}"
        )
        return pd.DataFrame()

    # Prepare arguments for each task. Each task processes data for one MMSI.
    # df.groupby("MMSI") creates an iterator of (name, group_df)
    # We create a list of tuples: (mmsi_value, mmsi_dataframe_group, threshold1, threshold2)
    tasks = [
        (
            mmsi,
            group_df,
            min_stop_duration_threshold,
            max_time_diff_within_stop_threshold,
        )
        for mmsi, group_df in df.groupby("MMSI")
        if not group_df.empty
    ]

    if not tasks:
        print("No data to process after grouping by MMSI for stop detection.")
        return pd.DataFrame()

    all_stops_dfs = []
    print(
        f"Starting parallel stop detection with {num_processes} processes for {len(tasks)} MMSI groups..."
    )
    with mp.Pool(processes=num_processes) as pool:
        for i, result_df in enumerate(
            pool.imap_unordered(process_single_mmsi_stops, tasks)
        ):
            if not result_df.empty:
                all_stops_dfs.append(result_df)
            if (i + 1) % 1000 == 0:  # Simple progress bar
                print(f"Processed {i+1}/{len(tasks)} MMSI groups for stops...")

    print(
        f"Finished parallel stop detection. Concatenating {len(all_stops_dfs)} results."
    )
    if not all_stops_dfs:
        print("No significant stops were found by any process.")
        return pd.DataFrame()

    final_stops_df = pd.concat(all_stops_dfs, ignore_index=True)
    print(
        f"\nFound {len(final_stops_df)} significant stops (duration >= {min_stop_duration_threshold.total_seconds()/3600:.1f}h, "
        f"segment gap <= {max_time_diff_within_stop_threshold.total_seconds()/60:.0f}min)."
    )

    return final_stops_df


def cluster_stops_dbscan(
    stops_df: pd.DataFrame,
    eps_rad: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    Clusters stop events using DBSCAN based on their geographical coordinates.
    """
    if stops_df.empty:
        print("Stops DataFrame is empty. No clustering performed.")
        stops_df["cluster"] = (
            -1
        )  # Add cluster column for consistency if expected downstream
        return stops_df
    if len(stops_df) < min_samples:
        print(
            f"Warning: Number of stops ({len(stops_df)}) is less than min_samples_stops ({min_samples}). "
            "All stops will likely be classified as noise (-1)."
        )
        stops_df["cluster"] = -1
        return stops_df
    coords_radians = np.radians(stops_df[["stop_latitude", "stop_longitude"]].values)
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
        n_jobs=-1,  # Utilizing all available CPUs for DBSCAN
    )
    try:
        stops_df["cluster"] = db.fit_predict(coords_radians)
    except Exception as e:
        print(f"Error during DBSCAN fitting: {e}")
        stops_df["cluster"] = -2
    return stops_df


def get_ship_type_distribution(series: pd.Series) -> str:
    """ "
    Returns a string summarizing the distribution of ship types in a Series.
    """
    counts = series.value_counts()
    return ", ".join([f"{ship_type}: {count}" for ship_type, count in counts.items()])


def create_cluster_summary_df(clustered_stops_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary DataFrame of clusters from the clustered stops DataFrame.
    The summary includes centroid coordinates, number of unique ships, average and total stop duration.
    """
    if "cluster" not in clustered_stops_df.columns:
        print("Error: 'cluster' column not found in clustered_stops_df.")
        return pd.DataFrame()
    actual_clusters = clustered_stops_df[clustered_stops_df["cluster"] != -1].copy()
    if actual_clusters.empty:
        print(
            "No actual clusters found (all points might be noise). Cannot create cluster summary."
        )
        return pd.DataFrame()
    aggregations = {
        "stop_latitude": "mean",
        "stop_longitude": "mean",
        "MMSI": "nunique",
        "stop_duration_hours": ["mean", "sum", "count"],
        "Ship type": lambda x: (x.mode()[0] if not x.mode().empty else "N/A"),
        "Navigational status": lambda x: (x.mode()[0] if not x.mode().empty else "N/A"),
    }
    cluster_summary = actual_clusters.groupby("cluster").agg(aggregations)
    cluster_summary.columns = [
        "centroid_latitude",
        "centroid_longitude",
        "num_unique_ships",
        "avg_stop_duration_hours",
        "total_stop_duration_hours",
        "num_stops",
        "most_common_ship_type",
        "most_common_navigational_status",
    ]
    ship_type_details = actual_clusters.groupby("cluster")["Ship type"].apply(
        get_ship_type_distribution
    )
    if not ship_type_details.empty:  # Check if ship_type_details is not empty
        cluster_summary["ship_type_distribution"] = ship_type_details
    else:
        cluster_summary["ship_type_distribution"] = "N/A"  # Assign default if empty

    cluster_summary = cluster_summary.reset_index().rename(
        columns={"cluster": "cluster_id"}
    )
    return cluster_summary
