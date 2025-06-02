import pandas as pd
from shapely.geometry import MultiPoint
import multiprocessing as mp


def _calculate_single_cluster_polygon(args_tuple):
    """
    Worker function to calculate the convex hull WKT for a single cluster's stops.
    Expected args_tuple: (cluster_id, group_data_df)
    """
    cluster_id, group_data = args_tuple

    # Ensure no NaN values are passed to MultiPoint and points are valid
    points_for_hull = [
        (lon, lat)
        for lon, lat in zip(group_data["stop_longitude"], group_data["stop_latitude"])
        if pd.notna(lon) and pd.notna(lat)
    ]

    polygon_wkt = None
    if not points_for_hull:  # No valid points for this cluster
        return {"cluster_id": cluster_id, "port_polygon_wkt": None}

    try:
        multi_point_geom = MultiPoint(points_for_hull)

        # convex_hull of <3 unique points results in Point or LineString.
        # Their .wkt is valid and will be stored.
        if (
            multi_point_geom.is_empty
        ):  # Should not happen if points_for_hull is not empty, but good check
            return {"cluster_id": cluster_id, "port_polygon_wkt": None}

        hull_geom = multi_point_geom.convex_hull
        polygon_wkt = hull_geom.wkt

    except Exception as e:
        print(f"Warning: Error calculating hull for cluster {cluster_id}: {e}")
        polygon_wkt = None  # Ensure None is returned on error

    return {"cluster_id": cluster_id, "port_polygon_wkt": polygon_wkt}


def generate_port_polygons_parallel(
    all_stops_with_clusters_df: pd.DataFrame, num_processes: int = None
) -> pd.DataFrame:
    """
    Calculates convex hull polygons for each cluster in parallel.

    Args:
        all_stops_with_clusters_df (pd.DataFrame): DataFrame containing all stop events
                                                   and their assigned 'cluster' label.
        num_processes (int, optional): Number of processes to use. Defaults to cpu_count().

    Returns:
        pd.DataFrame: A DataFrame with 'cluster_id' and 'port_polygon_wkt'.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    required_cols = ["cluster", "stop_longitude", "stop_latitude"]
    if not all(col in all_stops_with_clusters_df.columns for col in required_cols):
        missing = [
            col
            for col in required_cols
            if col not in all_stops_with_clusters_df.columns
        ]
        print(
            f"Error: 'all_stops_with_clusters_df' must contain {required_cols}. Missing: {missing}"
        )
        return pd.DataFrame(columns=["cluster_id", "port_polygon_wkt"])

    # Filter out noise points (DBSCAN labels noise as -1)
    actual_cluster_stops = all_stops_with_clusters_df[
        all_stops_with_clusters_df["cluster"] != -1
    ].copy()  # Use .copy() to ensure it's a separate DataFrame

    if actual_cluster_stops.empty:
        print(
            "No actual cluster stops found (all might be noise). Cannot generate polygons."
        )
        return pd.DataFrame(columns=["cluster_id", "port_polygon_wkt"])

    # Prepare tasks: list of (cluster_id, group_data_df) tuples
    tasks = [
        (cluster_id, group_data)
        for cluster_id, group_data in actual_cluster_stops.groupby("cluster")
        if not group_data.empty  # Ensure group is not empty before adding to tasks
    ]

    if not tasks:
        print("No tasks (clusters) to process for polygon generation.")
        return pd.DataFrame(columns=["cluster_id", "port_polygon_wkt"])

    print(
        f"Starting parallel polygon generation with {num_processes} processes for {len(tasks)} clusters..."
    )

    results_list = []
    with mp.Pool(processes=num_processes) as pool:
        for i, result_dict in enumerate(
            pool.imap_unordered(_calculate_single_cluster_polygon, tasks)
        ):
            if result_dict:  # Worker returns a dict or None
                results_list.append(result_dict)
            if (i + 1) % 50 == 0 and i + 1 < len(
                tasks
            ):  # Optional progress update, less frequent for many small tasks
                print(f"  Generated polygons for {i+1}/{len(tasks)} clusters...")

    print(
        f"Finished parallel polygon generation. Aggregated results for {len(results_list)} clusters."
    )
    if not results_list:
        return pd.DataFrame(columns=["cluster_id", "port_polygon_wkt"])

    return pd.DataFrame(results_list)
