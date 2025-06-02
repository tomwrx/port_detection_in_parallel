import os
from datetime import timedelta
import multiprocessing as mp
from parallel_data_processing import load_and_preprocess_data_parallel
from clustering_and_port_detection import (
    get_significant_stops_parallel,
    cluster_stops_dbscan,
    create_cluster_summary_df,
)
from parallel_polygon_processing import (
    generate_port_polygons_parallel,
)
from utils import create_and_save_folium_map

DATA_PATH = r"data\aisdk-2025-02-14.csv"
# Constants for significant stop detection and clustering
MIN_STOP_DURATION = timedelta(
    hours=1
)  # Minimum duration of a stop to be considered significant
MAX_TIME_DIFFERENCE_WITHIN_STOP = timedelta(
    minutes=15
)  # Maximum time difference between pings within a stop
EARTH_RADIUS_KM = 6371.0088  # Mean radius of the Earth in kilometers
EPS_KM = (
    1  # Maximum distance in kilometers to consider two points as part of the same stop
)
EPS_RAD = EPS_KM / EARTH_RADIUS_KM  # Convertion from kilometers to radians for DBSCAN
MIN_SAMPLES_STOPS = 7  # Minimum number of samples to form a cluster for stop detection


if __name__ == "__main__":
    mp.freeze_support()
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print("Starting data loading and preprocessing...")
    df = load_and_preprocess_data_parallel(csv_path=DATA_PATH)
    stops_df = get_significant_stops_parallel(
        df,
        min_stop_duration_threshold=MIN_STOP_DURATION,
        max_time_diff_within_stop_threshold=MAX_TIME_DIFFERENCE_WITHIN_STOP,
    )
    print(f"\nFound {len(stops_df)} significant stop events.")
    if not stops_df.empty:
        clustered_stops_df = cluster_stops_dbscan(
            stops_df, eps_rad=EPS_RAD, min_samples=MIN_SAMPLES_STOPS
        )
        print(f"\nClustered stops (potential ports are clusters != -1):")
        if "cluster" in clustered_stops_df.columns:
            print(
                clustered_stops_df[
                    [
                        "MMSI",
                        "stop_latitude",
                        "stop_longitude",
                        "stop_duration_hours",
                        "Ship type",
                        "Navigational status",
                        "cluster",
                    ]
                ].head()
            )
            summary_df = create_cluster_summary_df(clustered_stops_df)
            if not summary_df.empty:
                print(f"\n--- Found Potential Port Areas (Clusters) ---")
                print("Generating port polygons in parallel...")
                polygons_df = generate_port_polygons_parallel(clustered_stops_df)
                if not polygons_df.empty:
                    summary_with_polygons = summary_df.merge(
                        polygons_df, on="cluster_id", how="left"
                    )
                    print(
                        f"Added polygons to {len(summary_with_polygons[summary_with_polygons['port_polygon_wkt'].notna()])} clusters."
                    )
                    summary_with_polygons.to_excel(
                        os.path.join(output_dir, "detected_ports_with_polygons.xlsx"),
                        index=False,
                    )
                    clustered_stops_df.to_excel(
                        os.path.join(output_dir, "clustered_stops.xlsx"), index=False
                    )
                    print(
                        "\nGenerating and saving Folium map using function from utils.py..."
                    )
                    create_and_save_folium_map(
                        summary_df_with_wkt=summary_with_polygons,
                        wkt_column_name="port_polygon_wkt",
                        output_folder="results",
                        output_filename="detected_ports_map_final.html",
                    )
                else:
                    print(
                        "Summary DataFrame with polygons is empty or not created, skipping map generation."
                    )
            else:
                print("\nNo summary DataFrame created (no actual clusters found).")
        else:
            print("\nClustering did not add 'cluster' column.")
    else:
        print("\nNo significant stops found to cluster.")
