import pandas as pd
from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry import Point as ShapelyPoint
import geopandas
import folium
import os


def _load_wkt_geometry_safe(wkt_string: str):
    """
    Safely loads a WKT string to a Shapely geometry object.
    Returns None if parsing fails.
    """
    if not isinstance(wkt_string, str):
        return None
    try:
        return wkt.loads(wkt_string)
    except ShapelyError:
        # print(f"Warning: Could not parse WKT string (WKTReadingError): {wkt_string[:100]}...")
        return None
    except Exception as e:
        # print(f"Warning: Unexpected error parsing WKT string '{wkt_string[:100]}...': {e}")
        return None


def create_and_save_folium_map(
    summary_df_with_wkt: pd.DataFrame,
    wkt_column_name: str = "port_polygon_wkt",
    output_folder: str = "results",
    output_filename: str = "detected_ports_map.html",
    default_center_lat: float = 58.0,  # Central Baltic Sea approx.
    default_center_lon: float = 20.0,
    default_zoom: int = 5,
):
    """
    Generates an interactive Folium map from a DataFrame containing WKT geometries
    and saves it to an HTML file.

    Args:
        summary_df_with_wkt (pd.DataFrame): DataFrame containing a column with WKT geometries.
        wkt_column_name (str): Name of the column containing WKT strings.
        output_folder (str): Folder where the map HTML file will be saved.
        output_filename (str): Name of the HTML map file.
        default_center_lat (float): Default latitude for map center if calculation fails.
        default_center_lon (float): Default longitude for map center if calculation fails.
        default_zoom (int): Default zoom level for the map.
    """
    if not isinstance(summary_df_with_wkt, pd.DataFrame) or summary_df_with_wkt.empty:
        print("Input DataFrame is empty or not a DataFrame. Skipping map generation.")
        return

    if wkt_column_name not in summary_df_with_wkt.columns:
        print(f"Error: WKT column '{wkt_column_name}' not found in the DataFrame.")
        return

    df_for_map = summary_df_with_wkt.copy()

    # Convert WKT strings to Shapely geometry objects
    df_for_map["geometry"] = df_for_map[wkt_column_name].apply(_load_wkt_geometry_safe)

    # Remove rows where geometry could not be parsed or is None
    df_for_map = df_for_map.dropna(subset=["geometry"])

    if df_for_map.empty:
        print("No valid geometries found to plot after attempting to load WKT strings.")
        return

    # Create a GeoDataFrame
    try:
        gdf_ports = geopandas.GeoDataFrame(
            df_for_map, geometry="geometry", crs="EPSG:4326"
        )
    except Exception as e:
        print(f"Error creating GeoDataFrame: {e}. Skipping map generation.")
        return

    center_lat, center_lon, start_zoom = (
        default_center_lat,
        default_center_lon,
        default_zoom,
    )
    valid_geometries_gdf = gdf_ports[
        gdf_ports.geometry.is_valid & ~gdf_ports.geometry.is_empty
    ].copy()

    if not valid_geometries_gdf.empty:
        try:
            gdf_projected = valid_geometries_gdf.to_crs(
                "EPSG:3035"
            )  # ETRS89 / LAEA Europe
            projected_centroids = gdf_projected.geometry.centroid
            if not projected_centroids.empty:
                mean_projected_x = projected_centroids.x.mean()
                mean_projected_y = projected_centroids.y.mean()

                if pd.notna(mean_projected_x) and pd.notna(mean_projected_y):
                    mean_projected_centroid_point = ShapelyPoint(
                        mean_projected_x, mean_projected_y
                    )
                    mean_geo_centroid_gs = geopandas.GeoSeries(
                        [mean_projected_centroid_point], crs="EPSG:3035"
                    )
                    mean_geo_centroid_epsg4326 = mean_geo_centroid_gs.to_crs(
                        "EPSG:4326"
                    ).iloc[0]
                    center_lat, center_lon = (
                        mean_geo_centroid_epsg4326.y,
                        mean_geo_centroid_epsg4326.x,
                    )
        except Exception as e_center:
            print(
                f"Warning: Error during projected centroid calculation for map centering: {e_center}. Using fallback/default center."
            )
            # Fallback if sophisticated centering fails
            total_bounds = valid_geometries_gdf.total_bounds
            if len(total_bounds) == 4 and all(pd.notna(b) for b in total_bounds):
                center_lon = (total_bounds[0] + total_bounds[2]) / 2
                center_lat = (total_bounds[1] + total_bounds[3]) / 2
    else:
        print(
            "No valid geometries to calculate map center. Using default Baltic coordinates."
        )

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=start_zoom, tiles="OpenStreetMap"
    )
    # Add polygons to the Folium map
    if "geometry" in gdf_ports.columns:
        for _, row in gdf_ports.iterrows():
            if (
                row["geometry"] is not None
                and row["geometry"].is_valid
                and not row["geometry"].is_empty
            ):
                try:
                    tooltip_html = f"""
                    <b>Cluster ID:</b> {row.get('cluster_id', 'N/A')}<br>
                    <b>Unique Ships:</b> {row.get('num_unique_ships', 'N/A')}<br>
                    <b>Avg. Duration (h):</b> {row.get('avg_stop_duration_hours', 0.0):.1f}<br>
                    <b>Total Stops:</b> {row.get('num_stops', 'N/A')}
                    """

                    folium.GeoJson(
                        row["geometry"].__geo_interface__,
                        style_function=lambda x: {
                            "fillColor": "red",
                            "color": "darkblue",
                            "weight": 1.5,
                            "fillOpacity": 0.45,
                        },  # Slightly less transparent
                        tooltip=folium.Tooltip(tooltip_html),
                    ).add_to(m)
                except Exception as e_geojson:
                    print(
                        f"Could not add geometry for cluster {row.get('cluster_id', 'N/A')} to Folium map: {e_geojson}"
                    )

    try:
        os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
        map_filepath = os.path.join(output_folder, output_filename)
        m.save(map_filepath)
        print(f"\nInteractive map saved to: {map_filepath}")
    except Exception as e_save:
        print(f"Error saving map to {map_filepath}: {e_save}")
