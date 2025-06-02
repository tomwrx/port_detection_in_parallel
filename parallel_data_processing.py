import pandas as pd
import multiprocessing as mp


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.dropna().drop_duplicates()
    chunk["Timestamp"] = pd.to_datetime(
        chunk["# Timestamp"], dayfirst=True, errors="coerce"
    )
    chunk.drop("# Timestamp", axis=1, inplace=True)
    chunk = chunk[chunk["SOG"] <= 2]  # Filter out SOG > 2 knots
    return chunk


def load_and_preprocess_data_parallel(
    csv_path: str = None, num_processes: int = None
) -> pd.DataFrame:
    if num_processes is None:
        num_processes = mp.cpu_count()

    chunksize = 1000000  # 1 million rows per chunk
    results = []

    with mp.Pool(num_processes) as pool:
        for chunk_result in pool.imap(
            preprocess_chunk,
            pd.read_csv(
                csv_path,
                usecols=[
                    "# Timestamp",
                    "MMSI",
                    "Latitude",
                    "Longitude",
                    "Navigational status",
                    "SOG",
                    "Ship type",
                ],
                chunksize=chunksize,
            ),
        ):
            results.append(chunk_result)

    df = pd.concat(results, ignore_index=True)
    return df
