import sqlite3
import os
import s3fs
import xarray as xr
import logging
from dask.distributed import Client, LocalCluster
from pathlib import Path
import pandas as pd
import glob
import json
import numpy as np
from hydrotools.nwis_client import IVDataService
import hydroeval as he
import sys

logging.basicConfig(level=logging.INFO)


def download_nwm_output(gage, start_time, end_time) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    try:
        Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)
    store = s3fs.S3Map(
        f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        s3=s3fs.S3FileSystem(anon=True),
    )
    dataset = xr.open_zarr(store, consolidated=True)
    dataset = dataset.sel(time=slice(start_time, end_time))

    # the gage_id is stored as 15 bytes and isn't indexed by default
    # e.g. 10154200 -> b'0000000010154200'
    gage = np.bytes_(gage.rjust(15))

    # the indexer needs to be computed before the dataset is filtered as gage_id is a dask array
    indexer = (dataset.gage_id == gage).compute()

    dataset = dataset.where(indexer, drop=True)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset[["streamflow"]]

    dataset["streamflow"] = dataset["streamflow"] * 0.0283168

    return dataset


def get_gages_from_hydrofabric(folder_to_eval):
    # search inside the folder for _subset.gpkg recursively
    gpkg_file = None
    for root, dirs, files in os.walk(folder_to_eval):
        for file in files:
            if file.endswith("_subset.gpkg"):
                gpkg_file = os.path.join(root, file)
                break

    if gpkg_file is None:
        raise FileNotFoundError("No subset.gpkg file found in folder")

    with sqlite3.connect(gpkg_file) as conn:
        results = conn.execute(
            "SELECT id, rl_gages FROM flowpath_attributes WHERE rl_gages IS NOT NULL"
        ).fetchall()
    return results


def get_simulation_output(wb_id, folder_to_eval):
    csv_files = folder_to_eval / "outputs" / "troute" / "*.csv"
    id_stem = wb_id.split("-")[1]

    # read every csv file filter out featureID == id_stem, then merge using time as the key
    csv_files = glob.glob(str(csv_files))
    dfs = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        temp_df = temp_df[temp_df["featureID"] == int(id_stem)]
        dfs.append(temp_df)
    merged = pd.concat(dfs)

    # convert the time column to datetime
    merged["current_time"] = pd.to_datetime(merged["current_time"])
    # convert the flow to cms
    merged["flow"] = merged["flow"] * 0.0283168
    return merged


def get_simulation_start_end_time(folder_to_eval):
    realization = folder_to_eval / "config" / "realization.json"
    with open(realization) as f:
        realization = json.load(f)
    start = realization["time"]["start_time"]
    end = realization["time"]["end_time"]
    return start, end


if __name__ == "__main__":
    folder_to_eval = Path("/home/josh/ngiab_preprocess_output/gage-10154200")

    # get gages in folder, flowpath_attributes.rl_gages
    wb_gage_pairs = get_gages_from_hydrofabric(folder_to_eval)
    gages = [pair[1] for pair in wb_gage_pairs]

    logging.info(f"Found {len(wb_gage_pairs)} gages in folder")

    start_time, end_time = get_simulation_start_end_time(folder_to_eval)
    logging.info(f"Simulation start time: {start_time}, end time: {end_time}")

    logging.info("Downloaded NWM output")
    service = IVDataService()
    for wb_id, gage in wb_gage_pairs:
        usgs_data = service.get(sites=gage, startDT="2010-01-01", endDT="2010-01-10")
        gage_data = download_nwm_output(gage, start_time, end_time)
        simulation_output = get_simulation_output(wb_id, folder_to_eval)

        new_df = pd.merge(
            simulation_output,
            usgs_data,
            left_on="current_time",
            right_on="value_time",
            how="inner",
        )
        new_df = pd.merge(
            new_df, gage_data.to_dataframe(), left_on="current_time", right_on="time", how="inner"
        )
        new_df = new_df.dropna()
        # drop everything except the columns we want
        new_df = new_df[["flow", "value", "streamflow"]]
        new_df.columns = ["NGEN", "USGS", "NWM"]

        nwm_nse = he.evaluator(he.nse, new_df["NWM"], new_df["USGS"])
        ngen_nse = he.evaluator(he.nse, new_df["NGEN"], new_df["USGS"])
        nwm_kge = he.evaluator(he.kge, new_df["NWM"], new_df["USGS"])
        ngen_kge = he.evaluator(he.kge, new_df["NGEN"], new_df["USGS"])

        with open(f"gage-{gage}_results.txt", "w") as f:
            f.write(f"nwm_nse: {nwm_nse}\n")
            f.write(f"nwm_kge: {nwm_kge[0][0]}\n")
            f.write(f"nwm_kge_r: {nwm_kge[1][0]}\n")
            f.write(f"nwm_kge_a: {nwm_kge[2][0]}\n")
            f.write(f"nwm_kge_b: {nwm_kge[3][0]}\n")
            f.write(f"ngen_nse: {ngen_nse}\n")
            f.write(f"ngen_kge: {ngen_kge[0][0]}\n")
            f.write(f"ngen_kge_r: {ngen_kge[1][0]}\n")
            f.write(f"ngen_kge_a: {ngen_kge[2][0]}\n")
            f.write(f"ngen_kge_b: {ngen_kge[3][0]}")

        logging.info(f"Finished processing {gage}")
    Client.current().close()
    sys.exit(0)
