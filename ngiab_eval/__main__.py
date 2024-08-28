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
from colorama import Fore, Style, init
from ngiab_eval.output_formatter import write_output
import argparse

# Initialize colorama
init(autoreset=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def download_nwm_output(gage, start_time, end_time) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one
    try:
        Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)
    logger.debug("Creating s3fs object")
    store = s3fs.S3Map(
        f"s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
        s3=s3fs.S3FileSystem(anon=True),
    )
    logger.debug("Opening zarr store")
    dataset = xr.open_zarr(store, consolidated=True)
    logger.debug("Selecting time slice")
    dataset = dataset.sel(time=slice(start_time, end_time))

    # the gage_id is stored as 15 bytes and isn't indexed by default
    # e.g. 10154200 -> b'0000000010154200'
    gage = np.bytes_(gage.rjust(15))

    # the indexer needs to be computed before the dataset is filtered as gage_id is a dask array
    logger.debug("Computing indexer")
    indexer = (dataset.gage_id == gage).compute()

    logger.debug("Filtering dataset")
    dataset = dataset.where(indexer, drop=True)

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset[["streamflow"]]

    dataset["streamflow"] = dataset["streamflow"] * 0.0283168
    logger.debug("Returning dataset")
    return dataset.compute()


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


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{Fore.BLUE}{message}{Style.RESET_ALL}"
        if record.levelno == logging.WARNING:
            return f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        if record.levelno == logging.INFO:
            return f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        return message


def setup_logging() -> None:
    """Set up logging configuration with green formatting."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


# def plot_streamflow(df, gage):
#     import matplotlib.pyplot as plt

#     plt.plot(df["current_time"], df["streamflow"], label="NWM")
#     plt.plot(df["current_time"], df["value"], label="USGS")
#     plt.plot(df["current_time"], df["flow"], label="NGEN")
#     plt.legend()
#     plt.title(f"Streamflow for {gage}")
#     # save the plot
#     plt.savefig(f"gage-{gage}_streamflow.png")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Subsetting hydrofabrics, forcing generation, and realization creation"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Path to a csv or txt file containing a newline separated list of catchment IDs, when used with -l, the file should contain lat/lon pairs",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_arguments()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if not args.input_file:
        logger.error("No input file provided")
        exit(1)
    folder_to_eval = Path(args.input_file)
    logger.debug("Starting main.py")
    # folder_to_eval = Path("/home/josh/ngiab_preprocess_output/gage-10154200")

    # get gages in folder, flowpath_attributes.rl_gages
    logger.debug("Getting gages from hydrofabric")
    wb_gage_pairs = get_gages_from_hydrofabric(folder_to_eval)
    gages = [pair[1] for pair in wb_gage_pairs]
    logger.info(f"Found {len(wb_gage_pairs)} gages in folder")
    logger.debug(f"getting simulation start and end time")
    start_time, end_time = get_simulation_start_end_time(folder_to_eval)
    logger.info(f"Simulation start time: {start_time}, end time: {end_time}")

    logger.info("Downloaded NWM output")
    for wb_id, gage in wb_gage_pairs:
        logger.debug(f"Processing {gage}")
        logger.debug(f"Downloading USGS data for {gage}")
        service = IVDataService()
        usgs_data = service.get(sites=gage, startDT="2010-01-01", endDT="2010-01-10")
        service._restclient.close()
        logger.debug(f"Downloaded USGS data for {gage}")
        logger.debug(f"Downloading NWM data for {gage}")
        nwm_data = download_nwm_output(gage, start_time, end_time)
        logger.debug(f"Downloaded NWM data for {gage}")
        logger.debug(f"Getting simulation output for {gage}")
        simulation_output = get_simulation_output(wb_id, folder_to_eval)
        logger.debug(f"Got simulation output for {gage}")
        logger.debug(f"Merging simulation and gage data for {gage}")
        new_df = pd.merge(
            simulation_output,
            usgs_data,
            left_on="current_time",
            right_on="value_time",
            how="inner",
        )
        logger.debug(f"Merged in nwm data for {gage}")
        new_df = pd.merge(
            new_df, nwm_data.to_dataframe(), left_on="current_time", right_on="time", how="inner"
        )
        logger.debug(f"Merging complete for {gage}")
        new_df = new_df.dropna()
        # drop everything except the columns we want
        new_df = new_df[["flow", "value", "streamflow"]]
        new_df.columns = ["NGEN", "USGS", "NWM"]
        logger.debug(f"Calculating NSE and KGE for {gage}")
        nwm_nse = he.evaluator(he.nse, new_df["NWM"], new_df["USGS"])
        ngen_nse = he.evaluator(he.nse, new_df["NGEN"], new_df["USGS"])
        nwm_kge = he.evaluator(he.kge, new_df["NWM"], new_df["USGS"])
        ngen_kge = he.evaluator(he.kge, new_df["NGEN"], new_df["USGS"])

        # pickle ngen_kge
        write_output(folder_to_eval, gage, nwm_nse, nwm_kge, ngen_nse, ngen_kge)
        logger.info(f"Finished processing {gage}")
