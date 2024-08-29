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
from ngiab_eval.gage_to_feature_id import feature_ids
import argparse

# Initialize colorama
init(autoreset=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    # select the feature_id
    logger.debug("Selecting feature_id")
    dataset = dataset.sel(time=slice(start_time, end_time), feature_id=feature_ids[gage])

    # drop everything except coordinates feature_id, gage_id, time and variables streamflow
    dataset = dataset[["streamflow"]]

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


def plot_streamflow(output_folder, df, gage):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Seaborn and matplotlib are required for plotting, please pip install ngiab_eval[plot]"
        )

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for source in ["NWM", "USGS", "NGEN"]:
        sns.lineplot(x="time", y=source, data=df, label=source, ax=ax)

    ax.set(title=f"Streamflow for {gage}", xlabel="Time", ylabel="Streamflow (m³ s⁻¹)")
    ax.legend(title="Source")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plot_folder = Path(output_folder) / "eval" / "plots"
    plot_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_folder / f"gage-{gage}_streamflow.png")
    plt.close(fig)


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
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot streamflow data",
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
    logger.debug("Getting gages from hydrofabric")
    wb_gage_pairs = get_gages_from_hydrofabric(folder_to_eval)
    all_gages = {}
    for wb_id, g in wb_gage_pairs:
        gages = g.split(",")
        for gage in gages:
            all_gages[gage] = wb_id

    logger.info(f"Found {len(all_gages)} gages in folder")
    logger.debug(f"getting simulation start and end time")
    start_time, end_time = get_simulation_start_end_time(folder_to_eval)
    logger.info(f"Simulation start time: {start_time}, end time: {end_time}")

    for gage, wb_id in all_gages.items():
        logger.debug(f"Processing {gage}")
        logger.debug(f"Downloading USGS data for {gage}")
        cache_path = folder_to_eval / "nwisiv_cache.sqlite"
        service = IVDataService(cache_filename=cache_path)
        usgs_data = service.get(sites=gage, startDT=start_time, endDT=end_time)
        if usgs_data.empty:
            logger.warning(f"No data found for {gage} between {start_time} and {end_time}")
            service._restclient.close()
            continue
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
        new_df = new_df[["current_time", "flow", "value", "streamflow"]]
        new_df.columns = ["time", "NGEN", "USGS", "NWM"]
        # convert USGS to cms
        new_df["USGS"] = new_df["USGS"] * 0.0283168
        logger.debug(f"Calculating NSE and KGE for {gage}")
        nwm_nse = he.evaluator(he.nse, new_df["NWM"], new_df["USGS"])
        ngen_nse = he.evaluator(he.nse, new_df["NGEN"], new_df["USGS"])
        nwm_kge = he.evaluator(he.kge, new_df["NWM"], new_df["USGS"])
        ngen_kge = he.evaluator(he.kge, new_df["NGEN"], new_df["USGS"])

        write_output(folder_to_eval, gage, nwm_nse, nwm_kge, ngen_nse, ngen_kge)

        if args.debug:
            debug_output = folder_to_eval / "eval" / "debug"
            debug_output.mkdir(exist_ok=True)
            new_df.to_csv(debug_output / f"streamflow_at_{gage}.csv")

        logger.debug("plotting streamflow")
        if args.plot:
            plot_streamflow(folder_to_eval, new_df, gage)

        logger.info(f"Finished processing {gage}")
