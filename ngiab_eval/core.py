import sqlite3
import os
import s3fs
import xarray as xr
import logging
from dask.distributed import Client, LocalCluster, progress
from distributed.utils import silence_logging_cmgr
from pathlib import Path
import pandas as pd
import glob
import time
import json
import numpy as np
from hydrotools.nwis_client import IVDataService
import hydroeval as he
from colorama import Fore, Style, init
from ngiab_eval.output_formatter import write_output, write_streamflow_to_sqlite
from ngiab_eval.gage_to_feature_id import feature_ids
import warnings
import multiprocessing
from functools import partial

# we check this ourselves and log a warning so we can silence this
warnings.filterwarnings("ignore", message="No data was returned by the request.")

# Initialize colorama
init(autoreset=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_nwm_output(gage, start_time, end_time) -> xr.Dataset:
    """Load zarr datasets from S3 within the specified time range."""
    # if a LocalCluster is not already running, start one

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
    logger.debug("Computing dataset")
    logger.debug("Dataset: %s", dataset)

    return dataset


def check_local_cache(gage, start_time, end_time, cache_folder: Path = Path(".")) -> xr.Dataset:
    # check if the data is already in the cache
    # if it is, return it
    # if it is not, download it and return it
    cached_file = cache_folder / f"{gage}_{start_time}_{end_time}.nc"
    temp_file = cache_folder / f"{gage}_{start_time}_{end_time}_downloading.nc"

    if temp_file.exists():
        temp_file.unlink()

    if not cache_folder.exists():
        cache_folder.mkdir(exist_ok=True, parents=True)

    if cached_file.exists():
        dataset = xr.open_dataset(cached_file)
    else:
        dataset = download_nwm_output(gage, start_time, end_time)
        client = Client.current()
        logger.debug("client fetched")
        future = client.compute(dataset.to_netcdf(temp_file, compute=False))
        logger.debug("future created")
        # Display progress bar
        progress(future)
        future.result()
        temp_file.rename(cached_file)
        dataset = xr.open_dataset(cached_file)

    df = zip(dataset.time.values, dataset.streamflow.values)
    time_series = pd.DataFrame(df, columns=["time", "streamflow"])
    return time_series


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
    nc_file = folder_to_eval / "outputs" / "troute" / "*.nc"
    # find the nc file
    nc_files = glob.glob(str(nc_file))
    if len(nc_files) == 0:
        raise FileNotFoundError("No netcdf file found in the outputs/troute folder")
    if len(nc_files) > 1:
        logger.warning("Multiple netcdf files found in the outputs/troute folder")
        logger.warning("Using the most recent file")
        nc_files.sort(key=os.path.getmtime)
        file_to_open = nc_files[-1]
    if len(nc_files) == 1:
        file_to_open = nc_files[0]
    all_output = xr.open_dataset(file_to_open)
    print(all_output)
    id_stem = wb_id.split("-")[1]
    gage_output = all_output.sel(feature_id=int(id_stem))
    gage_output = gage_output.drop_vars(["type", "velocity", "depth", "nudge", "feature_id"])
    gage_output = gage_output.to_dataframe()
    print(gage_output)
    return gage_output.reset_index()


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
        message = message.replace("<module>", "main")
        time = message.split(" - ")[0] + " - "
        rest_of_message = " - ".join(message.split(" - ")[1:])
        if record.levelno == logging.DEBUG:
            return f"{time}{Fore.BLUE}{rest_of_message}{Style.RESET_ALL}"
        if record.levelno == logging.WARNING:
            return f"{time}{Fore.YELLOW}{rest_of_message}{Style.RESET_ALL}"
        if record.levelno == logging.INFO:
            return f"{time}{Fore.GREEN}{rest_of_message}{Style.RESET_ALL}"
        return message



def plot_streamflow(output_folder, df, gage):
    try:
        import seaborn as sns
        import matplotlib

        # use Agg backend for headless plotting
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Seaborn and matplotlib are required for plotting, please pip install ngiab_eval[plot]"
        )
    plot_folder = Path(output_folder) / "eval" / "plots"
    plot_folder.mkdir(exist_ok=True, parents=True)
    output_image = plot_folder / f"gage-{gage}_streamflow.png"

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for source in ["NWM", "USGS", "NGEN"]:
        sns.lineplot(x="time", y=source, data=df, label=source, ax=ax)

    ax.set(title=f"Streamflow for {gage}", xlabel="Time", ylabel="Streamflow (m³ s⁻¹)")
    ax.legend(title="Source")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close(fig)


def get_usgs_data(gage, start_time, end_time, cache_path):
    service = IVDataService(cache_filename=cache_path)
    logger.info(f"Downloading USGS data for {gage}")
    usgs_data = service.get(sites=gage, startDT=start_time, endDT=end_time)
    return usgs_data

def evaluate_gage(
    gage_wb_pair,
    cache_path,
    start_time,
    end_time,
    folder_to_eval,
    eval_output_folder,
    plot=False,
    debug=False,
):
    gage = gage_wb_pair[0]
    wb_id = gage_wb_pair[1]
    usgs_data = get_usgs_data(gage, start_time, end_time, cache_path)
    if usgs_data.empty:
        logger.warning(f"No data found for {gage} between {start_time} and {end_time}")
        time.sleep(2)
        return
    logger.info(f"Downloading NWM data for {gage}")
    nwm_data = check_local_cache(
        gage, start_time, end_time, cache_folder=eval_output_folder / "nwm_cache"
    )
    logger.debug(f"Downloaded NWM data for {gage}")
    logger.info(f"Getting simulation output for {gage}")
    simulation_output = get_simulation_output(wb_id, folder_to_eval)
    logger.debug(f"Got simulation output for {gage}")
    logger.debug(f"Merging simulation and gage data for {gage}")
    new_df = pd.merge(
        simulation_output,
        usgs_data,
        left_on="time",
        right_on="value_time",
        how="inner",
    )
    logger.debug(f"Merged in nwm data for {gage}")
    new_df = pd.merge(new_df, nwm_data, left_on="time", right_on="time", how="inner")
    logger.debug(f"Merging complete for {gage}")
    new_df = new_df.dropna()
    # drop everything except the columns we want
    new_df = new_df[["time", "flow", "value", "streamflow"]]
    new_df.columns = ["time", "NGEN", "USGS", "NWM"]
    print(new_df)
    # convert USGS to cms
    new_df["USGS"] = new_df["USGS"] * 0.0283168
    logger.info(f"Calculating NSE and KGE for {gage}")
    nwm_nse = he.evaluator(he.nse, new_df["NWM"], new_df["USGS"])
    ngen_nse = he.evaluator(he.nse, new_df["NGEN"], new_df["USGS"])
    nwm_kge = he.evaluator(he.kge, new_df["NWM"], new_df["USGS"])
    ngen_kge = he.evaluator(he.kge, new_df["NGEN"], new_df["USGS"])
    nwm_pbias = he.evaluator(he.pbias, new_df["NWM"], new_df["USGS"])
    ngen_pbias = he.evaluator(he.pbias, new_df["NGEN"], new_df["USGS"])

    write_output(
        eval_output_folder, gage, nwm_nse, nwm_kge, nwm_pbias, ngen_nse, ngen_kge, ngen_pbias
    )


    debug_output = eval_output_folder / "debug"
    debug_output.mkdir(exist_ok=True)
    new_df.to_csv(debug_output / f"streamflow_at_{gage}.csv")
    write_streamflow_to_sqlite(new_df, gage, eval_output_folder)

    if plot:
        logger.info(f"plotting streamflow for {gage}")
        plot_streamflow(folder_to_eval, new_df, gage)

    logger.info(f"Finished processing {gage}")


def evaluate_folder(folder_to_eval: Path, plot: bool = False, debug: bool = False) -> None:
    if not folder_to_eval.exists():
        raise FileNotFoundError(f"Folder {folder_to_eval} does not exist")

    if debug:
        global logger
        logger.setLevel(logging.DEBUG)

    eval_output_folder = folder_to_eval / "eval"
    eval_output_folder.mkdir(exist_ok=True)

    logger.info("Getting gages from hydrofabric")
    wb_gage_pairs = get_gages_from_hydrofabric(folder_to_eval)
    all_gages = {}
    for wb_id, g in wb_gage_pairs:
        gages = g.split(",")
        for gage in gages:
            all_gages[gage] = wb_id

    logger.info(f"Found {len(all_gages)} gages in the hydrofabric")
    logger.debug(f"getting simulation start and end time")
    start_time, end_time = get_simulation_start_end_time(folder_to_eval)
    logger.info(f"Simulation start time: {start_time}, end time: {end_time}")
    cache_path = eval_output_folder / "nwisiv_cache.sqlite"

    evaluate_gage_partial = partial(
        evaluate_gage,
        cache_path=cache_path,
        start_time=start_time,
        end_time=end_time,
        folder_to_eval=folder_to_eval,
        eval_output_folder=eval_output_folder,
        plot=plot,
        debug=debug,
    )

    try:
        client = Client.current()
    except ValueError:
        cluster = LocalCluster()
        client = Client(cluster)

    for gage in all_gages.items():
        evaluate_gage_partial(gage)

    logger.info("Finished evaluation")
