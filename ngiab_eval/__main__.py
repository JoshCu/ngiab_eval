import logging
from pathlib import Path
import argparse
import warnings
from ngiab_eval import evaluate_folder
from ngiab_eval import setup_logging

# we check this ourselves and log a warning so we can silence this
warnings.filterwarnings("ignore", message="No data was returned by the request.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    args = parse_arguments()
    setup_logging(args.debug)
    logger.info("Starting evaluation")
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if not args.input_file:
        logger.error("No input file provided")
        exit(1)

    folder_to_eval = Path(args.input_file)
    evaluate_folder(folder_to_eval, args.plot, args.debug)
