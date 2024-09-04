import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


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


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration with green formatting."""
    handler = logging.StreamHandler()
    date_fmt = "%H:%M:%S"
    str_format = "%(asctime)s - %(levelname)7s - %(message)s"
    if debug:
        str_format = "%(asctime)s,%(msecs)02d - %(levelname)7s - %(funcName)s - %(message)s"

    handler.setFormatter(ColoredFormatter(str_format, date_fmt))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
