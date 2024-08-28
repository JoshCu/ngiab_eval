from dataclasses import dataclass
import dataclasses
import numpy as np
import json
from pathlib import Path


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class KGE:
    kge: float
    r: float
    a: float
    b: float

    def __init__(self, hydroeval_kge_output: np.ndarray):
        self.kge = hydroeval_kge_output[0][0]
        self.r = hydroeval_kge_output[1][0]
        self.a = hydroeval_kge_output[2][0]
        self.b = hydroeval_kge_output[3][0]


@dataclass
class results:
    kge: KGE
    nse: float

    def __init__(self, kge_output: np.ndarray, nse_output: np.ndarray):
        self.kge = KGE(kge_output)
        self.nse = nse_output[0]


def create_output_folders(output_folder):
    output_folder = Path(output_folder)
    eval_folder = output_folder / "eval"
    json_folder = eval_folder / "json"
    # plot_folder = eval_folder / "plots"
    folders = [output_folder, eval_folder, json_folder]
    for folder in folders:
        folder.mkdir(exist_ok=True)


def write_output(output_folder, gage, nwm_nse, nwm_kge, ngen_nse, ngen_kge):
    create_output_folders(output_folder)
    output = {}
    output["ngen"] = results(ngen_kge, ngen_nse)
    output["nwm"] = results(nwm_kge, nwm_nse)
    output_file = Path(output_folder) / "eval" / "json" / f"gage-{gage}_results.json"
    with open(output_file, "w") as f:
        f.write(json.dumps(output, cls=EnhancedJSONEncoder, indent=4))
