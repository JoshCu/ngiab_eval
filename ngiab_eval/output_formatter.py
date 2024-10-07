from dataclasses import dataclass
import dataclasses
import numpy as np
import json
from pathlib import Path
import sqlite3


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
    pbias: float

    def __init__(self, kge_output: np.ndarray, nse_output: np.ndarray, pbias_output: np.ndarray):
        self.kge = KGE(kge_output)
        self.nse = nse_output[0]
        self.pbias = pbias_output[0]


def create_output_folders(output_folder):
    output_folder = Path(output_folder)
    json_folder = output_folder / "json"
    folders = [output_folder, json_folder]
    for folder in folders:
        folder.mkdir(exist_ok=True)


def write_output(output_folder, gage, nwm_nse, nwm_kge, nwm_pbias, ngen_nse, ngen_kge, ngen_pbias):
    create_output_folders(output_folder)
    output = {}
    output["ngen"] = results(ngen_kge, ngen_nse, ngen_pbias)
    output["nwm"] = results(nwm_kge, nwm_nse, nwm_pbias)
    output_file = Path(output_folder) / "json" / f"gage-{gage}_results.json"
    with open(output_file, "w") as f:
        f.write(json.dumps(output, cls=EnhancedJSONEncoder, indent=4))
    write_to_sqlite(output_folder, output, gage)

def write_to_sqlite(output_folder,  output, gage):
    # add results to a table called stats
    # gage_id, source, kge, kge_r, kge_a, kge_b, nse, pbias
    database = Path(output_folder) / "streamflow.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS stats (gage_id TEXT, source TEXT, kge REAL, kge_r REAL, kge_a REAL, kge_b REAL, nse REAL, pbias REAL)")
        conn.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (gage, "ngen", output["ngen"].kge.kge, output["ngen"].kge.r, output["ngen"].kge.a, output["ngen"].kge.b, output["ngen"].nse, output["ngen"].pbias))
        conn.execute("INSERT INTO stats VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (gage, "nwm", output["nwm"].kge.kge, output["nwm"].kge.r, output["nwm"].kge.a, output["nwm"].kge.b, output["nwm"].nse, output["nwm"].pbias))

def write_streamflow_to_sqlite(df, gage, output_folder):
    database = Path(output_folder) / "streamflow.db"
    with sqlite3.connect(database) as conn:
        df.to_sql(f"{gage}", conn, if_exists="replace", index=False)
