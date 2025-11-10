import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from pathlib import Path
import yaml

config = yaml.safe_load(Path("src/config.yaml").read_text())


def get_data(start: time, end: time, ignore_missing: bool = False) -> pd.DataFrame:
    start_date = datetime.strptime(config["metadata"]["date_start"], "%Y-%m-%d")
    datetime_start = datetime.combine(start_date.date(), start)
    datetime_end = datetime.combine(start_date.date(), end)

    data = []
    current = datetime_start
    while current <= datetime_end:
        file_name = (
            Path(config["paths"]["data_dir"]) / f"{current.strftime('%H%M%S')}.npy"
        )
        if not file_name.exists() and current == datetime_start:
            raise FileNotFoundError(f"The start time data file {file_name} not found.") 
        elif not file_name.exists() and not ignore_missing:
            raise FileNotFoundError(f"Data file {file_name} not found.")
        elif file_name.exists():
            data.append(np.load(file_name))
        current += timedelta(seconds=config["metadata"]["file_duration"])

    data = np.concatenate(data)
    index = pd.date_range(
        start=datetime_start, periods=len(data), freq=f"{config['metadata']['dt']}s"
    )
    columns = np.arange(len(data[0])) * config["metadata"]["dx"]

    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df
