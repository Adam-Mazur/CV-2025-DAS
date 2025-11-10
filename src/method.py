from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Method(ABC):
    @abstractmethod
    def detect(self, data: pd.DataFrame | np.ndarray) -> list[tuple]:
        pass
