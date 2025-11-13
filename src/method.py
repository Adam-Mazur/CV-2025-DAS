from abc import ABC, abstractmethod
import pandas as pd


class Method(ABC):
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> list[tuple]:
        pass
