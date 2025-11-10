import pandas as pd
from abc import ABC, abstractmethod
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import cv2


class Transform(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        pass


class AbsoluteValue(Transform):
    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        return df.abs()


class Normalize(Transform):
    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        return (df - df.min()) / (df.max() - df.min())


class ZScoreTransform(Transform):
    def __init__(
        self,
        window: int = 51,
        min_periods: int | None = None,
        eps: float = 1e-8,
    ):
        self.window = int(window)
        self.min_periods = (
            int(min_periods) if min_periods is not None else max(3, self.window // 2)
        )
        self.eps = float(eps)

    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        data = df.astype(float).copy()
        data = data.transpose()

        rolling_mean = data.rolling(
            window=self.window, center=True, min_periods=self.min_periods
        ).mean()

        ddof = 1 if self.window > 1 else 0
        rolling_std = data.rolling(
            window=self.window, center=True, min_periods=self.min_periods
        ).std(ddof=ddof)

        rolling_std = rolling_std.where(rolling_std > 0, other=np.nan).fillna(self.eps)

        z = (data - rolling_mean) / rolling_std
        z = z.transpose()

        return z


class MedianFilter(Transform):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = int(kernel_size)

    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        data = df.astype(float).copy()
        filtered_data = median_filter(
            data.values, size=(self.kernel_size, self.kernel_size), mode="nearest"
        )
        return pd.DataFrame(filtered_data, index=data.index, columns=data.columns)


class TotalVariationDenoising(Transform):
    def __init__(self, weight: float = 0.1):
        self.weight = float(weight)

    def apply(self, df: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        data = df.astype(float).copy()
        denoised_data = denoise_tv_chambolle(data.values, weight=self.weight)
        return pd.DataFrame(denoised_data, index=data.index, columns=data.columns)


class Resize(Transform):
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def apply(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        img = (df.values * 255).astype(np.uint8)
        resized_img = cv2.resize(img, (self.width, self.height))
        return resized_img