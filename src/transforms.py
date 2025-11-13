import pandas as pd
from abc import ABC, abstractmethod
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import cv2


class Transform(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class AbsoluteValue(Transform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.abs()


class Normalize(Transform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        min_val = df.min().min()
        max_val = df.max().max()
        return (df - min_val) / (max_val - min_val)


class Clip(Transform):
    def __init__(self, first_percentile: float = 1.0, last_percentile: float = 99.0):
        self.first_percentile = float(first_percentile)
        self.last_percentile = float(last_percentile)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        lower_bound = np.percentile(df.values, self.first_percentile)
        upper_bound = np.percentile(df.values, self.last_percentile)
        return df.clip(lower=lower_bound, upper=upper_bound)


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

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.astype(float).copy()
        filtered_data = median_filter(
            data.values, size=(self.kernel_size, self.kernel_size), mode="nearest"
        )
        return pd.DataFrame(filtered_data, index=data.index, columns=data.columns)


class TotalVariationDenoising(Transform):
    def __init__(self, weight: float = 0.1):
        self.weight = float(weight)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.astype(float).copy()
        denoised_data = denoise_tv_chambolle(data.values, weight=self.weight)
        return pd.DataFrame(denoised_data, index=data.index, columns=data.columns)


class NonLocalMeansDenoising(Transform):
    def __init__(
        self, h: float = 20, templateWindowSize: int = 7, searchWindowSize: int = 35
    ):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def apply(self, df):
        data = (df.values * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(
            data,
            None,
            h=self.h,
            templateWindowSize=self.templateWindowSize,
            searchWindowSize=self.searchWindowSize,
        )
        denoised = denoised.astype(float) / 255.0
        return pd.DataFrame(denoised, index=df.index, columns=df.columns)


class Resize(Transform):
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        img = (df.values * 255).astype(np.uint8)
        resized_img = cv2.resize(img, (self.width, self.height))
        resized_img = resized_img.astype(float) / 255.0

        new_freq = (
            (df.index[1] - df.index[0]).total_seconds()
            * df.shape[0]
            / resized_img.shape[0]
        )

        new_index = pd.date_range(
            start=df.index[0], periods=resized_img.shape[0], freq=f"{new_freq}s"
        )

        new_columns = (
            np.arange(resized_img.shape[1])
            * (df.columns[1] - df.columns[0])
            * df.shape[1]
            / resized_img.shape[1]
        )

        return pd.DataFrame(resized_img, index=new_index, columns=new_columns)
