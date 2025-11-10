from src.method import Method
import numpy as np
import pandas as pd
import cv2


class HoughMethod(Method):
    def __init__(
        self,
        threshold1: int = 50,
        threshold2: int = 150,
        rho: int = 1,
        theta: float = np.pi / 180,
        hough_threshold: int = 100,
    ):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.rho = rho
        self.theta = theta
        self.hough_threshold = hough_threshold

    def detect(self, data: pd.DataFrame | np.ndarray) -> list[tuple]:
        if isinstance(data, pd.DataFrame):
            data = (data.values * 255).astype(np.uint8)

        edges = cv2.Canny(data, self.threshold1, self.threshold2)
        lines = cv2.HoughLines(edges, self.rho, self.theta, self.hough_threshold)

        if lines is None:
            return []

        return [line[0] for line in lines]

