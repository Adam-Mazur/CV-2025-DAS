from src.method import Method
import numpy as np
import pandas as pd
import cv2


class HoughProbMethod(Method):
    def __init__(
        self,
        threshold1: int = 50,
        threshold2: int = 150,
        rho: int = 1,
        theta: float = np.pi / 180,
        hough_threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.rho = rho
        self.theta = theta
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def detect(self, data: pd.DataFrame) -> list[tuple]:
        data = (data.values * 255).astype(np.uint8)

        edges = cv2.Canny(data, self.threshold1, self.threshold2)
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        return [line[0] for line in lines]
