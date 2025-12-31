import cv2
import numpy as np
from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager


class ModelLoader:

    # =================================================
    # OpenCV Background Subtractor Wrapper
    # =================================================
    class BackgroundSubtractorWrapper:
        def __init__(self, cv_model, learning_rate):
            self.model = cv_model
            self.learning_rate = learning_rate

        def apply(self, image):
            return self.model.apply(image, learningRate=self.learning_rate)

    # =================================================
    # Frame Differencing
    # =================================================
    class FrameDifferencingWrapper:
        def __init__(self,  alpha=0.05):
            self.prev_gray = None
            self.alpha = alpha

        def apply(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                return np.zeros_like(gray)

            diff = cv2.absdiff(gray, self.prev_gray)
            self.prev_gray = gray
            return diff

    # =================================================
    # Running Average
    # =================================================
    class RunningAverageWrapper:
        def __init__(self, alpha=0.05):
            self.alpha = alpha
            self.bg = None

        def apply(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.bg is None:
                self.bg = gray.astype(np.float32)
                return np.zeros_like(gray)

            cv2.accumulateWeighted(gray, self.bg, self.alpha)
            bg_uint8 = cv2.convertScaleAbs(self.bg)

            diff = cv2.absdiff(gray, bg_uint8)
            return diff

    # =================================================
    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.logger = LoggerManager()

    # =================================================
    def load_model(self):
        model_type = self.application.get_param(self.config, "type")

        # ---------------- MOG2 ----------------
        if model_type == "MOG2":
            model = cv2.createBackgroundSubtractorMOG2(
                history=self.application.get_param(self.config, "history"),
                varThreshold=self.application.get_param(self.config, "varThreshold"),
                detectShadows=self.application.get_param(self.config, "detectShadows")
            )
            return self.BackgroundSubtractorWrapper(
                model,
                self.application.get_param(self.config, "learningRate")
            )

        # ---------------- KNN ----------------
        if model_type == "KNN":
            model = cv2.createBackgroundSubtractorKNN(
                history=self.application.get_param(self.config, "history"),
                dist2Threshold=self.application.get_param(self.config, "dist2Threshold"),
                detectShadows=self.application.get_param(self.config, "detectShadows")
            )
            return self.BackgroundSubtractorWrapper(
                model,
                self.application.get_param(self.config, "learningRate")
            )

        # ---------------- FRAME DIFF ----------------
        if model_type == "FrameDifferencing":
            return self.FrameDifferencingWrapper(
                self.application.get_param(self.config, "learningRate")
            )

        # ---------------- RUNNING AVG ----------------
        if model_type == "RunningAverage":
            return self.RunningAverageWrapper(
                self.application.get_param(self.config, "learningRate")
            )

        raise ValueError(f"Unsupported model type: {model_type}")
