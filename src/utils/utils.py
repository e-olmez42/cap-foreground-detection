import cv2
from sdks.novavision.src.base.logger import LoggerManager
from sdks.novavision.src.base.application import Application


class ModelLoader:

    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.logger = LoggerManager()
        self.executor = self.application.get_param(config=config, name="ConfigExecutor")["name"]

    def load_model(self):
        model_type = self.application.get_param(self.config, "type")

        history = self.application.get_param(self.config, "history")
        detectShadows = self.application.get_param(self.config, "detectShadows")

        if model_type == "MOG2":
            varThreshold = self.application.get_param(self.config, "varThreshold")

            model = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=varThreshold,
                detectShadows=detectShadows
            )

            # -------------------------
            # Ek parametreler
            # -------------------------
            model.setNMixtures(3)               # Gaussian count
            model.setShadowThreshold(0.7)        # Shadow detection factor
            model.setBackgroundRatio(0.8)        # BG probability
            model.setVarMin(16)                  # Min variance
            model.setVarMax(5625)                # Max variance (75*75)

        elif model_type == "KNN":
            dist2Threshold = self.application.get_param(self.config, "dist2Threshold", default=400)

            model = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2Threshold,
                detectShadows=detectShadows
            )

            # -------------------------
            # KNN parametreleri
            # -------------------------
            model.setNSamples(20)
            model.setkNNSamples(2)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model
