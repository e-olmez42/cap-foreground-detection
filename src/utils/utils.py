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
        model_type = self.application.get_param(config=self.config, name="type")

        varThreshold = self.application.get_param(config=self.config, name="varThreshold")
        history = self.application.get_param(config=self.config, name="history")
        detectShadows = self.application.get_param(config=self.config, name="detectShadows")

        if model_type == "MOG2":
            model = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=varThreshold,
                detectShadows=detectShadows
            )
            model.setNMixtures(3)  # Gauss sayısı
            model.setShadowThreshold(0.7)  #Gölgelerin katsayısı
            model.setBackgroundRatio(0.8)  #arka tarafın oranı
            model.setVarMin(4 * 4)
            model.setVarMax(75 * 75)

        elif model_type == "KNN":
            model = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=varThreshold,
                detectShadows=detectShadows
            )
            model.setNSamples(20)
            model.setkNNSamples(2)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

