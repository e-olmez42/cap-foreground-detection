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

        # Common parameters
        history = self.application.get_param(self.config, "history")
        detectShadows = self.application.get_param(self.config, "detectShadows")

        # -----------------------------
        # MOG2 MODEL
        # -----------------------------
        if model_type == "MOG2":

            varThreshold = self.application.get_param(self.config, "varThreshold")
            nMixtures = self.application.get_param(self.config, "nMixtures")
            shadowThreshold = self.application.get_param(self.config, "shadowThreshold")
            backgroundRatio = self.application.get_param(self.config, "backgroundRatio")
            varMin = self.application.get_param(self.config, "varMin")
            varMax = self.application.get_param(self.config, "varMax")
            varThresholdGen = self.application.get_param(self.config, "varThresholdGen")
            varInit = self.application.get_param(self.config, "varInit")
            complexityReductionThreshold = self.application.get_param(self.config, "complexityReductionThreshold")

            # Create model
            model = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=varThreshold,
                detectShadows=detectShadows
            )

            # Set advanced MOG2 parameters
            model.setNMixtures(nMixtures)
            model.setShadowThreshold(shadowThreshold)
            model.setBackgroundRatio(backgroundRatio)
            model.setVarMin(varMin)
            model.setVarMax(varMax)
            model.setVarThresholdGen(varThresholdGen)
            model.setVarInit(varInit)
            model.setComplexityReductionThreshold(complexityReductionThreshold)

        # -----------------------------
        # KNN MODEL
        # -----------------------------
        elif model_type == "KNN":

            dist2Threshold = self.application.get_param(self.config, "dist2Threshold")
            nSamples = self.application.get_param(self.config, "nSamples")
            kNNSamples = self.application.get_param(self.config, "kNNSamples")

            model = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2Threshold,
                detectShadows=detectShadows
            )

            model.setNSamples(nSamples)
            model.setkNNSamples(kNNSamples)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model
