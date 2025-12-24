import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection.src.utils.response import build_response
from sdks.novavision.src.base.model import Detection, BoundingBox
from capsules.ForegroundDetection.src.utils.utils import ModelLoader
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel


class AODForegroundDetection(Capsule):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        config_fast = config.copy()
        config_fast['learning_rate'] = "short"
        model_fast = ModelLoader(config=config_fast).load_model()

        config_medium = config.copy()
        config_medium['learning_rate'] = "long"
        model_medium = ModelLoader(config=config_medium).load_model()

        return {
            "model_fast": model_fast,
            "model_medium": model_medium,
            "model_slow": None
        }


    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value

        img_uint8 = frame.astype(np.uint8)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
