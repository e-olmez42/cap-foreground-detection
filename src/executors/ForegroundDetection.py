
import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection.src.utils.response import build_response
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel


class ForegroundDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.history = self.request.get_param("history")
        self.varThreshold = self.request.get_param("varThreshold")
        self.detectShadows = self.request.get_param("detectShadows")
        self.image = self.request.get_param("inputImage")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=self.detectShadows
        )
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def foreground_mask(self, image):
        mask = self.bg_subtractor.apply(image)
        return mask

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        fg_mask = self.foreground_mask(img.value)
        fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        img.value = fg_color
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
