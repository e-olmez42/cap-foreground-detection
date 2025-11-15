import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection.src.utils.response import build_response
from sdks.novavision.src.base.model import Detection, BoundingBox
from capsules.ForegroundDetection.src.utils.utils import ModelLoader
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel


class ForegroundDetection(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.model = self.bootstrap.get("model")
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return {"model": model}

    def foreground_mask(self, image):
        return self.model.apply(image, learningRate=-1)

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value
        fg_mask = self.foreground_mask(frame)

        # Temizlik işlemleri
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _, motion_mask = cv2.threshold(fg_mask, 1, 255, cv2.THRESH_BINARY)

        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 500
        average_object_confidence = 1.0
        self.detections = []

        fg_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(fg_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detection = Detection(
                boundingBox=BoundingBox(left=x, top=y, width=w, height=h),
                confidence=round(float(average_object_confidence), 2),
                classId=0,
                classLabel="foreground",
                imgUID=self.uID,
                keyPoints=[]
            )
            self.detections.append(detection)

        img.value = fg_color
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
