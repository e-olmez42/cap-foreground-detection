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


class ForegroundDetection(Capsule):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.model = self.bootstrap.get("model")
        self.min_contour_area = self.request.get_param("minContourArea")
        self.model_type = self.request.get_param("type")
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        config = config.copy()
        model = ModelLoader(config=config).load_model()

        return {
            "model": model
        }

    def clean_mask(self, raw_mask):

        if self.model_type == "MOG2":
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            _, mask = cv2.threshold(raw_mask, 100, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        elif self.model_type == "KNN":
            _, mask = cv2.threshold(raw_mask, 100, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

        else:
            mask = raw_mask

        return mask

    def foreground_mask(self, image):
        mask = self.model.apply(image)
        final_mask = self.clean_mask(mask)
        return final_mask

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value

        img_uint8 = frame.astype(np.uint8)
        fg_mask = self.foreground_mask(img_uint8)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        average_object_confidence = 1.0
        self.detections = []

        fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(fg_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(fg_color, "Moving Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
