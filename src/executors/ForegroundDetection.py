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
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel
from capsules.ForegroundDetection.src.utils.utils import ModelLoader


class ForegroundDetection(Capsule):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.model = self.bootstrap.get("model")
        self.threshold = self.request.get_param("threshold")
        self.min_contour_area = self.request.get_param("minContourArea")
        self.model_type = self.request.get_param("type")
        self.type = self.request.get_param("type")
        if self.type == "RunningAverage":
            self.fps = self.redis_db.redis_get_flag("injection")
            self.frame_buffer = self.bootstrap.get("frame_buffer")
            self.frame_count = self.bootstrap.get("frame_count")
            self.bg_init_duration = self.request.get_param("bgInitDuration")
            self.bg_type = self.request.get_param("bgType")

        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return {"model": model, "frame_buffer": [],"frame_count": 0}

    # ------------------------------------------------
    def clean_mask(self, raw_mask):
        _, mask = cv2.threshold(raw_mask, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ------------------------------------------------
    def foreground_mask(self, image):
        raw_mask = self.model.apply(image)
        return self.clean_mask(raw_mask)

    # ------------------------------------------------
    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value.astype(np.uint8)

        fg_mask = self.foreground_mask(frame)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.detections = []
        vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            detection = Detection(
                boundingBox=BoundingBox(left=x, top=y, width=w, height=h),
                confidence=1.0,
                classId=0,
                classLabel="foreground",
                imgUID=self.uID,
                keyPoints=[]
            )
            self.detections.append(detection)

        img.value = vis
        self.image = Image.set_frame(
            img=img, package_uID=self.uID, redis_db=self.redis_db
        )

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
