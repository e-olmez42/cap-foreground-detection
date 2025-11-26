import os
import cv2
import sys
import numpy as np
from collections import Counter, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection.src.utils.response import build_response
from sdks.novavision.src.base.model import Detection, BoundingBox
from capsules.ForegroundDetection.src.utils.utils import ModelLoader
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel

CONSECUTIVE_FRAMES = 20
STABILITY_THRESHOLD = 100
MAX_ABSENCE_FRAMES = 200

class ForegroundDetection(Capsule):
    top_contour_dict = defaultdict(int)
    obj_detected_dict = defaultdict(int)
    frameno = 0

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.model = self.bootstrap.get("model")
        self.track_master = self.bootstrap.get("track_master")
        self.model_type = self.request.get_param("type")
        self.detections = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = ModelLoader(config=config).load_model()
        return {"model": model,"track_master":[]}

    def clean_mask(self, raw_mask):

        if self.model_type == "MOG2":
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            _, mask = cv2.threshold(raw_mask, 250, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        elif self.model_type == "KNN":
            raw_mask[raw_mask == 127] = 0
            _, mask = cv2.threshold(raw_mask, 100, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

        else:
            mask = raw_mask

        return mask

    def foreground_mask(self, image):
        return self.model.apply(image)

    def run(self):
        current_frameno = ForegroundDetection.frameno
        ForegroundDetection.frameno += 1

        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        frame = img.value

        img_uint8 = frame .astype(np.uint8)
        fg_mask = self.foreground_mask(img_uint8)
        motion_mask = self.clean_mask(fg_mask)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 500
        average_object_confidence = 1.0
        self.detections = []

        fg_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area or area>20000:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0: continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            sumcxcy = cx + cy


            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(fg_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.track_master.append([sumcxcy, current_frameno])

            unique_frames = set(j for i, j in self.track_master)
            if len(unique_frames) > CONSECUTIVE_FRAMES:
                min_frameno = min(j for i, j in self.track_master)
                self.track_master = [item for item in self.track_master if item[1] != min_frameno]

            # Sabitlik Sayımı
            countcxcy = Counter(i for i, j in self.track_master)

            for centroid_val, count in countcxcy.items():
                if count >= CONSECUTIVE_FRAMES:
                    ForegroundDetection.top_contour_dict[centroid_val] += 1

            if sumcxcy in self.top_contour_dict and self.top_contour_dict[sumcxcy] > STABILITY_THRESHOLD:
                self.obj_detected_dict[sumcxcy] = current_frameno


            detection = Detection(
                boundingBox=BoundingBox(left=x, top=y, width=w, height=h),
                confidence=round(float(average_object_confidence), 2),
                classId=0,
                classLabel="foreground",
                imgUID=self.uID,
                keyPoints=[]
            )
            self.detections.append(detection)

        keys_to_remove = []
        for centroid_val, last_frame in list(ForegroundDetection.obj_detected_dict.items()):
            if current_frameno - last_frame > MAX_ABSENCE_FRAMES:
                keys_to_remove.append(centroid_val)
                ForegroundDetection.top_contour_dict[centroid_val] = 0

        for key in keys_to_remove:
            del ForegroundDetection.obj_detected_dict[key]


        img.value = fg_color
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        return build_response(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
