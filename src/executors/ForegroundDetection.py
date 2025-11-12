"""
Foreground Detection using MOG2 (Mixture of Gaussians v2)

This component applies background subtraction to extract moving objects (foreground)
from static background using OpenCV’s BackgroundSubtractorMOG2.
"""

import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.ForegroundDetection.src.utils.response import build_response
from capsules.ForegroundDetection.src.models.PackageModel import PackageModel


class Package(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # Config parametreleri
        self.history = self.request.get_param("history")
        self.varThreshold = self.request.get_param("varThreshold")
        self.detectShadows = self.request.get_param("detectShadows")
        self.image = self.request.get_param("inputImage")

        # MOG2 oluştur
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=self.detectShadows
        )

    @staticmethod
    def bootstrap(config: dict) -> dict:
        """Initialize required runtime objects or cache"""
        return {}

    def foreground_mask(self, image):
        """Uygulanan foreground extraction işlemi"""
        mask = self.bg_subtractor.apply(image)
        return mask

    def run(self):
        """Ana çalıştırma fonksiyonu"""
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)

        # Ön plan maskesi oluştur
        fg_mask = self.foreground_mask(img.value)

        # Maskeyi 3 kanala dönüştür (isteğe bağlı)
        fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Çıktı olarak kaydet
        img.value = fg_color
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)

        # Model cevabı oluştur
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
