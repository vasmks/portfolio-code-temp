from __future__ import annotations

import unittest

import numpy as np

from toontra.modules import Yolo26BubbleDetector


class Yolo26DetectorContractTests(unittest.TestCase):
    def test_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(iou_threshold=-0.1)
        with self.assertRaises(ValueError):
            Yolo26BubbleDetector(imgsz=0)

    def test_detect_returns_contract_compliant_detections(self) -> None:
        # A blank page checks the BubbleDetector contract on a real inference
        # pass; detector accuracy is evaluated separately.
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        detector = Yolo26BubbleDetector()
        detections = detector.detect(image)
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertGreaterEqual(detection.box.x1, 0)
            self.assertGreaterEqual(detection.box.y1, 0)
            self.assertLessEqual(detection.box.x2, 256)
            self.assertLessEqual(detection.box.y2, 256)
            self.assertTrue(0.0 <= detection.score <= 1.0)
        self.assertIsNotNone(detector.metadata.sha256)

    def test_detect_is_sorted_top_to_bottom(self) -> None:
        image = np.full((512, 384, 3), 255, dtype=np.uint8)
        detections = Yolo26BubbleDetector().detect(image)
        boxes = [item.box for item in detections]
        self.assertEqual(
            boxes,
            sorted(boxes, key=lambda box: (box.y1, box.x1, box.y2, box.x2)),
        )


if __name__ == "__main__":
    unittest.main()
