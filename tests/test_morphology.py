import unittest
import numpy as np
from tumor_analysis.infrastructure.analyzer import OpenCVTumorAnalyzer


class TestMorphologyAnalyzer(unittest.TestCase):
    """Unit tests to verify physical tumor area and brain tissue percentages."""

    def setUp(self):
        self.analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)

    def test_zero_tumor_mask(self):
        """Verify that an empty mask returns zero area and zero occupancy."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 10  # Dark brain background
        
        res = self.analyzer.analyze(
            mask=mask,
            original_image=img,
            pixel_spacing_mm=1.0
        )
        
        self.assertEqual(res.pixel_count, 0)
        self.assertEqual(res.tumor_area_mm2, 0.0)
        self.assertEqual(res.tumor_percentage_brain, 0.0)

    def test_synthetic_tumor_mask(self):
        """Verify area calculations with custom dimensions and spacing."""
        # Create a 256x256 image with a central 10x10 active square mask (100 pixels)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:110, 100:110] = 1
        
        # Brain outline (200x200 pixel box = 40,000 pixels)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[28:228, 28:228] = 50  # Over background threshold of 10
        
        # Test spacing = 1.0 mm (100 pixels * 1.0 * 1.0 = 100 mm^2)
        res_1 = self.analyzer.analyze(
            mask=mask,
            original_image=img,
            pixel_spacing_mm=1.0
        )
        self.assertEqual(res_1.pixel_count, 100)
        self.assertEqual(res_1.tumor_area_mm2, 100.0)
        # 100 tumor pixels / 40,000 brain pixels * 100% = 0.25%
        self.assertAlmostEqual(res_1.tumor_percentage_brain, 0.25)

        # Test spacing = 0.5 mm (100 pixels * 0.5 * 0.5 = 25 mm^2)
        res_2 = self.analyzer.analyze(
            mask=mask,
            original_image=img,
            pixel_spacing_mm=0.5
        )
        self.assertEqual(res_2.tumor_area_mm2, 25.0)


if __name__ == "__main__":
    unittest.main()
