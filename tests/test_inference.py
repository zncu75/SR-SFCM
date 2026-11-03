import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

import cv2
import numpy as np

from sr_sfcm import core
from sr_sfcm.infer import input_images, main, write_image


@contextmanager
def test_directory():
    # Use inherited permissions, including when running under a Windows sandbox.
    base = Path(tempfile.gettempdir()).resolve()
    root = base / ('sr-sfcm-test-' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        assert root.resolve().parent == base and root.name.startswith('sr-sfcm-test-')
        shutil.rmtree(root)


class InferenceChecks(unittest.TestCase):
    def test_fcm_really_updates_centers(self):
        # Overlapping asymmetric observations should move beyond K-means init.
        X = np.array([[0.0], [0.1], [0.3], [0.8], [1.0], [2.0]])
        initial = core.membership_from_centers(X, core.init_centers_kmeans(X), core.FCM_M)
        U, iterations, converged, delta = core.run_fcm(X)
        self.assertGreater(iterations, 1)
        self.assertTrue(converged)
        self.assertLess(delta, core.FCM_TOL)
        self.assertGreater(np.max(np.abs(U - initial)), 1e-3)
        np.testing.assert_allclose(U.sum(axis=1), 1.0)
        centers = ((U ** core.FCM_M).T @ X) / (U ** core.FCM_M).sum(axis=0)[:, None]
        np.testing.assert_allclose(U, core.membership_from_centers(X, centers, core.FCM_M), atol=2e-5)

    @staticmethod
    def image():
        image = np.zeros((128, 128, 3), np.uint8)
        cv2.ellipse(image, (64, 64), (43, 53), 0, 0, 360, (90, 90, 180), -1)
        cv2.ellipse(image, (64, 57), (22, 32), 0, 0, 360, (170, 185, 200), -1)
        return image

    def test_segmentation_partitions_roi_deterministically(self):
        image = self.image()
        roi = core.extract_tongue_roi(image)
        first = core.method_D_slic_spatial_fcm(image, roi)
        second = core.method_D_slic_spatial_fcm(image, roi)
        np.testing.assert_array_equal(first['coating'], second['coating'])
        self.assertFalse(np.any(first['coating'] & first['body']))
        np.testing.assert_array_equal(first['coating'] | first['body'], roi)
        self.assertEqual(first['coating'].shape, image.shape[:2])

    def test_cli_outputs_and_refuses_overwrite(self):
        with test_directory() as folder:
            root = Path(folder)
            write_image(root / 'input' / 'synthetic.png', self.image())
            args = ['--input', str(root/'input'), '--output', str(root/'output')]
            main(args)
            manifest = json.loads((root/'output'/'run.json').read_text())
            self.assertTrue(manifest['cases'][0]['success'])
            self.assertTrue((root/'output'/'coating_masks'/'synthetic.png').is_file())
            with self.assertRaises(ValueError):
                main(args)

    def test_decode_failure_and_duplicate_stems(self):
        with test_directory() as folder:
            root = Path(folder)
            (root/'input').mkdir()
            (root/'input'/'broken.png').write_bytes(b'not an image')
            with self.assertRaises(SystemExit) as result:
                main(['--input', str(root/'input'), '--output', str(root/'output')])
            self.assertEqual(result.exception.code, 1)
            manifest = json.loads((root/'output'/'run.json').read_text())
            self.assertFalse(manifest['cases'][0]['success'])
            (root/'input'/'broken.jpg').write_bytes(b'not an image')
            with self.assertRaises(ValueError):
                input_images(root/'input')


if __name__ == '__main__':
    unittest.main()
