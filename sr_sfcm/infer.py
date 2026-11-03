"""Apply one segmentation method to a directory, without statistical analysis."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import core

METHODS = {
    'sr-sfcm': core.method_D_slic_spatial_fcm,
    'kmeans': core.method_A_kmeans,
    'fcm': core.method_B_fcm,
    'spatial-fcm': core.method_C_spatial_fcm,
}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def input_images(directory):
    if not directory.is_dir():
        raise ValueError(f'Input directory does not exist: {directory}')
    paths = sorted(p for p in directory.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not paths:
        raise ValueError(f'No supported images in {directory}')
    names = [p.stem.casefold() for p in paths]
    if len(names) != len(set(names)):
        raise ValueError('Image stems must be unique, including across subdirectories.')
    return paths


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not core.imwrite_png_unicode(path, image):
        raise OSError(f'Could not write {path}')


def reserve_output(directory):
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Output must be empty to prevent overwriting: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, default='sr-sfcm')
    args = parser.parse_args(argv)
    paths = input_images(args.input)
    output = reserve_output(args.output)
    records = []
    for path in paths:
        record = {'image': path.name, 'method': args.method, 'success': False}
        try:
            image = core.imread_unicode(path)
            if image is None:
                raise ValueError('Image decoding failed')
            roi = core.extract_tongue_roi(image)
            if roi.sum() < 2:
                raise ValueError('Tongue ROI is empty or too small')
            result = METHODS[args.method](image, roi)
            coating, body = result['coating'], result['body']
            if np.any((coating > 0) & (body > 0)) or not np.array_equal(
                    (coating > 0) | (body > 0), roi > 0):
                raise RuntimeError('Candidate masks do not partition the ROI')
            for name, mask in [('tongue_roi', roi), ('coating_masks', coating), ('body_masks', body)]:
                write_image(output / name / f'{path.stem}.png', mask.astype(np.uint8) * 255)
            for name, mask in [('coating_images', coating), ('body_images', body)]:
                write_image(output / name / f'{path.stem}.png', cv2.bitwise_and(image, image, mask=mask))
            overlay = image.copy()
            contours, _ = cv2.findContours(coating.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            write_image(output / 'overlays' / f'{path.stem}.png', overlay)
            record.update({key: result[key] for key in [
                'iterations', 'converged', 'final_delta', 'uncertain_fraction']})
            record['success'] = True
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
        print(f'{path.name}: {"OK" if record["success"] else record["error"]}')
    # Non-finite optional diagnostics (e.g. K-means membership) become JSON null.
    for row in records:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    config = {key: value for key, value in vars(core).items()
              if key.isupper() and isinstance(value, (int, float))}
    (output / 'run.json').write_text(json.dumps({
        'input': str(args.input.resolve()), 'method': args.method,
        'parameters': config, 'cases': records,
    }, indent=2, allow_nan=False), encoding='utf-8')
    if any(not row['success'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
