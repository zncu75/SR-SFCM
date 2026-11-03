"""Generate Original-D/SR-D masks for review, without ablation statistics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from sr_sfcm import gain as formal
from sr_sfcm.infer import input_images, reserve_output, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--sr', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    originals = input_images(args.original)
    sr_lookup = {p.stem: p for p in input_images(args.sr)}
    if set(p.stem for p in originals) != set(sr_lookup):
        raise ValueError('Original and SR image stems must match exactly')
    output = reserve_output(args.output)
    rows = []
    for path in originals:
        row = {'image': path.name, 'success': False}
        try:
            original = formal.imread_unicode(path)
            sr = formal.imread_unicode(sr_lookup[path.stem])
            if original is None or sr is None:
                raise ValueError('Image decoding failed')
            h, w = original.shape[:2]
            if sr.shape[:2] != (2*h, 2*w):
                raise ValueError('SR image must have exactly twice the original dimensions')
            roi = formal.extract_tongue_roi(original)
            roi_sr = cv2.resize(roi, (2*w, 2*h), interpolation=cv2.INTER_NEAREST)
            if roi.sum() < 2 or roi_sr.sum() < 2:
                raise ValueError('Empty or too-small ROI')
            original_result = formal.segment_candidate_regions(original, roi, (h, w))
            sr_result = formal.segment_candidate_regions(sr, roi_sr, (h, w))
            aligned = sr_result['coating']
            write_image(output/'masks/original_coating'/f'{path.stem}.png', original_result['coating']*255)
            write_image(output/'masks/sr_coating_resized'/f'{path.stem}.png', aligned*255)
            row.update(success=True, original_converged=original_result['converged'],
                       sr_converged=sr_result['converged'])
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)
        print(path.name, 'OK' if row['success'] else row['error'])
    (output/'run.json').write_text(json.dumps({'cases':rows},indent=2),encoding='utf-8')
    if any(not row['success'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
