"""Real-ESRGAN x2 inference with the original RRDBNet and tiling settings."""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

from . import core
from .infer import input_images, reserve_output, write_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    paths = input_images(args.input)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    import torch
    from torchvision.transforms.functional import rgb_to_grayscale
    # BasicSR 1.4.x compatibility with newer torchvision, as in source step 07.
    compatibility = types.ModuleType('torchvision.transforms.functional_tensor')
    compatibility.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault('torchvision.transforms.functional_tensor', compatibility)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; pass --device cpu for CPU inference.')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(scale=2, model_path=str(args.weights), model=model,
                           tile=256, tile_pad=20, pre_pad=0,
                           half=args.device == 'cuda', gpu_id=0 if args.device == 'cuda' else None,
                           device=torch.device(args.device))
    output = reserve_output(args.output)
    digest = hashlib.sha256()
    with args.weights.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    records = []
    for path in paths:
        record = {'image': path.name, 'success': False}
        try:
            image = core.imread_unicode(path)
            if image is None:
                raise ValueError('Image decoding failed')
            sr, _ = upsampler.enhance(image, outscale=2)
            if sr.shape[:2] != (image.shape[0] * 2, image.shape[1] * 2):
                raise RuntimeError('Unexpected SR output dimensions')
            write_image(output / f'{path.stem}.png', sr)
            record['success'] = True
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
        print(f'{path.name}: {"OK" if record["success"] else record["error"]}')
    (output / 'run.json').write_text(json.dumps({
        'device': args.device, 'weights_sha256': digest.hexdigest(),
        'scale': 2, 'tile': 256, 'tile_pad': 20, 'pre_pad': 0,
        'half': args.device == 'cuda', 'cases': records,
    }, indent=2), encoding='utf-8')
    if any(not row['success'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
