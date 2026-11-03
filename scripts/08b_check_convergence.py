"""Check the operational run manifest produced by inference/mask preparation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding='utf-8'))['cases']
    if not cases:
        raise ValueError('Manifest contains no cases')
    failures = []
    for row in cases:
        if not row.get('success'):
            failures.append(row['image'])
            continue
        states = [value for key,value in row.items() if key.endswith('converged')]
        if not states:
            raise ValueError('Manifest must contain convergence diagnostics, not only SR generation records')
        if not row.get('success') or not all(value is True for value in states):
            failures.append(row['image'])
    print(f'{len(cases)} cases checked; {len(failures)} failed or nonconverged')
    for name in failures:
        print(name)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
