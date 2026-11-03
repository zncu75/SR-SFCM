"""Step 07: generate x2 SR images; run from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr_sfcm.super_resolution import main

if __name__ == '__main__':
    main()
