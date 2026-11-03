"""Step 08: generate candidate regions using the corrected inference kernels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr_sfcm.infer import main

if __name__ == '__main__':
    main()
