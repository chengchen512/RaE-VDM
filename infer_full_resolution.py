"""Native-resolution, raised-cosine tiled deterministic inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from raevdm.infer import main

if __name__ == "__main__":
    main(["--tiled", *sys.argv[1:]])
