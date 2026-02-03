import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # cartella progetto
sys.path.insert(0, str(ROOT))

SKIP = {
    "__init__",
}

def main():
    py_files = [p for p in ROOT.glob("*.py") if p.is_file()]
    failed = []

    print(f"Checking imports in: {ROOT}")
    for p in sorted(py_files):
        mod = p.stem
        if mod in SKIP:
            continue

        try:
            importlib.import_module(mod)
            print(f"[OK]  {mod}")
        except Exception as e:
            print(f"[FAIL] {mod}: {type(e).__name__}: {e}")
            failed.append((mod, e))

    if failed:
        print("\nSummary failures:")
        for mod, e in failed:
            print(f" - {mod}: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\nAll module imports OK.")

if __name__ == "__main__":
    main()

