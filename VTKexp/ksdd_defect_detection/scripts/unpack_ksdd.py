import argparse
from pathlib import Path
import zipfile
import shutil

def unzip(zip_path: Path, out_dir: Path):
    print(f"[INFO] Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="data/raw", help="directory containing dataset zip(s)")
    ap.add_argument("--out", type=str, default="data/ksdd", help="output directory to extract")
    ap.add_argument("--clean", action="store_true", help="remove existing out directory first")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip found in {raw_dir}. Put dataset zip into this folder first.")

    for zp in zips:
        unzip(zp, out_dir)

    print("[OK] Done. Extracted to:", out_dir.resolve())

if __name__ == "__main__":
    main()
