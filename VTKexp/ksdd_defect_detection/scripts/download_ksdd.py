import argparse
from pathlib import Path
import sys
import requests
from tqdm import tqdm

DEFAULT_URLS = [
    # ViCoS short-link (fine annotations)
    "https://go.vicos.si/kolektorsdd",
    # Prepared TF package (also contains masks + splits) - might be easier if accessible
    "https://data.vicos.si/datasets/KSDD/KolektorSDD-dilate%3D5-tensorflow.zip",
    "https://data.vicos.si/datasets/KSDD/KolektorSDD-dilate=5-tensorflow.zip",
    # Splits
    "https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip",
]

def download(url: str, out_path: Path, timeout=60):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        desc = f"Downloading {url}"
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw", help="output dir to store zips")
    ap.add_argument("--url", type=str, default="", help="custom dataset url (optional)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = [args.url] if args.url else DEFAULT_URLS

    ok = False
    for url in urls:
        try:
            name = url.split("/")[-1]
            if not name:
                name = "KolektorSDD.zip"
            if not name.lower().endswith(".zip"):
                name += ".zip"
            out_path = out_dir / name
            print(f"[INFO] Try: {url}")
            download(url, out_path)
            print(f"[OK] Saved: {out_path}")
            ok = True
            # Stop after first success
            break
        except Exception as e:
            print(f"[WARN] Failed: {url} -> {e}")

    if not ok:
        print("\n[ERROR] All downloads failed.")
        print("你可以：")
        print("1) 打开 https://www.vicos.si/resources/kolektorsdd/ 手动下载 fine annotations zip")
        print("2) 将zip放到 data/raw/，然后运行 scripts/unpack_ksdd.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
