"""
Download cat images for neuter detection dataset.
Uses icrawler to fetch from Bing/Google.
Categories:
  - neutered: cats with ear tipping (TNR mark — ear tip clipped)
  - intact: normal cats with intact ears
"""
import os
import shutil
import random
from icrawler.builtin import BingImageCrawler

BASE = os.path.expanduser("~/08_Drafts/CatNeuter/dataset")
RAW = os.path.expanduser("~/08_Drafts/CatNeuter/raw")

# Search queries
QUERIES = {
    "neutered": [
        "ear tipped cat",
        "TNR cat ear tip",
        "cat with clipped ear tip",
        "neutered feral cat ear",
        "ear tipped stray cat",
        "cat ear notch TNR",
        "community cat ear tip",
        "cat left ear tipped",
    ],
    "intact": [
        "cat face",
        "cat portrait",
        "stray cat",
        "cute cat ears",
        "cat sitting outside",
        "tabby cat",
        "black cat",
        "orange cat",
        "cat close up face",
        "domestic cat",
    ],
}

MAX_PER_QUERY = 80  # aim for ~500+ per class total
TRAIN_RATIO = 0.8

def download_images():
    for label, queries in QUERIES.items():
        raw_dir = os.path.join(RAW, label)
        os.makedirs(raw_dir, exist_ok=True)

        for i, query in enumerate(queries):
            save_dir = os.path.join(raw_dir, f"q{i:02d}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n>>> [{label}] Query {i+1}/{len(queries)}: '{query}'")

            crawler = BingImageCrawler(
                storage={"root_dir": save_dir},
                log_level=30,  # WARNING only
            )
            crawler.crawl(
                keyword=query,
                max_num=MAX_PER_QUERY,
                min_size=(100, 100),
            )

def organize_dataset():
    """Flatten raw downloads, dedup by size, split train/test."""
    for label in ["neutered", "intact"]:
        raw_dir = os.path.join(RAW, label)
        all_images = []
        seen_sizes = set()

        for root, dirs, files in os.walk(raw_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    path = os.path.join(root, f)
                    size = os.path.getsize(path)
                    if size > 1000 and size not in seen_sizes:  # skip tiny/broken, basic dedup
                        seen_sizes.add(size)
                        all_images.append(path)

        random.shuffle(all_images)
        split = int(len(all_images) * TRAIN_RATIO)
        train_imgs = all_images[:split]
        test_imgs = all_images[split:]

        for split_name, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(BASE, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            for i, src in enumerate(imgs):
                ext = os.path.splitext(src)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png'):
                    ext = '.jpg'
                dst = os.path.join(dst_dir, f"{label}_{i:04d}{ext}")
                shutil.copy2(src, dst)

        print(f"{label}: {len(all_images)} unique images → train: {len(train_imgs)}, test: {len(test_imgs)}")

if __name__ == "__main__":
    print("=== Step 1: Downloading images ===")
    download_images()
    print("\n=== Step 2: Organizing dataset ===")
    organize_dataset()
    print("\nDone! Dataset at:", BASE)
