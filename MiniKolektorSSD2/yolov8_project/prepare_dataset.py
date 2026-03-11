"""
Prepare a YOLO-compatible flat dataset from MiniKolektorSSD2.
Normal images are nested in subdirs — flatten them.
"""
import os
import shutil

SRC = os.path.expanduser("~/08_Drafts/MiniKolektorSSD2")
DST = os.path.expanduser("~/08_Drafts/MiniKolektorSSD2_flat")

for split in ["train", "test"]:
    for cls in ["abnormal", "normal"]:
        src_dir = os.path.join(SRC, split, cls)
        dst_dir = os.path.join(DST, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        count = 0
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    src_path = os.path.join(root, f)
                    # Make unique name using parent dir
                    rel = os.path.relpath(root, src_dir).replace(os.sep, "_")
                    if rel == ".":
                        dst_name = f
                    else:
                        dst_name = f"{rel}_{f}"
                    dst_path = os.path.join(dst_dir, dst_name)
                    os.symlink(src_path, dst_path)
                    count += 1
        print(f"{split}/{cls}: {count} images")

print("Done! Dataset at:", DST)
