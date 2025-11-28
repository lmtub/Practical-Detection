#!/usr/bin/env python3
import os
import csv
from pathlib import Path

# Thư mục chứa các gói npm/malicious_intent
DATASET_ROOT = Path("/home/thang/malicious-software-packages-dataset/samples/npm/malicious_intent")

# Thư mục metadata để lưu malicious.csv
METADATA_DIR = Path("/home/thang/malicious-software-packages-dataset/metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = METADATA_DIR / "malicious.csv"

def main():
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Cấu trúc: DATASET_ROOT / <package> / <version> / ...
        for pkg_dir in sorted(DATASET_ROOT.iterdir()):
            if not pkg_dir.is_dir():
                continue

            package = pkg_dir.name

            for ver_dir in sorted(pkg_dir.iterdir()):
                if not ver_dir.is_dir():
                    continue

                version = ver_dir.name

                # Chỉ lấy những version đã có change-features.csv
                cf_path = ver_dir / "change-features.csv"
                if not cf_path.is_file():
                    # Nếu bạn muốn lấy tất cả bất kể có feature hay không thì bỏ if này
                    continue

                writer.writerow([package, version])

    print(f"Đã ghi {OUT_PATH}")

if __name__ == "__main__":
    main()
