#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import argparse
from pathlib import Path

# --- DANH SÁCH CÁC FILE CẦN BỎ QUA ---
# Thêm tên các file rác bạn muốn code lờ đi vào đây
IGNORE_FILES = {
    "change-features.csv",  # File do tool sinh ra
    ".DS_Store",            # File rác của Mac
    "Thumbs.db",            # File rác của Windows
    ".git",                 # Folder git
    ".gitignore",
    "__pycache__"
}

def find_real_root(base_path: Path) -> Path:
    # (Giữ nguyên logic cũ)
    if base_path.name == "package.json" and base_path.is_file():
        return base_path.parent
    if (base_path / "package.json").is_file():
        return base_path
    candidates = []
    for root, dirnames, filenames in os.walk(base_path):
        if "node_modules" in dirnames:
            dirnames.remove("node_modules")
        if "package.json" in filenames:
            candidates.append(Path(root))
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))

def compute_hash(root_dir: Path) -> str:
    m = hashlib.md5()
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 1. Lọc folder rác (như .git)
        dirnames[:] = [d for d in dirnames if d not in IGNORE_FILES]
        dirnames.sort()
        
        filenames.sort()

        for filename in filenames:
            # 2. Lọc file rác (QUAN TRỌNG)
            if filename in IGNORE_FILES:
                continue
                
            # Logic tính toán cũ...
            file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(file_path, root_dir)
            
            m.update(f"{relative_path}\n".encode("utf-8"))

            if filename == "package.json":
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        pkg = json.load(f)
                    if "name" in pkg: pkg["name"] = ""
                    if "version" in pkg: pkg["version"] = ""
                    # Xóa luôn scripts nếu muốn hash thoáng hơn (tùy chọn)
                    # if "scripts" in pkg: del pkg["scripts"] 
                    m.update(json.dumps(pkg, sort_keys=True).encode("utf-8"))
                except Exception:
                    with open(file_path, "rb") as f:
                        m.update(f.read())
            else:
                with open(file_path, "rb") as f:
                    m.update(f.read())
                    
    return m.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Đường dẫn thư mục gói")
    args = parser.parse_args()

    target_path = Path(args.path).resolve()
    
    if not target_path.exists():
        print(f"[LỖI] Không tìm thấy: {target_path}")
        sys.exit(1)

    real_root = find_real_root(target_path)
    if not real_root:
        print("[LỖI] Không thấy package.json")
        sys.exit(1)
        
    print(f"--- Checking: {real_root.name} ---")
    print(f"    (Đang bỏ qua: {', '.join(IGNORE_FILES)})") # Báo cho biết đang lọc

    try:
        hash_val = compute_hash(real_root)
        print(f"HASH: \033[93m{hash_val}\033[0m")
    except Exception as e:
        print(f"[LỖI] {e}")

if __name__ == "__main__":
    main()
