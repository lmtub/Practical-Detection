#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import concurrent.futures
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

# ==================== Helper: entropy ====================

def shannon_entropy(data: bytes) -> float:
    """Tính Shannon entropy của chuỗi bytes (0 nếu rỗng)."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


# ==================== Keyword / pattern ====================

SUSPICIOUS_KEYWORDS = [
    "password", "passwd", "token", "secret", "apikey", "api_key",
    "ssh", "wallet", "bitcoin", "ethereum", "monero",
]

NETWORK_KEYWORDS = [
    "http://", "https://", "ftp://", "ws://", "wss://",
    "xmlhttprequest", "socket.io", "axios", "fetch(",
]

FS_KEYWORDS = [
    "require('fs'", 'require("fs"', "from 'fs'", 'from "fs"',
    "fs.readfile", "fs.writefile", "fs.appendfile",
    "fs.createwritestream", "fs.createreadstream",
]

CHILD_PROCESS_KEYWORDS = [
    "require('child_process'", 'require("child_process"',
    "child_process.exec", "child_process.spawn",
    ".exec(", ".spawn(", ".fork(",
]

PROCESS_KEYWORDS = [
    "process.env", "process.argv", "process.platform",
]

ENCODING_KEYWORDS = [
    "buffer.from(", "buffer.alloc(", "base64",
    "atob(", "btoa(", "string.fromcharcode", "fromcharcode",
    "decodeuri(", "decodeuricomponent(", "unescape(",
]

EVAL_KEYWORDS = [
    "eval(", "function(", "new function(",
]

INSTALL_SCRIPT_NAMES = [
    "install", "preinstall", "postinstall",
    "prepare", "prepublish", "prepublishonly", "postpublish",
    "preuninstall", "uninstall", "postuninstall",
]

SHELL_COMMANDS = [
    "curl", "wget", "ping", "nslookup",
]

# ==================== Phân tích nội dung text ====================

def analyze_js_ts_text(text: str) -> Tuple[Dict[str, float], float, int]:
    """
    Phân tích 1 file .js/.ts từ string:
    - Đếm các pattern đáng ngờ
    - Tính entropy dựa trên bytes utf-8
    - Trả về (feature_dict, entropy, số dòng)
    """
    features: Dict[str, float] = {}

    lower = text.lower()
    lines_count = len(text.splitlines())
    data = text.encode("utf-8", errors="ignore")
    ent = shannon_entropy(data)

    # đếm keyword nhạy cảm
    for kw in SUSPICIOUS_KEYWORDS:
        features[f"suspicious_string:{kw}"] = lower.count(kw)

    # network
    for kw in NETWORK_KEYWORDS:
        features[f"network_usage:{kw}"] = lower.count(kw)

    # file system
    for kw in FS_KEYWORDS:
        features[f"fs_usage:{kw}"] = lower.count(kw)

    # child_process
    for kw in CHILD_PROCESS_KEYWORDS:
        features[f"child_process_usage:{kw}"] = lower.count(kw)

    # process.*
    for kw in PROCESS_KEYWORDS:
        features[f"process_usage:{kw}"] = lower.count(kw)

    # encoding / obfuscation
    for kw in ENCODING_KEYWORDS:
        features[f"encoding_usage:{kw}"] = lower.count(kw)

    # eval / Function
    for kw in EVAL_KEYWORDS:
        features[f"eval_like_usage:{kw}"] = lower.count(kw)

    for cmd in SHELL_COMMANDS:
        pattern = r'(?i)\b' + re.escape(cmd) + r'\b'
        matches = re.findall(pattern, text)
        features[f"shell_command:{cmd}"] = float(len(matches))

    return features, ent, lines_count


def merge_feature_dicts(acc: Dict[str, float], part: Dict[str, float]) -> None:
    """Cộng dồn các giá trị feature."""
    for k, v in part.items():
        acc[k] = acc.get(k, 0.0) + float(v)


# ==================== Phân tích package.json ====================

def analyze_package_json_text(text: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    try:
        data = json.loads(text)
    except Exception:
        return features

    scripts = data.get("scripts", {}) or {}
    if isinstance(scripts, dict):
        features["num_scripts"] = float(len(scripts))
        features["has_scripts"] = 1.0 if scripts else 0.0
        has_install_script = any(name in scripts for name in INSTALL_SCRIPT_NAMES)
        features["has_install_script"] = 1.0 if has_install_script else 0.0

    deps = data.get("dependencies", {}) or {}
    dev_deps = data.get("devDependencies", {}) or {}
    opt_deps = data.get("optionalDependencies", {}) or {}
    peer_deps = data.get("peerDependencies", {}) or {}

    if isinstance(deps, dict):
        features["num_dependencies"] = float(len(deps))
    if isinstance(dev_deps, dict):
        features["num_dev_dependencies"] = float(len(dev_deps))
    if isinstance(opt_deps, dict):
        features["num_optional_dependencies"] = float(len(opt_deps))
    if isinstance(peer_deps, dict):
        features["num_peer_dependencies"] = float(len(peer_deps))

    total_deps = 0
    for dct in [deps, dev_deps, opt_deps, peer_deps]:
        if isinstance(dct, dict):
            total_deps += len(dct)
    features["num_total_dependencies"] = float(total_deps)

    # repository
    repo = data.get("repository")
    has_repo = False
    if isinstance(repo, str) and repo.strip():
        has_repo = True
    elif isinstance(repo, dict) and repo.get("url"):
        has_repo = True
    features["has_repository"] = 1.0 if has_repo else 0.0

    # bin
    bin_field = data.get("bin")
    has_bin = False
    if isinstance(bin_field, str) and bin_field.strip():
        has_bin = True
    elif isinstance(bin_field, dict) and bin_field:
        has_bin = True
    features["has_bin"] = 1.0 if has_bin else 0.0

    # main
    main_field = data.get("main")
    features["has_main"] = 1.0 if isinstance(main_field, str) and main_field.strip() else 0.0

    # length name / version (optional)
    name = data.get("name")
    version = data.get("version")
    if isinstance(name, str):
        features["package_name_length"] = float(len(name))
    if isinstance(version, str):
        features["package_version_length"] = float(len(version))

    return features


# ==================== Trích feature từ package root (FS) ====================

def extract_features_from_fs_root(pkg_root: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # package.json
    pkg_json_path = pkg_root / "package.json"
    if pkg_json_path.is_file():
        text = pkg_json_path.read_text(encoding="utf-8", errors="ignore")
        merge_feature_dicts(features, analyze_package_json_text(text))

    num_js_files = 0
    num_ts_files = 0
    num_json_files = 0
    total_js_ts_bytes = 0
    total_js_ts_lines = 0
    entropies: List[float] = []

    for dirpath, dirnames, filenames in os.walk(pkg_root):
        if "node_modules" in dirnames:
            dirnames.remove("node_modules")

        for fname in filenames:
            fpath = Path(dirpath) / fname
            suffix = fpath.suffix.lower()

            if suffix in {".js", ".mjs", ".cjs"}:
                num_js_files += 1
            elif suffix in {".ts", ".tsx"}:
                num_ts_files += 1
            elif suffix == ".json":
                num_json_files += 1

            if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
                try:
                    data = fpath.read_bytes()
                except Exception:
                    continue

                total_js_ts_bytes += len(data)
                text = data.decode("utf-8", errors="ignore")
                file_features, ent, lines_count = analyze_js_ts_text(text)
                merge_feature_dicts(features, file_features)
                entropies.append(ent)
                total_js_ts_lines += lines_count

    features["num_js_files"] = float(num_js_files)
    features["num_ts_files"] = float(num_ts_files)
    features["num_json_files"] = float(num_json_files)
    features["num_js_ts_files"] = float(num_js_files + num_ts_files)
    features["total_js_ts_bytes"] = float(total_js_ts_bytes)
    features["total_js_ts_lines"] = float(total_js_ts_lines)

    if entropies:
        avg_ent = sum(entropies) / len(entropies)
        var = sum((e - avg_ent) ** 2 for e in entropies) / len(entropies)
        std_ent = math.sqrt(var)
    else:
        avg_ent = 0.0
        std_ent = 0.0

    features["entropy average"] = float(avg_ent)
    features["entropy standard deviation"] = float(std_ent)

    return features


# ==================== Trích feature từ package trong .zip ====================

def extract_features_from_zip(zip_path: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}
    ZIP_PWD = b"infected"
    with ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # tìm các file package.json (trừ node_modules)
        pkg_json_candidates = [
            n for n in names
            if n.lower().endswith("package.json")
            and "node_modules/" not in n.lower()
        ]

        if not pkg_json_candidates:
            # print(f"  [WARN] Không tìm thấy package.json trong zip {zip_path.name}")
            return {}

        # chọn path dài nhất (sâu nhất)
        pkg_json_name = max(pkg_json_candidates, key=lambda x: x.count("/"))
        pkg_root_prefix = os.path.dirname(pkg_json_name).rstrip("/")

        # đọc package.json
        try:
            pkg_json_bytes = zf.read(pkg_json_name)
            pkg_json_text = pkg_json_bytes.decode("utf-8", errors="ignore")
            merge_feature_dicts(features, analyze_package_json_text(pkg_json_text))
        except Exception:
            pass

        num_js_files = 0
        num_ts_files = 0
        num_json_files = 0
        total_js_ts_bytes = 0
        total_js_ts_lines = 0
        entropies: List[float] = []

        prefix = pkg_root_prefix + "/"

        for member in names:
            if not member.startswith(prefix):
                continue
            if member.endswith("/"):
                continue

            _, ext = os.path.splitext(member.lower())

            try:
                data = zf.read(member, pwd=ZIP_PWD)
            except Exception:
                continue

            if ext in {".js", ".mjs", ".cjs"}:
                num_js_files += 1
            elif ext in {".ts", ".tsx"}:
                num_ts_files += 1
            elif ext == ".json":
                num_json_files += 1

            if ext in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
                total_js_ts_bytes += len(data)
                text = data.decode("utf-8", errors="ignore")
                file_features, ent, lines_count = analyze_js_ts_text(text)
                merge_feature_dicts(features, file_features)
                entropies.append(ent)
                total_js_ts_lines += lines_count

        features["num_js_files"] = float(num_js_files)
        features["num_ts_files"] = float(num_ts_files)
        features["num_json_files"] = float(num_json_files)
        features["num_js_ts_files"] = float(num_js_files + num_ts_files)
        features["total_js_ts_bytes"] = float(total_js_ts_bytes)
        features["total_js_ts_lines"] = float(total_js_ts_lines)

        if entropies:
            avg_ent = sum(entropies) / len(entropies)
            var = sum((e - avg_ent) ** 2 for e in entropies) / len(entropies)
            std_ent = math.sqrt(var)
        else:
            avg_ent = 0.0
            std_ent = 0.0

        features["entropy average"] = float(avg_ent)
        features["entropy standard deviation"] = float(std_ent)

    return features


# ==================== Trích feature cho 1 version dir ====================

def extract_features_for_version_dir(version_dir: Path) -> Dict[str, float]:
    """
    Tìm package root theo 2 cách:
    1) FS
    2) Zip
    """
    best_root = None
    best_depth = -1

    for root, dirnames, filenames in os.walk(version_dir):
        if "node_modules" in dirnames:
            dirnames.remove("node_modules")
        if "package.json" in filenames:
            root_path = Path(root)
            if "node_modules" in root_path.parts:
                continue
            depth = len(root_path.parts)
            if depth > best_depth:
                best_depth = depth
                best_root = root_path

    if best_root is not None:
        return extract_features_from_fs_root(best_root)

    zip_files = sorted([p for p in version_dir.iterdir()
                        if p.is_file() and p.suffix.lower() == ".zip"])

    if not zip_files:
        return {}

    zip_path = zip_files[0]
    return extract_features_from_zip(zip_path)


# ==================== Ghi CSV ====================

def write_features_csv(version_dir: Path, features: Dict[str, float], output_name: str) -> None:
    out_path = version_dir / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for name in sorted(features.keys()):
            w.writerow([name, features[name]])


# ==================== Worker Function (Chạy song song) ====================

def process_single_version(task_data):
    """
    Hàm này sẽ được chạy bởi worker process.
    task_data bao gồm: (version_dir, output_name)
    """
    version_dir, output_name = task_data
    try:
        # Lấy tên package và version để in log cho đẹp
        pkg_name = version_dir.parent.name
        ver_name = version_dir.name
        
        features = extract_features_for_version_dir(version_dir)
        if not features:
            return f"[SKIP] {pkg_name}@{ver_name} (No features found)"
        
        write_features_csv(version_dir, features, output_name)
        return f"[OK] {pkg_name}@{ver_name}"
    
    except Exception as e:
        return f"[ERR] {version_dir.name}: {e}"


# ==================== main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Quét dataset npm version-by-version (Multi-core support)."
    )
    parser.add_argument(
        "dataset_root",
        help="Thư mục gốc dataset.",
    )
    parser.add_argument(
        "--output-name",
        default="change-features.csv",
        help="Tên file output (mặc định: change-features.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=f"Số luồng xử lý (mặc định theo CPU máy: {os.cpu_count()})"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset_root không phải thư mục: {dataset_root}")

    # Bước 1: Thu thập toàn bộ danh sách các version cần xử lý
    print("--- Đang quét danh sách thư mục... ---")
    tasks = []
    
    # Duyệt folder Package
    for pkg_dir in sorted(dataset_root.iterdir()):
        if not pkg_dir.is_dir():
            continue
        
        # Duyệt folder Version trong Package
        for version_dir in sorted(pkg_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            
            # Thêm vào danh sách task
            tasks.append((version_dir, args.output_name))

    total_tasks = len(tasks)
    print(f"--- Tìm thấy {total_tasks} versions. Bắt đầu xử lý với {args.workers} cores... ---")

    # Bước 2: Chạy song song bằng ProcessPoolExecutor
    # ProcessPoolExecutor tốt hơn ThreadPoolExecutor cho các việc nặng về CPU (CPU bound)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit các task
        futures = {executor.submit(process_single_version, t): t for t in tasks}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            result_msg = future.result()
            
            # In tiến độ đơn giản
            print(f"[{completed_count}/{total_tasks}] {result_msg}")

    print("\n--- Hoàn tất! ---")

if __name__ == "__main__":
    # Windows cần dòng này để tránh lỗi multiprocessing
    multiprocessing.freeze_support()
    main()
