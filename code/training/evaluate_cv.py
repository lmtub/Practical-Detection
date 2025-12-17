#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def load_malicious_set(malicious_csv: Path):
    """Đọc danh sách (package, version) malicious từ CSV."""
    mal_set = set()
    with malicious_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            pkg, ver = row[0].strip(), row[1].strip()
            mal_set.add((pkg, ver))
    return mal_set


def collect_samples(malicious_set, training_dirs):
    """
    Quét tất cả change-features.csv trong các thư mục training_dirs,
    build:
      - feature_names: list tên feature
      - X: numpy array [n_samples, n_features]
      - y: nhãn "malicious"/"benign"
      - packages: tên package (làm group cho GroupKFold)
    """
    samples = []
    labels = []
    packages = []
    feature_names = []

    for root_dir in training_dirs:
        root_dir = Path(root_dir).resolve()
        if not root_dir.exists():
            print(f"[WARN] Thư mục không tồn tại: {root_dir}")
            continue

        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirpath = Path(dirpath)
            if "change-features.csv" not in filenames:
                continue

            # Giả sử cấu trúc: .../<package>/<version>/change-features.csv
            try:
                version = dirpath.name
                package = dirpath.parent.name
            except Exception:
                # fallback: dùng tên thư mục hiện tại làm package, version = "unknown"
                package = dirpath.name
                version = "unknown"

            feat_path = dirpath / "change-features.csv"
            feat_dict = {}
            with feat_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    feat, val = row
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                    feat_dict[feat] = v
                    if feat not in feature_names:
                        feature_names.append(feat)

            label = "malicious" if (package, version) in malicious_set else "benign"

            samples.append(feat_dict)
            labels.append(label)
            packages.append(package)

    n_samples = len(samples)
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features), dtype=float)
    y = np.array(labels)
    packages = np.array(packages)

    feat_index = {f: i for i, f in enumerate(feature_names)}

    for i, fdict in enumerate(samples):
        for feat, v in fdict.items():
            j = feat_index.get(feat)
            if j is not None:
                X[i, j] = v

    return feature_names, X, y, packages


def undersample_balance(X_train, y_train, seed=42):
    """
    Cân bằng số lượng benign/malicious trong tập train
    bằng cách undersample lớp nhiều hơn xuống bằng lớp ít hơn.
    """
    mal_idx = np.where(y_train == "malicious")[0]
    ben_idx = np.where(y_train == "benign")[0]

    if len(mal_idx) == 0 or len(ben_idx) == 0:
        # không cân bằng được, trả về nguyên trạng
        print("    [BALANCE] Không thể cân bằng: thiếu 1 trong 2 lớp.")
        return X_train, y_train

    target = min(len(mal_idx), len(ben_idx))
    rng = np.random.default_rng(seed)
    sel_mal = rng.choice(mal_idx, size=target, replace=False)
    sel_ben = rng.choice(ben_idx, size=target, replace=False)

    keep = np.concatenate([sel_mal, sel_ben])
    X_bal = X_train[keep]
    y_bal = y_train[keep]

    print(
        f"    [BALANCE] Train fold: {len(ben_idx)} benign, {len(mal_idx)} malicious "
        f"-> {target} + {target} = {2*target} samples"
    )

    return X_bal, y_bal


def main():
    parser = argparse.ArgumentParser(
        description=(
            "10-fold stratified Group K-Fold cho Decision Tree (group theo package, "
            "có tùy chọn cân bằng tập train và regularization)."
        )
    )
    parser.add_argument(
        "malicious_csv",
        help="CSV chứa (package,version) malicious, vd: metadata/malicious.csv",
    )
    parser.add_argument(
        "training_dirs",
        nargs="+",
        help="Một hoặc nhiều thư mục chứa change-features.csv (malicious_intent, benign-root, ...)",
    )
    parser.add_argument(
        "-k",
        "--folds",
        type=int,
        default=10,
        help="Số fold cho cross-validation (mặc định 10).",
    )
    parser.add_argument(
        "--balanced-train",
        choices=["true", "false"],
        default="true",
        help=(
            "Có cân bằng lại tập train trong từng fold không (undersample). "
            "Mặc định: true."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Giới hạn độ sâu Decision Tree (None = không giới hạn).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Số mẫu tối thiểu ở mỗi lá. Mặc định: 1.",
    )

    args = parser.parse_args()
    balanced_train = args.balanced_train == "true"

    malicious_csv = Path(args.malicious_csv).resolve()
    if not malicious_csv.is_file():
        raise SystemExit(f"Không tìm thấy malicious.csv: {malicious_csv}")

    print(f"[INFO] Đọc danh sách malicious từ: {malicious_csv}")
    malicious_set = load_malicious_set(malicious_csv)

    print("[INFO] Thu thập feature từ các thư mục:")
    for d in args.training_dirs:
        print(f"       - {d}")

    feature_names, X, y, packages = collect_samples(malicious_set, args.training_dirs)

    # Loại bỏ các feature leak về size / dependencies / meta package
    LEAKY = {
        "has_main",
    	"has_repository",
    	"has_scripts",
    	"num_dependencies",
    	"num_dev_dependencies",
    	"num_js_files",
    	"num_js_ts_files",
    	"num_json_files",
    	"num_scripts",
    	"num_total_dependencies",
    	"num_ts_files",
    	"package_name_length",
    	"package_version_length",
    	"total_js_ts_bytes",
    	"total_js_ts_lines",
        "entropy standard deviation",
        "entropy average",
    }

    use_idx = [i for i, f in enumerate(feature_names) if f not in LEAKY]
    X = X[:, use_idx]
    feature_names = [feature_names[i] for i in use_idx]
    print(f"[INFO] Dùng {len(feature_names)} feature (đã bỏ size/dep/meta leak)")

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_mal = int((y == "malicious").sum())
    n_ben = int((y == "benign").sum())

    print("\n[DATASET]")
    print(f"  Số sample: {n_samples}")
    print(f"  Số feature: {n_features}")
    print(f"  Malicious: {n_mal}")
    print(f"  Benign   : {n_ben}")
    print(
        f"[PARAMS] balanced_train={balanced_train}, "
        f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}"
    )

    skf = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=42,
    )

    # lưu metric test + train để tính trung bình
    test_precs = []
    test_recs = []
    train_precs = []
    train_recs = []

    all_importances = []  # lưu importance của từng fold

    fold_id = 1
    for train_idx, test_idx in skf.split(X, y, groups=packages):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if balanced_train:
            X_train_use, y_train_use = undersample_balance(
                X_train, y_train, seed=42 + fold_id
            )
        else:
            X_train_use, y_train_use = X_train, y_train

        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
#            class_weight='balanced',
        )
        clf.fit(X_train_use, y_train_use)

        # === Metric trên TRAIN (để soi overfit) ===
        y_pred_train = clf.predict(X_train_use)
        prec_tr, rec_tr, _, _ = precision_recall_fscore_support(
            y_train_use,
            y_pred_train,
            labels=["malicious", "benign"],
            zero_division=0,
        )
        prec_tr_mal, rec_tr_mal = prec_tr[0], rec_tr[0]
        train_precs.append(prec_tr_mal)
        train_recs.append(rec_tr_mal)

        # Lưu importance cho fold này
        importances = clf.feature_importances_
        all_importances.append(importances)

        # === Metric trên TEST (fold hold-out) ===
        y_pred = clf.predict(X_test)
        # === [BẮT ĐẦU CODE MỚI] ===
        # Lấy tên các package trong tập test hiện tại
        packages_test = packages[test_idx]

        # Tìm vị trí sai: Thực tế là "malicious" NHƯNG Máy đoán "benign"
        fn_indices = np.where((y_test == "malicious") & (y_pred == "benign"))[0]

        if len(fn_indices) > 0:
            print(f"  [CẢNH BÁO] Bỏ sót {len(fn_indices)} mẫu mã độc trong Fold {fold_id}:")
            missed_pkgs = packages_test[fn_indices]
            
            # 1. In mẫu 3 cái ra màn hình để xem nhanh
            for p in missed_pkgs[:3]:
                print(f"    - {p}")
            
            # 2. Ghi TẤT CẢ vào file missed_malware.txt để phân tích sau
            with open("missed_malware.txt", "a", encoding="utf-8") as f:
                f.write(f"--- FOLD {fold_id} ---\n")
                for p in missed_pkgs:
                    f.write(f"{p}\n")
        # === [KẾT THÚC CODE MỚI] ===
        prec_te, rec_te, _, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=["malicious", "benign"], zero_division=0
        )
        prec_mal, rec_mal = prec_te[0], rec_te[0]
        test_precs.append(prec_mal)
        test_recs.append(rec_mal)

        cm = confusion_matrix(y_test, y_pred, labels=["malicious", "benign"])
        tn = cm[1, 1]  # benign->benign
        fp = cm[1, 0]  # benign->malicious
        fn = cm[0, 1]  # malicious->benign
        tp = cm[0, 0]  # malicious->malicious

        print(f"\n[FOLD {fold_id}]")
        print(f"  TP (mal->mal): {tp}")
        print(f"  FN (mal->ben): {fn}")
        print(f"  FP (ben->mal): {fp}")
        print(f"  TN (ben->ben): {tn}")
        print(f"  Train Precision(malicious): {prec_tr_mal:.4f}")
        print(f"  Train Recall(malicious)   : {rec_tr_mal:.4f}")
        print(f"  Test  Precision(malicious): {prec_mal:.4f}")
        print(f"  Test  Recall(malicious)   : {rec_mal:.4f}")

        # In top 15 feature quan trọng của fold này
        indices_imp = np.argsort(importances)[::-1]
        print(f"  [TOP 15 FEATURE IMPORTANCE - FOLD {fold_id}]")
        for rank in range(15):
            idx = indices_imp[rank]
            if importances[idx] <= 0:
                break
            print(
                f"    {rank+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}"
            )

        fold_id += 1

    # Importance trung bình trên tất cả các fold
    if all_importances:
        mean_importance = np.mean(np.vstack(all_importances), axis=0)
        indices = np.argsort(mean_importance)[::-1]

        print("\n[TOP 20 FEATURE IMPORTANCE TRUNG BÌNH]")
        for rank in range(20):
            idx = indices[rank]
            if mean_importance[idx] <= 0:
                break
            print(
                f"  {rank+1:2d}. {feature_names[idx]:40s} {mean_importance[idx]:.4f}"
            )

    # === Trung bình train vs test để nhìn gap (overfit) ===
    print("\n[AVG OVER FOLDS]")
    print(f"  Train Precision(malicious) trung bình: {np.mean(train_precs):.4f}")
    print(f"  Train Recall(malicious)    trung bình: {np.mean(train_recs):.4f}")
    print(f"  Test  Precision(malicious) trung bình: {np.mean(test_precs):.4f}")
    print(f"  Test  Recall(malicious)    trung bình: {np.mean(test_recs):.4f}")
    print(
        f"  GAP Recall (Train - Test): "
        f"{(np.mean(train_recs) - np.mean(test_recs)):.4f}"
    )


if __name__ == "__main__":
    main()
