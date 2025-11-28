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
      - X: matrix (n_samples x n_features)
      - y: nhãn "benign"/"malicious"
      - packages: tên package (dùng cho GroupKFold)
    """
    feature_names = []
    samples = []   # list feature_dict
    labels = []
    packages = []

    for training_set_dir in training_dirs:
        training_set_dir = Path(training_set_dir).resolve()
        for root, _, files in os.walk(training_set_dir):
            if "change-features.csv" in files:
                root_path = Path(root)
                package = os.path.relpath(root_path.parent, training_set_dir)
                version = root_path.name

                feat_path = root_path / "change-features.csv"
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

    print(f"    [BALANCE] Train fold: {len(ben_idx)} benign, {len(mal_idx)} malicious "
          f"-> {target} + {target} = {2*target} samples")

    return X_bal, y_bal


def main():
    parser = argparse.ArgumentParser(
        description="10-fold stratified Group K-Fold cho Decision Tree (group theo package, "
                    "có tùy chọn cân bằng tập train và regularization)."
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
        "-k", "--folds",
        type=int,
        default=10,
        help="Số fold cho cross-validation (mặc định 10).",
    )
    parser.add_argument(
        "--balanced-train",
        choices=["true", "false"],
        default="true",
        help="Có cân bằng lại tập train trong từng fold không (undersample). Mặc định: true.",
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
    balanced_train = (args.balanced_train == "true")

    malicious_csv = Path(args.malicious_csv).resolve()
    if not malicious_csv.is_file():
        raise SystemExit(f"Không tìm thấy malicious.csv: {malicious_csv}")

    print(f"[INFO] Đọc danh sách malicious từ: {malicious_csv}")
    malicious_set = load_malicious_set(malicious_csv)

    print("[INFO] Thu thập feature từ các thư mục:")
    for d in args.training_dirs:
        print(f"       - {d}")

    feature_names, X, y, packages = collect_samples(malicious_set, args.training_dirs)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_mal = int((y == "malicious").sum())
    n_ben = int((y == "benign").sum())

    print(f"\n[DATASET]")
    print(f"  Số sample: {n_samples}")
    print(f"  Số feature: {n_features}")
    print(f"  Malicious: {n_mal}")
    print(f"  Benign   : {n_ben}")
    print(f"[PARAMS] balanced_train={balanced_train}, "
          f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}")

    skf = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=42
    )

    precs = []
    recs = []

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
        )
        clf.fit(X_train_use, y_train_use)

        y_pred = clf.predict(X_test)

        prec, rec, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=["malicious", "benign"], zero_division=0
        )
        prec_mal, rec_mal = prec[0], rec[0]
        precs.append(prec_mal)
        recs.append(rec_mal)

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
        print(f"  Precision(malicious): {prec_mal:.4f}")
        print(f"  Recall(malicious)   : {rec_mal:.4f}")

        fold_id += 1

    print("\n[AVG OVER FOLDS]")
    print(f"  Precision(malicious) trung bình: {np.mean(precs):.4f}")
    print(f"  Recall(malicious)    trung bình: {np.mean(recs):.4f}")


if __name__ == "__main__":
    main()
