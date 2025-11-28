#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
from pathlib import Path


def load_model(model_path: Path):
    with model_path.open("rb") as f:
        data = pickle.load(f)
    classifier = data["classifier"]
    feature_names = data["feature_names"]
    booleanize = data.get("booleanize", False)
    positive = data.get("positive", False)
    return classifier, feature_names, booleanize, positive


def load_feature_vector(change_features_path: Path, feature_names, booleanize: bool, positive: bool):
    """
    Đọc change-features.csv và dựng feature vector theo đúng thứ tự feature_names.
    Nếu thiếu feature nào thì để 0.
    Nếu file có feature lạ mà model không dùng thì bỏ qua.
    """
    feature_dict = {}

    with change_features_path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            feature, value = row
            try:
                v = float(value)
            except ValueError:
                continue

            # Áp dụng cùng quy tắc như lúc train
            if positive and v < 0:
                v = 0.0
            if booleanize:
                v = 1.0 if v > 0 else 0.0

            feature_dict[feature] = v

    # map feature -> index để truy cập nhanh
    idx_map = {name: i for i, name in enumerate(feature_names)}

    vec = [0.0] * len(feature_names)
    for feat, v in feature_dict.items():
        idx = idx_map.get(feat)
        if idx is not None:
            vec[idx] = v
        # nếu feature không có trong model thì bỏ qua

    return vec, feature_dict


def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán benign/malicious cho một package version (thư mục chứa change-features.csv)."
    )
    parser.add_argument(
        "package_dir",
        help="Thư mục version, vd: /home/thang/.../malicious_intent/react-server-native/0.0.6",
    )
    parser.add_argument(
        "-m", "--model",
        default="model.pkl",
        help="Đường dẫn tới model.pkl (mặc định: model.pkl trong thư mục hiện tại).",
    )

    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()
    model_path = Path(args.model).resolve()

    change_features_path = package_dir / "change-features.csv"
    if not change_features_path.is_file():
        raise SystemExit(f"Không tìm thấy {change_features_path}")

    if not model_path.is_file():
        raise SystemExit(f"Không tìm thấy model: {model_path}")

    print(f"[INFO] Dùng model: {model_path}")
    print(f"[INFO] Dự đoán cho: {package_dir}")

    clf, feature_names, booleanize, positive = load_model(model_path)
    vec, feature_dict = load_feature_vector(
        change_features_path, feature_names, booleanize, positive
    )

    # dự đoán label
    pred = clf.predict([vec])[0]

    print(f"\nKẾT QUẢ:")
    print(f"  Label dự đoán: {pred}")

    # nếu classifier hỗ trợ predict_proba thì in luôn xác suất
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([vec])[0]
        classes = clf.classes_
        print("  Xác suất:")
        for cls, p in zip(classes, probs):
            print(f"    {cls:9}: {p:.4f}")

    # in thử vài feature quan trọng (top theo giá trị tuyệt đối)
    print("\nMột vài feature đầu vào (top theo |value|):")
    fv_items = sorted(feature_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in fv_items[:15]:
        print(f"  {feat:40} {val}")


if __name__ == "__main__":
    main()
