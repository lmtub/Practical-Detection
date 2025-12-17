#! /usr/bin/env python3

import argparse
import csv
import os
import pickle
import random

from datetime import timedelta
from graphviz import Source
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer
from util import parse_date, version_date

# features with continuous values
CONTINUOUS_FEATURES = ["entropy average", "entropy standard deviation", "time"]

# các feature bị xem là "leaky" (size/dep/meta) – KHÔNG dùng khi train
LEAKY_FEATURES = [
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
]

def train_classifier(
    classifier,
    malicious_path,
    training_sets,
    output,
    booleanize=False,
    hashing=False,
    exclude_features=None,
    nu=0.001,
    positive=False,
    render=False,
    randomize=False,
    view=False,
    leave_out=None,
    until=None,
    performance=None,
    max_depth=6,
    min_samples_leaf=10,
):

    # giữ lại kiểu classifier gốc (string), tránh bị ghi đè bởi model sklearn
    classifier_type = classifier

    if exclude_features is None:
        exclude_features = []

    # Luôn loại bỏ các feature bị leak (size/dep/meta)
    for f in LEAKY_FEATURES:
        if f not in exclude_features:
            exclude_features.append(f)

    if leave_out is None:
        leave_out = []

    # Naive Bayes implicitly booleanizes the feature vectors
    if classifier_type == "naive-bayes":
        booleanize = True

    # exclude continuous features when booleanizing
    if booleanize:
        exclude_features.extend(CONTINUOUS_FEATURES)

    # names of features
    feature_names = []
    # an array of arrays, each of which is a feature vector
    training_set = []
    # label each row of the feature matrix as either "benign" or "malicious"
    labels = []

    # load known malicious (package,version) pairs or their hashes
    malicious = set()
    with open(malicious_path, "r") as mal:
        reynolds = csv.reader(mal)
        for row in reynolds:
            if hashing:
                hash_res = row[0]
                malicious.add(hash_res)
            else:
                package, version = row
                malicious.add((package, version))

    versions = {}

    for training_set_dir in training_sets:
        for root, _, files in os.walk(training_set_dir):
            for f in files:
                if f == "change-features.csv" and f"{root}" not in leave_out:
                    package = os.path.relpath(
                        os.path.dirname(root), training_set_dir
                    )
                    version = os.path.basename(root)

                    # Không phải dataset nào cũng có versions.csv (Datadog, benign tự tải)
                    # => nếu thiếu thì coi như không có date, không filter theo until.
                    try:
                        date = version_date(versions, root)
                    except FileNotFoundError:
                        date = None

                    print(f"{package}@{version}: {date}")

                    if until is not None and date is not None and date >= until:
                        print(
                            f"Skipping {package}@{version}. Date {date} is outside the boundaries."
                        )
                        continue

                    print(f"Processing {package}@{version}")

                    # load features for this package
                    with open(os.path.join(root, f), "r") as feature_file:
                        # first, read features into a dictionary
                        feature_dict = {}
                        for row in csv.reader(feature_file):
                            feature, value = row
                            value = float(value)

                            if positive and value < 0:
                                value = 0
                            if booleanize:
                                value = 1 if value > 0 else 0
                            if feature not in exclude_features:
                                feature_dict[feature] = value

                        # assign indices to any features we have not seen before
                        for feature in feature_dict.keys():
                            if feature not in feature_names:
                                feature_names.append(feature)

                        # convert the feature dictionary into a feature vector
                        feature_vec = []
                        for feature, value in feature_dict.items():
                            idx = feature_names.index(feature)
                            if idx >= len(feature_vec):
                                feature_vec.extend(
                                    [0] * (idx - len(feature_vec) + 1)
                                )
                            feature_vec[idx] = value

                        # add the feature vector to the training set
                        training_set.append(feature_vec)

                        # add the label to the labels list
                        label = "benign"
                        if hashing:
                            hash_file = os.path.join(root, "hash.csv")
                            if os.path.isfile(hash_file) and os.path.getsize(
                                hash_file
                            ) > 0:
                                with open(hash_file, "r") as rfi:
                                    hash_res = csv.reader(rfi).__next__()[0]
                                if hash_res in malicious:
                                    label = "malicious"
                        else:
                            if (package, version) in malicious:
                                label = "malicious"
                        labels.append(label)

    # normalize length of feature vectors by extending with zeros
    num_features = len(feature_names)
    for i in range(len(training_set)):
        length = len(training_set[i])
        if length < num_features:
            training_set[i].extend([0] * (num_features - length))

    # nếu bật randomize, cân bằng kích thước 2 class benign/malicious
    if randomize:
        # gom index theo label
        indices_by_label = {}
        for i, y in enumerate(labels):
            indices_by_label.setdefault(y, []).append(i)

        # chỉ xử lý khi có đủ cả 2 lớp
        if "benign" in indices_by_label and "malicious" in indices_by_label:
            benign_indices = indices_by_label["benign"]
            malicious_indices = indices_by_label["malicious"]

            # chọn kích thước nhỏ hơn trong 2 class
            target_size = min(len(benign_indices), len(malicious_indices))

            if target_size == 0:
                print(
                    "[WARN] Không thể cân bằng: một trong hai lớp có 0 mẫu."
                )
            else:
                # random chọn target_size mẫu từ mỗi class
                benign_selected = set(
                    random.sample(benign_indices, target_size)
                )
                malicious_selected = set(
                    random.sample(malicious_indices, target_size)
                )

                keep = benign_selected | malicious_selected

                training_set = [
                    row
                    for i, row in enumerate(training_set)
                    if i in keep
                ]
                labels = [
                    lab
                    for i, lab in enumerate(labels)
                    if i in keep
                ]

                print(
                    f"[INFO] Balanced dataset: {target_size} benign + {target_size} malicious = {2*target_size} samples."
                )
        else:
            print(
                "[WARN] Không đủ cả 2 lớp benign/malicious để cân bằng."
            )

    start = timer()

    # train the classifier
    if classifier_type == "decision-tree":
        # Decision Tree Amalfi + hyper-parameters đã tune
        clf = tree.DecisionTreeClassifier(
            criterion="entropy",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        clf.fit(training_set, labels)

    elif classifier_type == "random-forest":
        clf = RandomForestClassifier(
            criterion="entropy"
        )
        clf.fit(training_set, labels)

    elif classifier_type == "naive-bayes":
        clf = naive_bayes.BernoulliNB()
        clf.fit(training_set, labels)

    else:
        # svm / one-class svm
        clf = svm.OneClassSVM(
            gamma="scale", nu=nu, kernel="linear"
        )
        clf.fit(
            [
                datum
                for i, datum in enumerate(training_set)
                if labels[i] == "benign"
            ]
        )

    end = timer()
    diff = timedelta(seconds=end - start)

    if performance is not None:
        with open(performance, "a+") as wfi:
            writer = csv.writer(wfi)
            writer.writerow([diff])

    # render the tree if requested; only applicable for decision trees
    if classifier_type == "decision-tree" and render:
        file, ext = os.path.splitext(render)
        if ext != ".png":
            print("Rendering tree to PNG requires a file name ending in .png")
            exit(1)
        outfile = Source(
            tree.export_graphviz(
                clf, out_file=None, feature_names=feature_names
            ),
            format="png",
        )
        outfile.render(file, view=view, cleanup=True)

    # store the classifier and metadata
    with open(output, "wb") as f:
        pickle.dump(
            {
                "feature_names": feature_names,
                "booleanize": booleanize,
                "positive": positive,
                "classifier": clf,
            },
            f,
        )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
        description="Train a classifier"
    )
    argparse.add_argument(
        "classifier",
        help="Type of classifier to be trained.",
        choices=["decision-tree", "random-forest", "naive-bayes", "svm"],
    )
    argparse.add_argument(
        "malicious",
        help="CSV file listing known malicious package versions.",
    )
    argparse.add_argument(
        "training_sets",
        help="Directories with features for package versions to train on.",
        nargs="*",
    )
    argparse.add_argument(
        "-b",
        "--booleanize",
        help="Whether to booleanize feature vectors.",
        choices=["true", "false"],
        default="false",
    )
    argparse.add_argument(
        "--hashing",
        help=(
            "Whether hashes are required to label malicious packages. "
            "Default is pairs of <package,version>"
        ),
        choices=["true", "false"],
        default="false",
    )
    argparse.add_argument(
        "-x",
        "--exclude-features",
        help="List of features to exclude.",
        required=False,
        nargs="*",
        default=[],
    )
    argparse.add_argument(
        "-n",
        "--nu",
        help="nu value for svm.",
        required=False,
        type=float,
        default=0.001,
    )
    argparse.add_argument(
        "-o",
        "--output",
        help="Output file to store the pickled classifier in.",
        required=True,
    )
    argparse.add_argument(
        "-p",
        "--positive",
        help="Whether to keep only positive values in features",
        choices=["true", "false"],
        default="false",
    )
    argparse.add_argument(
        "-r",
        "--render",
        help=(
            "PNG file to render the decision tree to. "
            "Ignored for other types of classifiers."
        ),
        required=False,
    )
    argparse.add_argument(
        "--randomize",
        help="Balance datasets.",
        choices=["true", "false"],
        default="false",
    )
    argparse.add_argument(
        "-v",
        "--view",
        help=(
            "View the decision tree graphically. "
            "Ignored unless --render is specified."
        ),
        action="store_true",
    )
    argparse.add_argument(
        "-l",
        "--leave_out",
        help="Training files to leave out",
        required=False,
        nargs="*",
        default=[],
    )
    argparse.add_argument(
        "-u",
        "--until",
        help=(
            "Specify the date up to which samples should be considered "
            "for training."
        ),
        required=False,
        default="2100-01-01T00:00:00.000Z",
    )
    argparse.add_argument(
        "--max-depth",
        help=(
            "Max depth for decision tree / random forest "
            "(None = unlimited)."
        ),
        required=False,
        type=int,
        default=6,  # giống cấu hình đã tune trong evaluate_multi.py
    )
    argparse.add_argument(
        "--min-samples-leaf",
        help="Minimum number of samples per leaf for tree-based models.",
        required=False,
        type=int,
        default=10,  # giống cấu hình đã tune trong evaluate_multi.py
    )

    args = argparse.parse_args()
    booleanize = True if args.booleanize == "true" else False
    hashing = True if args.hashing == "true" else False
    positive = True if args.positive == "true" else False
    randomize = True if args.randomize == "true" else False
    until = parse_date(args.until)

    train_classifier(
        args.classifier,
        args.malicious,
        args.training_sets,
        args.output,
        booleanize=booleanize,
        hashing=hashing,
        exclude_features=args.exclude_features,
        nu=args.nu,
        positive=positive,
        render=args.render,
        randomize=randomize,
        view=args.view,
        leave_out=args.leave_out,
        until=until,
        performance=None,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
