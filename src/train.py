import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from dataset import PEMalwareDataset, KronodroidDataset


def get_args_parser():
    parser = argparse.ArgumentParser("Training and evaluation script", add_help=False)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type",
        choices=["DT", "XGBoost", "LR", "KNN", "GNB", "RF", "MLP"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
        choices=["bodmas", "ember", "kronodroid"],
    )
    parser.add_argument(
        "--train_start_date",
        default=201908,
        type=int,
        help="year in 4 digits followed by month in 2 digits",
    )
    parser.add_argument(
        "--train_end_date",
        default=201912,
        type=int,
        help="year in 4 digits followed by month in 2 digits",
    )
    parser.add_argument(
        "--test_start_date",
        default=202001,
        type=int,
        help="year in 4 digits followed by month in 2 digits",
    )
    parser.add_argument(
        "--test_end_date",
        default=202004,
        type=int,
        help="year in 4 digits followed by month in 2 digits",
    )
    parser.add_argument("--output_dir", default="exps/", type=str)
    parser.add_argument(
        "--feat_select",
        action="store_true",
        help="whether to apply feature selection or not",
    )
    parser.add_argument(
        "--top_k_feat", default=1000, type=int, help="top k features to be selected"
    )
    parser.add_argument(
        "--debug", action="store_true", help="allow additional prints for debugging"
    )
    return parser


def select_top_features(data, top_k, debug=False):
    selector = SelectKBest(f_classif, k=top_k)
    top_feat = selector.fit_transform(data.features, data.labels)
    data.features = top_feat

    if debug:
        selected_feats = np.where(selector.get_support())[0]
        print("selected feature indices", selected_feats)
    return data


def get_model(model_type):
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "XGBoost":
        model = XGBClassifier(
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            learning_rate=0.1,
            n_estimators=100,
            random_state=0,
        )
    elif model_type == "LR":
        model = LogisticRegression(
            penalty="l2", tol=0.001, C=0.1, max_iter=8000, random_state=0
        )
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "GNB":
        model = GaussianNB(var_smoothing=1e-1)
    elif model_type == "RF":
        model = RandomForestClassifier(n_estimators=10)
    elif model_type == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 256),
            max_iter=50,
            alpha=0.0001,
            solver="sgd",
            verbose=10,
            random_state=0,
            learning_rate_init=0.01,
        )
    else:
        raise NotImplementedError
    return model


def train(model, X_train, y_train):
    model.fit(X_train, y_train)


def eval(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    cls_report = classification_report(y_test, preds, output_dict=True)
    f1 = cls_report["macro avg"]["f1-score"]
    # print(cls_report)
    recall = cls_report["1"]["recall"]
    precision = cls_report["1"]["precision"]
    return {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}


def train_and_eval(model, X_train, y_train, X_test, y_test):
    train(model, X_train, y_train)
    return eval(model, X_test, y_test)


def run(
    model_type,
    dataset,
    train_start_date,
    train_end_date,
    test_start_date,
    test_end_date,
    output_dir="exps/",
):
    print("Loading the dataset...")
    if dataset == 'kronodroid':
        import warnings
        warnings.filterwarnings("ignore")

        data = KronodroidDataset(path=os.path.expanduser(os.getcwd()+'/Kronodroid'))
    else:
        data = PEMalwareDataset.from_name(dataset)

    from collections import Counter
    print(Counter(data.dates))
    if args.feat_select:
        data = select_top_features(data, args.top_k_feat, args.debug)

    train_set = data.filter_by_date(train_start_date, train_end_date)
    test_set = data.filter_by_date(test_start_date, test_end_date)

    counts = Counter(train_set.labels)
    assert (
        len(counts) > 1
    ), "Training data only has 1 class, consider expanding the time period"

    X_train, X_val, y_train, y_val = train_test_split(
        train_set.features, train_set.labels, test_size=0.20, random_state=0
    )

    model = get_model(model_type)

    print("Dataset is loaded.")
    print(
        "Train set size: {0}, Val. set size: {1}, Test set size: {2}".format(
            X_train.shape[0], X_val.shape[0], test_set.labels.shape[0]
        )
    )
    print("Training the model...")

    train(model, X_train, y_train)

    print("Training completed.")
    print("Evaluating...")

    train_scores = eval(model, X_train, y_train)
    val_scores = eval(model, X_val, y_val)
    test_scores = eval(model, test_set.features, test_set.labels)

    # Accuracy
    print(
        "Accuracy on training data (same time period):",
        round(train_scores["accuracy"], 3),
    )
    print(
        "Accuracy on validation data (same time period):",
        round(val_scores["accuracy"], 3),
    )
    print(
        "Accuracy on test data (different time period):",
        round(test_scores["accuracy"], 3),
    )

    # F1
    print("F1 on training data (same time period):", round(train_scores["f1"], 3))
    print("F1 on validation data (same time period):", round(val_scores["f1"], 3))
    print("F1 on test data (different time period):", round(test_scores["f1"], 3))

    # Recall
    print(
        "Recall on training data (same time period):", round(train_scores["recall"], 3)
    )
    print(
        "Recall on validation data (same time period):", round(val_scores["recall"], 3)
    )
    print(
        "Recall on test data (different time period):", round(test_scores["recall"], 3)
    )

    # Precision
    print(
        "Precision on training data (same time period):",
        round(train_scores["precision"], 3),
    )
    print(
        "Precision on validation data (same time period):",
        round(val_scores["precision"], 3),
    )
    print(
        "Precision on test data (different time period):",
        round(test_scores["precision"], 3),
    )


def main(args):
    run(
        args.model_type,
        args.dataset,
        args.train_start_date,
        args.train_end_date,
        args.test_start_date,
        args.test_end_date,
        args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
