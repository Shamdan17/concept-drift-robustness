from train import train, eval, get_model
from dataset import PEMalwareDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import argparse
import os
import json

# Only arg is output directory
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="exps/")
args = parser.parse_args()

# Make output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# First experiment, robustness to fixed-period concept drift of different models
params_by_dataset = {
    # "bodmas": {
    #     "start_year": 2019,
    #     "start_month": 8,
    #     "end_year": 2020,
    #     "end_month": 9,
    #     "train_duration": 4,
    #     "test_gap": 1,
    #     "test_duration": 2,
    # },
    "ember": {
        "start_year": 2017,
        "start_month": 11,
        "end_year": 2018,
        "end_month": 12,
        "train_duration": 4,
        "test_gap": 1,
        "test_duration": 2,
    },
}

# models = ["DT", "XGBoost", "LR", "KNN", "RF", "MLP"]
models = ["DT", "XGBoost"]


def validate_date(date):
    if date % 100 > 12:
        return date + 88
    return date


def to_str(date):
    year = date // 100
    month = date % 100
    return f"{year}-{month}"


for dataset in params_by_dataset:
    data = PEMalwareDataset.from_name(dataset)

    last_stamp = (
        params_by_dataset[dataset]["end_year"] * 100
        + params_by_dataset[dataset]["end_month"]
    )

    statistics = {}

    for model_name in models:
        first_stamp = (
            params_by_dataset[dataset]["start_year"] * 100
            + params_by_dataset[dataset]["start_month"]
        )

        statistics[model_name] = []
        while True:
            train_start = first_stamp
            train_end = validate_date(
                train_start + params_by_dataset[dataset]["train_duration"] - 1
            )
            test_start = validate_date(
                train_end + params_by_dataset[dataset]["test_gap"]
            )
            test_end = validate_date(
                test_start + params_by_dataset[dataset]["test_duration"] - 1
            )

            if test_end > last_stamp:
                break
            print(
                f"Running {model_name} on {dataset} from {train_start} to {train_end} and {test_start} to {test_end}"
            )

            train_set = data.filter_by_date(train_start, train_end)
            test_set = data.filter_by_date(test_start, test_end)

            X_train, X_val, y_train, y_val = train_test_split(
                train_set.features, train_set.labels, test_size=0.20, random_state=0
            )

            X_test, y_test = test_set.features, test_set.labels

            model = get_model(model_name)
            train(model, X_train, y_train)

            val_scores = eval(model, X_val, y_val)
            test_scores = eval(model, X_test, y_test)

            statistics[model_name].append(
                {
                    "train_period": (train_start, train_end),
                    "test_period": (test_start, test_end),
                    "val": val_scores,
                    "test": test_scores,
                }
            )

            # Move to next month after training set
            first_stamp = validate_date(first_stamp + 1)

    # Plot results with x-axis as test start and y-axis as f1 score
    # Figure of size 10, 8
    fix, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot params for saving
    params = []

    for model_name in statistics:
        ax.plot(
            [to_str(s["test_period"][0]) for s in statistics[model_name]],
            [s["test"]["f1"] for s in statistics[model_name]],
            label=model_name,
        )
        params.append(
            {
                "model": model_name,
                "label": model_name,
                "x": [to_str(s["test_period"][0]) for s in statistics[model_name]],
                "y": [s["test"]["f1"] for s in statistics[model_name]],
            }
        )

    plt.xlabel("Test Period")
    plt.ylabel("F1 score")
    plt.title(f"Concept drift robustness on {dataset}")
    plt.legend()
    # Save figure
    plt.savefig(os.path.join(args.output_dir, f"experiment-1-{dataset}.png"))
    with open(os.path.join(args.output_dir, f"experiment-1-{dataset}.json"), "w") as f:
        f.write(json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False))
    # plt.show()


# Second experiment, robustness of the best models in first experiment to increasing concept drift
params_by_dataset = {
    "bodmas": {
        "start_year": 2019,
        "start_month": 8,
        "end_year": 2020,
        "end_month": 9,
        "train_duration": 3,
        "test_gap": 1,
        "test_duration": 5,
        "test_stride": 2,
    },
    "ember": {
        "start_year": 2017,
        "start_month": 11,
        "end_year": 2018,
        "end_month": 12,
        "train_duration": 3,
        "test_gap": 1,
        "test_duration": 5,
        "test_stride": 2,
    },
}

models = ["XGBoost", "LR", "RF"]
# models = ["LR", "RF"]

all_dataset_statistics = {}
for dataset in params_by_dataset:
    data = PEMalwareDataset.from_name(dataset)

    last_stamp = (
        params_by_dataset[dataset]["end_year"] * 100
        + params_by_dataset[dataset]["end_month"]
    )

    statistics = {}

    for model_name in models:
        first_stamp = (
            params_by_dataset[dataset]["start_year"] * 100
            + params_by_dataset[dataset]["start_month"]
        )

        statistics[model_name] = {}
        while True:
            train_start = first_stamp
            train_end = validate_date(
                train_start + params_by_dataset[dataset]["train_duration"] - 1
            )
            test_start = validate_date(
                train_end + params_by_dataset[dataset]["test_gap"]
            )
            test_end = validate_date(
                test_start + params_by_dataset[dataset]["test_duration"] - 1
            )

            if test_end > last_stamp:
                break
            print(
                f"Running {model_name} on {dataset} from {train_start} to {train_end} and {test_start} to {test_end}"
            )

            train_set = data.filter_by_date(train_start, train_end)
            test_sets = []

            for offset in range(
                0,
                params_by_dataset[dataset]["test_duration"]
                - params_by_dataset[dataset]["test_stride"]
                + 1,
            ):
                cur_test_start = validate_date(test_start + offset)
                cur_test_end = validate_date(
                    cur_test_start + params_by_dataset[dataset]["test_stride"] - 1
                )
                test_sets.append(
                    (offset, data.filter_by_date(cur_test_start, cur_test_end))
                )

            X_train, X_val, y_train, y_val = train_test_split(
                train_set.features, train_set.labels, test_size=0.20, random_state=0
            )

            model = get_model(model_name)
            train(model, X_train, y_train)

            val_scores = eval(model, X_val, y_val)
            for offset, test_set in test_sets:
                X_test, y_test = test_set.features, test_set.labels
                cur_test_scores = eval(model, X_test, y_test)

                if offset not in statistics[model_name]:
                    statistics[model_name][offset] = []

                statistics[model_name][offset].append(
                    {
                        "val": val_scores,
                        "test": cur_test_scores,
                    }
                )

            # Move to next month after training set
            first_stamp = validate_date(first_stamp + 1)

    # Average results by offset
    for model_name in statistics:
        for offset in statistics[model_name]:
            statistics[model_name][offset] = {
                "val": {
                    "accuracy": np.mean(
                        [s["val"]["accuracy"] for s in statistics[model_name][offset]]
                    ),
                    "f1": np.mean(
                        [s["val"]["f1"] for s in statistics[model_name][offset]]
                    ),
                    "precision": np.mean(
                        [s["val"]["precision"] for s in statistics[model_name][offset]]
                    ),
                    "recall": np.mean(
                        [s["val"]["recall"] for s in statistics[model_name][offset]]
                    ),
                },
                "test": {
                    "accuracy": np.mean(
                        [s["test"]["accuracy"] for s in statistics[model_name][offset]]
                    ),
                    "f1": np.mean(
                        [s["test"]["f1"] for s in statistics[model_name][offset]]
                    ),
                    "precision": np.mean(
                        [s["test"]["precision"] for s in statistics[model_name][offset]]
                    ),
                    "recall": np.mean(
                        [s["test"]["recall"] for s in statistics[model_name][offset]]
                    ),
                },
            }

    all_dataset_statistics[dataset] = statistics


# Plot results with x-axis as test start and y-axis as f1 score
# Figure of size 10, 8
fix, ax = plt.subplots(1, 1, figsize=(10, 8))
color = cm.rainbow(np.linspace(0, 1, len(models)))
linestyles = ["dotted", "dashed", "solid"]
params = []

for linestyle, dataset in zip(linestyles, all_dataset_statistics):
    statistics = all_dataset_statistics[dataset]
    for c, model_name in zip(color, statistics):
        offsets = sorted(statistics[model_name].keys())
        ax.plot(
            [offset + params_by_dataset[dataset]["test_gap"] for offset in offsets],
            [statistics[model_name][offset]["test"]["f1"] for offset in offsets],
            label=f"{model_name} on {dataset}",
            color=c,
            linestyle=linestyle,
        )
        params.append(
            {
                "model": model_name,
                "dataset": dataset,
                "x": [
                    offset + params_by_dataset[dataset]["test_gap"]
                    for offset in offsets
                ],
                "y": [
                    statistics[model_name][offset]["test"]["f1"] for offset in offsets
                ],
                "label": f"{model_name} on {dataset}",
                "color": c,
                "linestyle": linestyle,
            }
        )

plt.xlabel("Months after last training month")
plt.ylabel("F1 score")
plt.title(f"Concept drift robustness versus time")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "experiment-2.png"))
with open(os.path.join(args.output_dir, f"experiment-2.json"), "w") as f:
    f.write(json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False))
# plt.show()


# Results with limited data, only on BODMAS (Or kronodroid maybe later)
params_by_dataset = {
    "bodmas": {
        "start_year": 2019,
        "start_month": 8,
        "end_year": 2020,
        "end_month": 9,
        "train_duration": 3,
        "test_gap": 1,
        "test_duration": 5,
        "test_stride": 2,
    },
}

models = ["LR", "RF"]

data_ratios = [1.0, 0.1, 0.01]
dataset = "bodmas"
all_dataset_statistics = {}
for data_ratio in data_ratios:
    data = PEMalwareDataset.from_name(dataset)

    last_stamp = (
        params_by_dataset[dataset]["end_year"] * 100
        + params_by_dataset[dataset]["end_month"]
    )

    statistics = {}

    for model_name in models:
        first_stamp = (
            params_by_dataset[dataset]["start_year"] * 100
            + params_by_dataset[dataset]["start_month"]
        )

        statistics[model_name] = {}
        while True:
            train_start = first_stamp
            train_end = validate_date(
                train_start + params_by_dataset[dataset]["train_duration"] - 1
            )
            test_start = validate_date(
                train_end + params_by_dataset[dataset]["test_gap"]
            )
            test_end = validate_date(
                test_start + params_by_dataset[dataset]["test_duration"] - 1
            )

            if test_end > last_stamp:
                break
            print(
                f"Running {model_name} on {dataset} from {train_start} to {train_end} and {test_start} to {test_end}"
            )

            train_set = data.filter_by_date(train_start, train_end)
            test_sets = []

            for offset in range(
                0,
                params_by_dataset[dataset]["test_duration"]
                - params_by_dataset[dataset]["test_stride"]
                + 1,
            ):
                cur_test_start = validate_date(test_start + offset)
                cur_test_end = validate_date(
                    cur_test_start + params_by_dataset[dataset]["test_stride"] - 1
                )
                test_sets.append(
                    (offset, data.filter_by_date(cur_test_start, cur_test_end))
                )

            X_train, X_val, y_train, y_val = train_test_split(
                train_set.features,
                train_set.labels,
                test_size=(1 - 0.80 * data_ratio),
                random_state=0,
            )

            model = get_model(model_name)
            train(model, X_train, y_train)

            val_scores = eval(model, X_val, y_val)
            for offset, test_set in test_sets:
                X_test, y_test = test_set.features, test_set.labels
                cur_test_scores = eval(model, X_test, y_test)

                if offset not in statistics[model_name]:
                    statistics[model_name][offset] = []

                statistics[model_name][offset].append(
                    {
                        "val": val_scores,
                        "test": cur_test_scores,
                    }
                )

            # Move to next month after training set
            first_stamp = validate_date(first_stamp + 1)

    # Average results by offset
    for model_name in statistics:
        for offset in statistics[model_name]:
            statistics[model_name][offset] = {
                "val": {
                    "accuracy": np.mean(
                        [s["val"]["accuracy"] for s in statistics[model_name][offset]]
                    ),
                    "f1": np.mean(
                        [s["val"]["f1"] for s in statistics[model_name][offset]]
                    ),
                    "precision": np.mean(
                        [s["val"]["precision"] for s in statistics[model_name][offset]]
                    ),
                    "recall": np.mean(
                        [s["val"]["recall"] for s in statistics[model_name][offset]]
                    ),
                },
                "test": {
                    "accuracy": np.mean(
                        [s["test"]["accuracy"] for s in statistics[model_name][offset]]
                    ),
                    "f1": np.mean(
                        [s["test"]["f1"] for s in statistics[model_name][offset]]
                    ),
                    "precision": np.mean(
                        [s["test"]["precision"] for s in statistics[model_name][offset]]
                    ),
                    "recall": np.mean(
                        [s["test"]["recall"] for s in statistics[model_name][offset]]
                    ),
                },
            }

    all_dataset_statistics[data_ratio] = statistics


# Plot results with x-axis as test start and y-axis as f1 score
# Figure of size 10, 8
fix, ax = plt.subplots(1, 1, figsize=(10, 8))
color = cm.rainbow(np.linspace(0, 1, len(models)))
linestyles = ["dotted", "dashed", "solid"]
params = []

for linestyle, data_ratio in zip(linestyles, all_dataset_statistics):
    statistics = all_dataset_statistics[data_ratio]
    for c, model_name in zip(color, statistics):
        offsets = sorted(statistics[model_name].keys())
        ax.plot(
            [offset + params_by_dataset[dataset]["test_gap"] for offset in offsets],
            [statistics[model_name][offset]["test"]["f1"] for offset in offsets],
            label=f"{model_name}, {data_ratio}% data",
            color=c,
            linestyle=linestyle,
        )
        params.append(
            {
                "model": model_name,
                "data_ratio": data_ratio,
                "x": [
                    offset + params_by_dataset[dataset]["test_gap"]
                    for offset in offsets
                ],
                "y": [
                    statistics[model_name][offset]["test"]["f1"] for offset in offsets
                ],
                "label": f"{model_name}, {data_ratio}% data",
                "color": c,
                "linestyle": linestyle,
            }
        )

plt.xlabel("Months after last training month")
plt.ylabel("F1 score")
plt.title(f"Concept drift robustness versus time on {dataset} dataset")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "experiment-3.png"))
with open(os.path.join(args.output_dir, "experiment-3-params.json"), "w") as f:
    f.write(json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False))
# plt.show()
