import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler

PE_DATASETS = {
    "bodmas": ("data/bodmas.npz", "data/bodmas_metadata.csv"),
    "ember": ("data/ember.npz", "data/ember_metadata.csv"),
}


# Generic Malware Dataset class
class MalwareDataset:
    # Constructor
    def __init__(self):
        raise NotImplementedError

    # Get feature width
    def get_feature_width(self):
        raise NotImplementedError

    # Get the number of samples in the dataset
    def __len__(self):
        raise NotImplementedError

    # Get a sample from the dataset given an index
    def __getitem__(self, index):
        raise NotImplementedError

    # Filter dataset by date
    def filter_by_date(self, start_date, end_date):
        raise NotImplementedError

    def filter_by_lambda(
        self,
        filter_lambda,
        include_label=False,
        include_index=False,
        include_date=False,
    ):
        raise NotImplementedError


class PEMalwareDataset(MalwareDataset):
    """
    Class for loading and filtering PE malware datasets

    Attributes:
        path (str): Path to the dataset
        metadata_path (str): Path to the metadata file

        features (np.ndarray): Array of features
        labels (np.ndarray): Array of labels
        dates (np.ndarray): Array of dates

    Usage:
        # By path
        dataset = PEMalwareDataset("data/bodmas.npz", "data/bodmas_metadata.csv")
        # By name
        dataset = PEMalwareDataset.from_name("bodmas")
        # Filter by date
        dataset = dataset.filter_by_date(201701, 201712)
    """

    # Constructor
    def __init__(self, path, metadata_path, normalize=True):
        self.path = path
        self.metadata_path = metadata_path

        # Load the dataset
        dataset = np.load(path)

        self.features = dataset["X"]

        if normalize:
            self.features = StandardScaler().fit_transform(self.features)

        self.labels = dataset["y"].astype(np.int)

        # Load the metadata
        metadata = pd.read_csv(metadata_path)

        # Extract and concatenate the year and month
        dates = [
            int(metadata.iloc[i][1][:7].replace("-", "")) for i in range(len(metadata))
        ]

        self.dates = np.array(dates)

    # Get feature width
    def get_feature_width(self):
        return self.features.shape[1]

    # Get the number of samples in the dataset
    def __len__(self):
        return len(self.features)

    # Get a sample from the dataset given an index
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    # Filter dataset by date
    def filter_by_date(self, start_date, end_date, max_instances=50000):
        # Filter by date
        mask = np.logical_and(self.dates >= start_date, self.dates <= end_date)

        # Perform shallow copy
        selfcopy = copy.copy(self)

        selfcopy.features = self.features[mask]
        selfcopy.labels = self.labels[mask]
        selfcopy.dates = self.dates[mask]

        # If the number of instances is greater than max_instances, sample
        if len(selfcopy) > max_instances:
            # Sample the dataset
            sampled_indices = np.random.choice(
                np.arange(len(selfcopy)), max_instances, replace=False
            )

            selfcopy.features = selfcopy.features[sampled_indices]
            selfcopy.labels = selfcopy.labels[sampled_indices]
            selfcopy.dates = selfcopy.dates[sampled_indices]

        # Return the filtered dataset
        return selfcopy

    def filter_by_lambda(
        self,
        filter_lambda,
        include_label=False,
        include_index=False,
        include_date=False,
    ):
        to_iterate = [self.features]

        if include_label:
            to_iterate.append(self.labels)

        if include_index:
            to_iterate.append(np.arange(len(self)))

        if include_date:
            to_iterate.append(self.dates)

        mask = [filter_lambda(*args) for args in zip(*to_iterate)]

        mask = np.array(mask)

        # Perform shallow copy
        selfcopy = copy.copy(self)

        selfcopy.features = self.features[mask]
        selfcopy.labels = self.labels[mask]
        selfcopy.dates = self.dates[mask]

        # Return the filtered dataset
        return selfcopy

    # Static method to instantiate by dataset name
    @staticmethod
    def from_name(name):
        if name not in PE_DATASETS:
            raise ValueError("Dataset not found")

        return PEMalwareDataset(*PE_DATASETS[name])

    def __repr__(self):
        return "Dataset: {}\nInstances:{}".format(self.path, len(self))
