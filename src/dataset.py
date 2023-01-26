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


class KronodroidDataset(MalwareDataset):
    def __init__(self, path, p_type='emu', date='LastModDate', normalize=True, shuffle=True):
        """
        path: directory containing *_v1.csv files.
            
        Reads 'legitimate' and 'malware' csv files of 'p_type: emu|real' separately and merges them.
        """
        assert p_type in ['emu', 'real'], "program type must be one of: 'emu', 'real'!"
        assert date in ['FirstModDate', 'LastModDate'], "used date must be one of 'FirstModDate', 'LastModDate'"
        self.path = path
        # Read DataFrames
        path_legit = os.path.join(path, f"{p_type}_legitimate_v1.csv")
        legit_df = pd.read_csv(path_legit, low_memory=False)
        
        path_malware = os.path.join(path, f"{p_type}_malware_v1.csv")
        malware_df = pd.read_csv(path_legit, low_memory=False)
        
        
        # Pre-process (drop some columns and fix nans)
        legit_df = fix_columns(legit_df)
        malware_df = fix_columns(malware_df)
        
        legit_df, malware_df = filter_matching_columns(legit_df, malware_df)
    
        # Merged dataset
        dataset = pd.concat([legit_df, malware_df])
    
        # Parse dates
        firstmoddate = list(map(lambda x: parse_date(x), dataset['FirstModDate']))
        dataset['FirstModDate'] = firstmoddate
        
        lastmoddate = list(map(lambda x: parse_date(x), dataset['LastModDate']))
        dataset['LastModDate'] = lastmoddate
        self.dates = dataset['LastModDate']
        dataset.drop(['FirstModDate', 'LastModDate'], 'columns', inplace=True) 
        dataset = dataset.to_numpy()
        self.labels = dataset[:, 0].astype(np.int)
        self.features = dataset[:, 1:].astype(np.float)
        if normalize:
            self.features = StandardScaler().fit_transform(self.features)
        if shuffle:
            idx = np.arange(self.features.shape[0])
            np.random.shuffle(idx)
            self.features = self.features[idx]
            self.labels = self.labels[idx]
            
    def get_feature_width(self):
        return self.features.shape[1]
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
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


    def __repr__(self):
        return "Dataset: {}\nInstances:{}".format(self.path, self.features.shape[0])

    

def fix_columns(df):
    """
    Drop irrelevant columns
    Replace NaNs with 0 in all columns
    """

    drop_columns = []
    fix_nan_columns = []
    for col_name in df.columns:
        if df[col_name].dtype != 'int64' and col_name not in ['FirstModDate', 'LastModDate']:
            if 'Nr' in col_name:
                fix_nan_columns.append(col_name)
            else:
                # print(col_name, "\t", df[col_name].dtype)
                drop_columns.append(col_name)

                
    for col in fix_nan_columns:
        df[col].replace('None', 0, inplace=True)
        df[col] = df[col].astype(int)

    df.drop(drop_columns, 'columns', inplace=True)
    return df

def parse_date(date):
    try:
        m, _, y = date.split('/')
    except:
        y, m, _ = date.split('-')
    m = str(m).zfill(2)
    return int(y+m)


def filter_matching_columns(df1, df2):
    df1_columns = df1.columns
    df2_columns = df2.columns

    drop_columns = list(set(df1) - set(df2))
    df1.drop(drop_columns, 'columns', inplace=True)

    drop_columns = list(set(df2) - set(df1))
    df2.drop(drop_columns, 'columns', inplace=True)
    return df1, df2
