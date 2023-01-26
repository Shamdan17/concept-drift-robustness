import os
import sys
from glob import glob
import pandas as pd
import numpy as np
from dataset import MalwareDataset
from sklearn.preprocessing import StandardScaler
import copy

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
