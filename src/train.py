import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from dataset import PEMalwareDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Training and evaluation script', add_help=False)
    parser.add_argument('--model_type', default='', type=str)
    parser.add_argument('--train_start_date', default=201908, type=int, help='year in 4 digits followed by month in 2 digits')
    parser.add_argument('--train_end_date', default=201912, type=int, help='year in 4 digits followed by month in 2 digits')
    parser.add_argument('--test_start_date', default=202001, type=int, help='year in 4 digits followed by month in 2 digits')
    parser.add_argument('--test_end_date', default=202004, type=int, help='year in 4 digits followed by month in 2 digits')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--data_root', default='data/', type=str)
    parser.add_argument('--output_dir', default='exps/', type=str)
    parser.add_argument('--feat_select', action='store_true', help='whether to apply feature selection or not')
    parser.add_argument('--top_k_feat', default=1000, type=int, help='top k features to be selected')


    return parser

def select_top_features(data, top_k):
    top_feat = SelectKBest(f_classif, k=top_k).fit_transform(data.features, data.labels)
    data.features = top_feat
    return data

def main(args):

    np.random.seed(0)

    print('Loading the dataset...')
    data = PEMalwareDataset.from_name(args.dataset)

    if args.feat_select:
        data = select_top_features(data, args.top_k_feat)

    train_set = data.filter_by_date(args.train_start_date, args.train_end_date)
    test_set = data.filter_by_date(args.test_start_date, args.test_end_date)

    counts = Counter(train_set.labels) 
    assert len(counts) > 1, 'Training data only has 1 class, consider expanding the time period'

    X_train, X_val, y_train, y_val = \
        train_test_split(train_set.features, train_set.labels, test_size=0.20, random_state=0) 
    
    print('Dataset is loaded.')
    print('Train set size: {0}, Val. set size: {1}, Test set size: {2}'.format(X_train.shape[0], X_val.shape[0], test_set.labels.shape[0]))
    print('Training the model...')

    if args.model_type == 'DT':
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif args.model_type == 'XGBoost':
        model = XGBClassifier(max_depth=5, min_child_weight=1, gamma=0, learning_rate=0.1 ,n_estimators=100, random_state=0)
    elif args.model_type == 'LR':
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=8000, random_state=0)
    elif args.model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif args.model_type == 'GNB':
        model = GaussianNB(var_smoothing=1e-1)
    elif args.model_type == 'RF':
        model = RandomForestClassifier(n_estimators=10)
    else:
        raise NotImplementedError

    model.fit(X_train, y_train)

    print('Training completed.')
    print('Evaluating...')

    preds = model.predict(X_train)
    accuracy = accuracy_score(y_train, preds)
    print('Accuracy on training data (same time period):',round(accuracy,3))

    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    print('Accuracy on validation data (same time period):',round(accuracy,3))

    preds = model.predict(test_set.features)
    accuracy = accuracy_score(test_set.labels, preds)
    print('Accuracy on test data (different time period):',round(accuracy,3))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
