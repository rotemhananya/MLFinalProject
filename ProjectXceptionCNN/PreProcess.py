import pandas as pd
import numpy as np
import tensorflow as tf
import time as t
import math
import os
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import Globals as g
from sklearn.preprocessing import OneHotEncoder


def transform_y(arr):
    """ Transforms y from vector to one-hot representation"""
    enc = OneHotEncoder()
    oh = enc.fit_transform(arr.reshape(-1, 1))
    return oh.toarray()


def apply_filter(fil):
    """ Applies the given filter on the Base Image """
    global rev_image_arr
    x = tf.constant(rev_image_arr, dtype=tf.float32)
    kernel = tf.constant(fil, dtype=tf.float32)
    return np.array(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')).reshape((1, 224, 224, 3))


def create_filter(sample):
    """ Creates a filter from a given sample """
    k = int(math.sqrt(len(sample)))
    fil = np.array(sample).reshape((k, k))
    return (fil - np.mean(fil)).reshape((k, k, 1, 1))


def add_padding(df):
    """ Adds Padding to all of the samples to match the closest odd k^2 number (i.e., 9(3x3), 25(5x5), 49(7x7), ....)
        e.g. len(sample) == 34 ---padding to--> 49 """
    number_of_features_sqrt = int(math.sqrt(df.shape[1])) + 1
    if number_of_features_sqrt == math.sqrt(df.shape[1]) + 1 and number_of_features_sqrt % 2 == 0:
        return np.array(df)
    padding_number = number_of_features_sqrt ** 2 - df.shape[1] if number_of_features_sqrt % 2 == 1 \
        else (number_of_features_sqrt + 1) ** 2 - df.shape[1]
    padding_df = pd.DataFrame({f'p{i + 1}': np.zeros(df.shape[0]) for i in range(padding_number)})
    return np.array(pd.concat((df, padding_df), axis=1))


def split_train_test(X, y, test_size=0.25):
    """ split X, y to train and test using test size """
    return train_test_split(X, y, test_size=test_size, random_state=73)


def get_X_array(X):
    """ Transform X form tabular to Image set, by using each sample as a filter on tha Base Image """
    samples = []
    for sample in X:
        fil = create_filter(sample)
        samples.append(apply_filter(fil))
    return np.concatenate(samples, axis=0)


def preprocess(ds_name):
    """ Given the dataset name this function splits the dataset to 10-Folds, for cross validation.
        The origin dataset is tabular the out but sets are """
    global dir_path, image_arr
    st = t.time_ns()
    df = pd.read_csv(f'{dir_path}{ds_name}.csv')
    # adds padding to the dataset
    X, y = add_padding(df[df.columns[:-1]]), np.array(df[df.columns[-1]])
    #  Creating output dirs. If using linux these to lines need to be change to:
    # 1. mkdir -p "Data/Datasets/Processed/{ds_name}/train/"
    # 2. mkdir -p "Data/Datasets/Processed/{ds_name}/test/"
    os.system(f'if not exist "Data/Datasets/Processed/{ds_name}/train/" mkdir "Data/Datasets/Processed/{ds_name}/train/"')
    os.system(f'if not exist "Data/Datasets/Processed/{ds_name}/test/" mkdir "Data/Datasets/Processed/{ds_name}/test/"')

    # splits the data to 10-sets, by using 10-fold CV
    skf = StratifiedKFold(n_splits=10)
    for i, (train_index, test_index) in zip(range(10), skf.split(X, y)):
        print(f'Starting Fold: {ds_name} fold-{i}')
        mt = t.time_ns()
        np.save(f'Data/Datasets/Processed/{ds_name}/train/x{i}', get_X_array(X[train_index]))
        np.save(f'Data/Datasets/Processed/{ds_name}/train/y{i}', transform_y(y[train_index]))
        np.save(f'Data/Datasets/Processed/{ds_name}/test/x{i}', get_X_array(X[test_index]))
        np.save(f'Data/Datasets/Processed/{ds_name}/test/y{i}', transform_y(y[test_index]))
        print(f'Finished Fold: {i} Time: {g.get_time_passed(mt, u="sec")} sec')
    print(f'Finished: {ds_name} Time: {g.get_time_passed(st)} min')


dir_path = 'Data/Datasets/Raw/'
base_image = Image.open('Data/Base Images/Rick&Morty.png')
image_arr = np.asarray(base_image) / 255 #  Image.fromarray()
rev_image_arr = np.array(image_arr).reshape((3, 224, 224, 1))
datasets_names = g.datasets_names
st = t.time_ns()
ns = 10 ** 9

# Applies the preprocessing on all dataset in datasets_names from Globals.py
for ds in datasets_names:
    mt = t.time_ns()
    print(f'Start Processing: {ds}')
    preprocess(ds)
    print(f'Finished Processing: {ds} Time: {g.get_time_passed(mt)}')

print(f'Finished - Total Time: {g.get_time_passed(st)}')