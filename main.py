import pandas as pd
import os
from glob import glob


DF_CONSOLE_WIDTH = 500
pd.set_option('display.width', DF_CONSOLE_WIDTH)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

DATA_PATH = 'data/train'
IMAGE_WIDTH = 266
IMAGE_HEIGHT = 266


def create_path_to_slice(root_dir, case_id):
    case_folder = case_id.split("_")[0]
    day_folder = "_".join(case_id.split("_")[:2])
    file_starter = "_".join(case_id.split("_")[2:])
    # fetching folder paths
    folder = os.path.join(root_dir, case_folder, day_folder, "scans")
    # fetching filenames with similar pattern
    file = glob(f"{folder}/{file_starter}*")
    # returning the first file, though it will always hold one file.
    return file[0]


def preprocess():
    df = pd.read_csv('data/train.csv')
    df['segmentation'] = df['segmentation'].astype('str')
    df['case_id'] = df['id'].apply(lambda x: x.split("_")[0][4:])
    df['day_id'] = df['id'].apply(lambda x: x.split("_")[1][3:])
    df['slice_id'] = df['id'].apply(lambda x: x.split("_")[-1])
    df['path'] = df['id'].apply(lambda x: create_path_to_slice(DATA_PATH, x))
    df.to_csv('data/preprocessed_train.csv')
    print(df[df['segmentation'] != 'nan'].head())
    print(df.head())


if __name__ == '__main__':
    df_ = pd.read_csv('data/preprocessed_train.csv')
    print(df_[df_['case_id'] == 123])
