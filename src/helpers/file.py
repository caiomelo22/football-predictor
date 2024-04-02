import os
import pandas as pd
from joblib import dump, load


def save_file(directory, file, file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)

    dump(file, f"{directory}/{file_name}.joblib")

def load_from_file(path, file_name):
    return load(f"{path}/{file_name}.joblib")

def parse_df_to_csv(dataframe, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
        
    dataframe.to_csv("{}/{}".format(path, filename))
