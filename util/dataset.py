import os
import zipfile
import pandas as pd


class Dataset:
    def __init__(self, zip_archive_path):
        self.path = zip_archive_path
        self.extractLocation = ''

    def extract_to(self, location):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            zip_ref.extractall(location)
            self.extractLocation = location

    def load_data(self):
        csv_file = [f for f in os.listdir(self.extractLocation) if f.endswith(".csv")][0]
        return pd.read_csv(csv_file)

    def split(self,df,target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y