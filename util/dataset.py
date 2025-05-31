import os
import zipfile
import pandas as pd


class Dataset:
    """
       A class for handling datasets stored inside ZIP archives.
       Provides methods for extracting, loading, and splitting the dataset.
    """
    def __init__(self, zip_archive_path):
        """
            Initialize the Dataset with the path to a ZIP archive.
            Args:
                zip_archive_path (str): Path to the ZIP file containing the dataset.
        """
        self.path = zip_archive_path
        self.extractLocation = ''

    def extract_to(self, location):
        """
            Extracts the ZIP archive to the specified location.

            Args:
                location (str): Directory path where the archive should be extracted.
        """
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            zip_ref.extractall(location)
            self.extractLocation = location

    def load_data(self):
        """
            Loads the first CSV file from the extracted archive directory.

            Returns:
                pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        csv_file = [f for f in os.listdir(self.extractLocation) if f.endswith(".csv")][0]
        return pd.read_csv(csv_file)

    def split(self,df,target_column):
        """
            Splits a DataFrame into features and target column.

            Args:
                df (pd.DataFrame): The DataFrame to split.
                target_column (str): Name of the column to use as the target.

            Returns:
                Tuple[pd.DataFrame, pd.Series]: A tuple (X, y) with features and target.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y