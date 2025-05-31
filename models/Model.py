from abc import abstractmethod


class Model():


    @abstractmethod
    def dump_model(self,dump_to):
        """
            Saves the trained model and scaler to the specified directory.

            Args:
                dump_to (str): The path to the subfolder where the files should be saved.
        """
        pass