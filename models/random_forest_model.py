import os

from models.Model import Model


class RandomForestModel(Model):
    def dump_model(self, dump_to):
        os.makedirs(dump_to, exist_ok=True)
