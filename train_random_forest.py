from util.dataset import Dataset

dataset = Dataset("transactions.csv.zip")
dataset.extract_to(".")
df = dataset.load_data()