from data_loaders.dataset import CSVDataset

ds = CSVDataset.from_path("C:/Users/MOHITH/practice/pytorch/pr/attempt-1/DATASET B/1/71.csv/71.csv")
print(ds.data.head())