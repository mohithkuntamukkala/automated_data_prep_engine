from dataset import Dataset
from typing import List

class DatasetBundle():
    def __init__(self,train_datasets,val_datasets,test_datasets):
        self.train_datasets : List[Dataset] | None = train_datasets
        self.val_datasets: List[Dataset] | None = val_datasets
        self.test_datasets: List[Dataset] | None = test_datasets
        