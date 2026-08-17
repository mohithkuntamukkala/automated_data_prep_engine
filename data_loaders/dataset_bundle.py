from data_loaders.dataset import Dataset,CSVDataset
from typing import List
import uuid

class DatasetBundle():
    def __init__(self):
        self.train_dataset_paths = None
        self.val_dataset_paths = None
        self.test_dataset_paths = None
        self.train_datasets : List[Dataset] | None  = []
        self.val_datasets : List[Dataset] | None  = []
        self.test_datasets : List[Dataset] | None  = []
        self.dataset_bundle_id = f'datasets-{uuid.uuid4().hex}'
    @classmethod
    def from_paths(cls,train_dataset_paths,val_dataset_paths,test_dataset_paths):
        obj = cls()
        if train_dataset_paths: 
            obj.train_dataset_paths = train_dataset_paths
            for train_path in obj.train_dataset_paths:
                if train_path.endswith('.csv'):
                    obj.train_datasets.append(CSVDataset.from_path(train_path,obj.dataset_bundle_id))
        if val_dataset_paths: 
            obj.val_dataset_paths = val_dataset_paths
            for val_path in obj.val_dataset_paths:
                if val_path.endswith('.csv'):
                    obj.val_datasets.append(CSVDataset.from_path(val_path,obj.dataset_bundle_id))
        if test_dataset_paths: 
            obj.test_dataset_paths = test_dataset_paths
            for test_path in obj.test_dataset_paths:
                if test_path.endswith('.csv'):
                    obj.test_datasets.append(CSVDataset.from_path(test_path,obj.dataset_bundle_id))

        return obj