import logging
import os
import pathlib
from pathlib import Path
import uuid
import pandas as pd
import shutil
from features.base import Feature,FeatureRelationship
from typing import List
from features.utils import classify_features_from_df,FEATURE_CLASSES
from features.features import NumericalFeature,CategoricalFeature,TextFeature,DateTimeFeature,BooleanFeature 
from features.relationships import create_feature_relations

logger = logging.getLogger(__name__)

class Dataset():
    def __init__(self):
        self.data = None
        self.source_path = None
        self.source_file = None
        self.working_path = None
        self.dataset_id = f'file-{uuid.uuid4().hex}'
    @classmethod
    def from_path(cls,path):
        obj = cls()
        obj.source_path = Path(path)
        try:
            obj.working_path = Path('temp') / obj.dataset_id / obj.source_path.name
            obj.working_path.parent.mkdir(parents=True,exist_ok=True)
            obj.save_file_from_local(obj.source_path,obj.working_path)
            obj.data = obj.load_data(obj.working_path)
            return obj
        except Exception:
            logger.exception(f'Failed to load file from {obj.source_path}')
            raise
    @classmethod
    def from_streamlit_file_upload(cls,file):
        obj = cls()
        obj.source_file = file
        try:
            obj.working_path = Path('temp') / obj.dataset_id / obj.source_file.name
            obj.working_path.parent.mkdir(parents=True,exist_ok=True)
            obj.save_uploaded_file(obj.source_file,obj.working_path)
            obj.data = obj.load_data(obj.working_path)
            return obj
        except Exception:
            logger.exception(f'Failed to load file {obj.source_file.name}')
            raise
    def save_file_from_local(self,source_path,working_path):
        shutil.copy2(src = source_path, dst = working_path)
    def load_data(self,path):
        pass
    def save_uploaded_file(self,file,path):
        with open(path,'wb') as f:
            f.write(file.read())
        file.seek(0)

class CSVDataset(Dataset):
    def __init__(self,separator = ',',encoding = 'utf-8'):
        super().__init__()
        self.separator = separator
        self.encoding = encoding
        self.features : List[Feature] = []
        self.n_features = 0
        self.n_samples = 0
        self.total_cells = 0
        self.n_duplicate_samples = 0
        self.duplicate_sample_percentage = 0.0
        self.n_missing_cells = 0
        self.missing_cell_percentage = 0
        self.feature_type_to_column = None
        self.column_to_feature = {}
        self.column_to_type = {}
        self.feature_relations : List[FeatureRelationship] = []
    def load_data(self,path):
        df = pd.read_csv(path,sep = self.separator, encoding = self.encoding)
        self.n_features,self.n_samples = df.shape[1],df.shape[0]
        self.total_cells = self.n_samples*self.n_features
        self.n_duplicate_samples = df.duplicated().sum()
        self.duplicate_sample_percentage = (self.n_duplicate_samples/self.n_samples)*100
        self.n_missing_cells = df.isna().sum().sum()
        self.missing_cell_percentage = (self.n_missing_cells/self.total_cells)*100
        self.feature_type_to_column = classify_features_from_df(df)
        for class_ in FEATURE_CLASSES:
            class_list = self.feature_type_to_column[class_]
            for column_name in class_list:
                feature = load_feature(class_,df,column_name)
                self.features.append(feature)
                self.column_to_feature[column_name] = feature
                self.column_to_type[column_name] = class_
        self.feature_relations = create_feature_relations(df,self.column_to_type)
        return df
    
def load_feature(class_type,df,column_name):
    if class_type == 'Numerical':
        return NumericalFeature.from_series(df,column_name)
    if class_type == 'Categorical':
        return CategoricalFeature.from_series(df,column_name)
    if class_type == 'Boolean':
        return BooleanFeature.from_series(df,column_name)
    if class_type == 'Text':
        return TextFeature.from_series(df,column_name)
    if class_type == 'Datetime':
        return DateTimeFeature.from_series(df,column_name)