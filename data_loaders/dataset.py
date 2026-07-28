import logging
import os
import pathlib
from pathlib import Path
import uuid
import pandas as pd
import shutil

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
    def load_data(self,path):
        return pd.read_csv(path,sep = self.separator, encoding = self.encoding)