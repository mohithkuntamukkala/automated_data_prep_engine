from features.base import FeatureRelationship
import pandas as pd
import numpy as np

class NumericalToNumericalRelation(FeatureRelationship):
    def __init__(self,df,col1,col2,pearson,spearman):
        super().__init__()
        self.column_names = [col1,col2]
        self.pearson = pearson
        self.spearman = spearman
        
class NumericalToCategoricalRelation(FeatureRelationship):
    def __init__(self,df,col1,col2,eta_squared):
        super().__init__()
        self.column_names = [col1,col2]
        self.eta_squared = eta_squared
        self.group_means = df.groupby(col2)[col1].mean().to_dict()
        self.group_medians = df.groupby(col2)[col1].median().to_dict()
        
class NumericalToBooleanRelation(FeatureRelationship):
    def __init__(self,df,col1,col2,point_biserial_corr):
        self.column_names = [col1,col2]
        self.point_biserial_corr = point_biserial_corr
        self.mean_true = df.loc[df[col2].astype(bool), col1].mean()
        self.mean_false = df.loc[~df[col2].astype(bool), col1].mean()
        
class CategoricalToCategoricalRelation(FeatureRelationship):
    def __init__(self,df,col1,col2,cramers_v):
        super().__init__()
        self.column_names = [col1,col2]
        self.cramers_v = cramers_v
        self.one_to_one_mapping = (
            df.groupby(col1)[col2].nunique().max() == 1
            and
            df.groupby(col2)[col1].nunique().max() == 1
        )
        
class CategoricalToBooleanRelation(FeatureRelationship):
    def __init__(self,df,col1,col2,cramers_v):
        super().__init__()
        self.column_names = [col1,col2]
        self.cramers_v = cramers_v
        self.positive_rate_per_category = (df.groupby(col1)[col2].mean().to_dict())
        
def create_features(df,column_to_feature,feature_type_to_column):
    relations = []
    return relations
    # first look for numerical. check categorical,numerical,boolean
    # then for categorical. check categorical,boolean
    # dont create duplicate relations