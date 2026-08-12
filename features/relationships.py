from features.base import FeatureRelationship
import pandas as pd
import numpy as np
from features.relation_utils import pearson_coefficient,spearman_coefficient,cramers_v,point_biserial,eta_squared

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
        super().__init__()
        self.column_names = [col1,col2]
        self.point_biserial_corr = point_biserial_corr
        mapping = {
            "true": True, "false": False,
            "yes": True,  "no": False,
            "y": True,    "n": False,
            "1": True,    "0": False,
        }
        b = (
            df[col2]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )
        self.mean_true = df.loc[b == True, col1].mean()
        self.mean_false = df.loc[b == False, col1].mean()
        # mapping = {
        #     "true": True, "false": False,
        #     "yes": True,  "no": False,
        #     "y": True,    "n": False,
        #     "1": True,    "0": False,
        # }

        # b = (
        #     df[col2]
        #     .astype(str)
        #     .str.strip()
        #     .str.lower()
        #     .map(mapping))
        # self.mean_true = df.loc[b.astype(bool), col1].mean()
        # self.mean_false = df.loc[~b.astype(bool), col1].mean()

        # self.mean_true = df.loc[df[col2].astype(bool), col1].mean()
        # self.mean_false = df.loc[~df[col2].astype(bool), col1].mean()
        
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
        
def create_feature_relations(df,column_to_type):
    relations = []
    columns = df.columns
    for i,col in enumerate(columns):
        if column_to_type[columns[i]] in {'Datetime','Text','Boolean'}:
            continue 
        for j in range(i+1,len(columns)):
            if column_to_type[columns[j]] in {'Datetime','Text'}:
                continue
            if column_to_type[columns[i]] == 'Numerical':
                if column_to_type[columns[j]] == 'Numerical':
                    pearson = pearson_coefficient(df,columns[i],columns[j]) 
                    spearman = spearman_coefficient(df,columns[i],columns[j])
                    if abs(spearman) >= 0.7 or abs(pearson) >= 0.7:
                        relations.append(NumericalToNumericalRelation(df,columns[i],columns[j],pearson,spearman))
                if column_to_type[columns[j]] == 'Categorical':
                    eta_2 = eta_squared(df,columns[i],columns[j])
                    if eta_2 >= 0.14:
                        relations.append(NumericalToCategoricalRelation(df,columns[i],columns[j],eta_2))
                if column_to_type[columns[j]] == 'Boolean':
                    point_2 = point_biserial(df,columns[i],columns[j])
                    if abs(point_2) >= 0.3:
                        relations.append(NumericalToBooleanRelation(df,columns[i],columns[j],point_2))
            if column_to_type[columns[i]] == 'Categorical':
                if column_to_type[columns[j]] == 'Categorical':
                    cramer = cramers_v(df,columns[i],columns[j])
                    if cramer >= 0.5:
                        relations.append(CategoricalToCategoricalRelation(df,columns[i],columns[j],cramer))
                if column_to_type[columns[j]] == 'Boolean':
                    cramer = cramers_v(df,columns[i],columns[j])
                    if cramer >= 0.3:
                        relations.append(CategoricalToBooleanRelation(df,columns[i],columns[j],cramer))    
    return relations
    # first look for numerical. check categorical,numerical,boolean
    # then for categorical. check categorical,boolean
    # dont create duplicate relations