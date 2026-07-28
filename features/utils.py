import pandas as pd
from collections import defaultdict

CLASSES = [
    'Numerical',
    'Categorical',
    'Boolean',
    'Text',
    'Datetime'
]

def classify_features_from_df(df: pd.DataFrame):
    columns = df.columns
    category_map = defaultdict(list)
    for col in columns:
        category_map[classify_feature(df[col])].append(col)
    return category_map
        
def classify_feature(series: pd.Series):
    if str(series.dtype).lower().startswith('int'):
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / len(series)
        if n_unique <= 2:
            return "Boolean"
        elif n_unique <= 20 and unique_ratio <= 0.05:
            return "Categorical"
        else:
            return "Numerical"
    elif str(series.dtype).lower().startswith('float'):
        return 'Numerical'
    else:
        dt_parsed = pd.to_datetime(series,errors = 'coerce')
        if dt_parsed.notna().mean() >= 0.95:
            return 'Datetime'
        if series.dtype == 'boolean':
            return 'Boolean'
        if series.nunique(dropna = True) == 2:
            vals = set(series.dropna().astype(str).str.strip().str.lower())
            matches = {'0','1','y','n','yes','no','true','false','t','f'}
            if vals.issubset(matches):
                return 'Boolean'
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / len(series)
        strings = series.dropna().astype(str)
        if not strings.empty:
            avg_length = strings.str.len().mean()
            avg_words = strings.str.split().str.len().mean()
            if (unique_ratio > 0.3 and (avg_length >= 15 or avg_words >= 3)):
                return "Text"
        return 'Categorical'
    
        
    