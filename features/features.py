from base import Feature
import numpy as np
import pandas as pd

class NumericalFeature(Feature):
    # add histogram before passing to agent for analysis
    def __init__(self):
        super().__init__()
        self.mean = None
        self.median = None
        self.max = None
        self.min = None
        self.range = None
        self.mode = None
        self.var = None
        self.sd = None
        self.skewness = None
        self.kurtosis = None
        self.percentile_5 = None
        self.percentile_95 = None
        self.q3 = None
        self.q1 = None
        self.iqr = None

    @classmethod
    def from_series(cls, df, column_name):
        obj = super().from_series(df, column_name)
        s = df[column_name]
        obj.mean = s.mean()
        obj.median = s.median()
        obj.max = s.max()
        obj.min = s.min()
        obj.range = (obj.min, obj.max)
        obj.mode = s.mode().tolist()
        obj.var = s.var()
        obj.sd = s.std()
        obj.skewness = s.skew()
        obj.kurtosis = s.kurt()
        obj.percentile_5 = s.quantile(0.05)
        obj.percentile_95 = s.quantile(0.95)
        obj.q3 = s.quantile(0.75)
        obj.q1 = s.quantile(0.25)
        obj.iqr = obj.q3 - obj.q1
        return obj
        
class CategoricalFeature(Feature):
    def __init__(self):
        super().__init__()
        self.mode = None
        self.freq = None
        self.class_proportions = None
        self.dominant_class_percentage = None
        self.rare_class_count = None
        self.gini_impurity = None
        self.shannon_entropy = None

    @classmethod
    def from_series(cls, df, column_name):
        obj = super().from_series(df, column_name)
        s = df[column_name]
        obj.mode = s.mode().tolist()
        obj.freq = s.value_counts(dropna = False)
        obj.class_proportions = obj.freq/obj.freq.sum()
        obj.dominant_class_percentage = obj.class_proportions.iloc[0]*100
        obj.rare_class_count = (obj.class_proportions < 0.1).sum()
        obj.gini_impurity = 1 - (obj.class_proportions**2).sum()
        obj.shannon_entropy = -(obj.class_proportions*np.log2(obj.class_proportions)).sum()
        return obj
        
        
class TextFeature(Feature):
    def __init__(self):
        super().__init__()
        self.min_length = None
        self.max_length = None
        self.mean_length = None
        self.std_length = None
        self.vocab_size= None
        self.empty_string_count = None
        self.whitespace_only_count = None

    @classmethod
    def from_series(cls, df, column_name):
        obj = super().from_series(df, column_name)
        s = df[column_name].dropna().astype(str)
        obj.min_length = s.str.len().min()
        obj.max_length = s.str.len().max()
        obj.mean_length = s.str.len().mean()
        obj.std_length = s.str.len().std()
        obj.vocab_size = (s.str.split().explode().nunique())
        obj.empty_string_count = (s == "").sum()
        obj.whitespace_only_count = s.str.fullmatch(r"\s+").sum()
        return obj
    
class BooleanFeature(Feature):
    def __init__(self):
        super().__init__()
        self.freq = None

    @classmethod
    def from_series(cls, df, column_name):
        obj = super().from_series(df, column_name)
        s = df[column_name]
        obj.freq = s.value_counts(dropna = False)
        return obj
    
class DateTimeFeature(Feature):
    def __init__(self):
        super().__init__()
        self.earliest_timestamp = None
        self.latest_timestamp = None
        self.resolution = None
        self.timezone = None
        self.time_span = None
        self.count_by_year = None
        self.count_by_month = None
        self.count_by_weekday = None
        self.count_by_hour = None
    
    @classmethod
    def from_series(cls, df, column_name):
        obj = super().from_series(df, column_name)
        s = pd.to_datetime(df[column_name],errors = 'coerce').dropna()
        obj.earliest_timestamp = s.min()
        obj.latest_timestamp = s.max()
        obj.time_span = s.max() - s.min()
        obj.timezone = s.dt.tz
        obj.resolution = pd.infer_freq(s.sort_values())
        obj.count_by_year = s.dt.year.value_counts().sort_index()
        obj.count_by_month = s.dt.month_name().value_counts().reindex(['January','February','March','April','May','June','July','August','September','October','November','December']).dropna()
        obj.count_by_weekday = s.dt.day_name().value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).dropna()
        obj.count_by_hour = s.dt.hour.value_counts().sort_index()
        return obj