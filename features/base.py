class Feature():
    def __init__(self):
        self.column_name = None
        self.pandas_dtype = None
        self.value_count = 0
        self.missing_count = 0
        self.missing_percentage = 0.0
        self.unique_count = 0
        self.is_constant = False
        self.is_binary = False
        self.cardinality = None
    @classmethod
    def from_series(cls,df,column_name):
        obj = cls()
        obj.column_name = column_name
        obj.value_count = df[column_name].size
        obj.pandas_dtype = df[column_name].dtype
        obj.missing_count = df[column_name].isna().sum()
        obj.missing_percentage = (obj.missing_count/obj.value_count)*100
        obj.unique_count = df[column_name].nunique(dropna = True)
        obj.is_constant = True if obj.unique_count == 1 else False
        obj.is_binary = True if obj.unique_count == 2 else False
        if obj.unique_count > 50 and obj.unique_count/obj.value_count > 0.05:
            obj.cardinality = 'High'
        if obj.unique_count < 20:
            obj.cardinality = 'Low'
        return obj
    
        