from base import Feature

class NumericalFeature(Feature):
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
        