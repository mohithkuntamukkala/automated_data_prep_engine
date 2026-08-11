import pandas as pd
import numpy as np

def pearson_coefficient(df,col1,col2):
    s1,s2 = df[col1],df[col2]
    cov = s1.cov(s2)
    return cov/(s1.std()*s2.std())

def spearman_coefficient(df, col1, col2):
    s1 = df[col1].rank()
    s2 = df[col2].rank()
    cov = s1.cov(s2)
    return cov/(s1.std()*s2.std())


def eta_squared(df, num_col, cat_col):
    grand_mean = df[num_col].mean()
    ss_between = (
        df.groupby(cat_col)[num_col]
          .apply(lambda x: len(x)*(x.mean()-grand_mean)**2)
          .sum()
    )
    ss_total = ((df[num_col]-grand_mean)**2).sum()
    return ss_between / ss_total


def point_biserial(df, bool_col, num_col):
    x = df[num_col]
    b = df[bool_col].astype(bool)
    mean_true = x[b].mean()
    mean_false = x[~b].mean()
    p = b.mean()
    q = 1 - p
    return ((mean_true-mean_false)/x.std())*np.sqrt(p*q)


def cramers_v(df, col1, col2):
    observed = pd.crosstab(df[col1], df[col2])
    row_totals = observed.sum(axis=1).to_numpy()[:, None]
    col_totals = observed.sum(axis=0).to_numpy()[None, :]
    n = observed.to_numpy().sum()
    expected = (row_totals @ col_totals) / n
    chi2 = (((observed.to_numpy() - expected) ** 2) / expected).sum()
    r, c = observed.shape
    return np.sqrt((chi2 / n) / min(r - 1, c - 1))