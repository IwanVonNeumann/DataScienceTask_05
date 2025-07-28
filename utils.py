import numpy as np
import pandas as pd

from pandas.io.formats.style import Styler


def thousand_separators(x):
    return '{:,}'.format(x).replace(',', ' ')


def get_target_pivot(df: pd.DataFrame, col: str, target_col: str = 'ltv_30') -> Styler:
    grouped_df = df.groupby(col, observed=True)

    summary = grouped_df[target_col].agg([
        (target_col, 'mean'),
        ('n_users', 'count')
    ]).reset_index()

    summary['%_users'] = summary['n_users'] / len(df)

    target_by_row = summary[target_col] * summary['n_users']

    summary['%_rev'] = target_by_row / target_by_row.sum()

    summary = summary.set_index(col)

    if df[col].dtype == 'object':
        summary = summary.sort_values(by=target_col, ascending=False)
    else:
        summary = summary.sort_index()

    if summary.index.dtype == 'float':
        summary.index = summary.index.map('{:.3f}'.format)

    cols_format = {
        target_col: '{:.2f}'.format,
        'n_users': thousand_separators,
        '%_users': '{:.1%}'.format,
        '%_rev': '{:.1%}'.format
    }

    return summary.style.format(cols_format).background_gradient(axis=0, subset=[target_col], cmap='RdYlGn')


def merge_rare(series: pd.Series, threshold: float = 0.01, default: str = 'OTHER') -> np.ndarray:
    abs_f = series.value_counts(dropna=False) / len(series)

    top_values = abs_f[abs_f >= threshold].index.tolist()

    return np.where(series.isin(top_values), series, default)


def enumerate_float(x: pd.Series, q: int = 10, process_zeros: bool = False) -> pd.Series:
    if process_zeros:
        x_0 = x[x == 0]
        x = x[x != 0]
    else:
        x_0 = pd.Series()

    t = pd.qcut(x, q=q)

    bins = t.value_counts(dropna=False).index.sort_values().tolist()

    d = {b: b.left for i, b in enumerate(bins, start=1)}

    t = t.map(d)

    if process_zeros:
        t = pd.concat(objs=[x_0, t])

    if np.nan in d:
        return t.cat.add_categories(0).fillna(0)
    else:
        return t


def get_gain_ranking(model):
    importance_df = pd.DataFrame()

    importance_df['gain'] = model.get_booster().get_score(importance_type='gain')

    importance_df['gain'] = importance_df['gain'] / importance_df['gain'].sum()

    return importance_df.sort_values(by='gain', ascending=False)
