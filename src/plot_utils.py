import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_groupby(df, stats, by='id', show_labels=False):
    groupby_obj = df.groupby(by=by)
    for cols in df.columns[2:]:
        lables = cols if show_labels else None
        if stats == 'min':
            plt.plot(groupby_obj.min()[cols], label=lables)
        elif stats == 'max':
            plt.plot(groupby_obj.max()[cols], label=lables)
        elif stats == 'mean':
            plt.plot(groupby_obj.mean()[cols], label=lables)
        elif stats == 'std':
            plt.plot(groupby_obj.std()[cols], label=lables)
        else:
            raise ValueErrror("Unknown stats type")
    plt.title(stats)
    if show_labels:
        plt.legend()


def plot_feature(df, feature, timeseries=None, show_legend=False, title=None):
    timeseries_length = [df[df['id'] == i]['cycle'].max() for i in df.id.unique()]
    iter_ = df.id.unique() if timeseries is None else timeseries
    for i in iter_:
        plt.plot(np.arange(max(timeseries_length), 0, -1),
                 np.pad(df[feature][df.id == i], (max(timeseries_length) - df[df['id'] == i]['cycle'].max(), 0),
                        constant_values=np.nan),
                 label=i)
    plt.gca().invert_xaxis()
    plt.xlabel('Cycle to failure')
    plt.title(title)
    if show_legend:
        plt.legend()


def crossplot_pred(pred, true):
    total_min = min(np.nanmin(pred), np.nanmin(true))
    total_max = max(np.nanmax(pred), np.nanmax(true))
    plt.scatter(np.array(pred), np.array(true), s=1)
    plt.plot((total_min, total_max), (total_min, total_max), c='r')
    plt.xlim(total_min, total_max)
    plt.ylim(total_min, total_max)
    plt.xlabel("Predict")
    plt.ylabel("True")
    plt.title(f"MAE: {np.nanmean(np.abs(np.array(pred) - np.array(true))):.3f}")
