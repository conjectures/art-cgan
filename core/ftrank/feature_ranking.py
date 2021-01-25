"""
    Functions used to transform features into 'ranking'
"""
import numpy as np

__all__ = ('rank_features')


def rank_features(features):
    """ Gather features and feed into weighting function """
    keys_order = features.keys()
    # make into array
    # ft_array = np.stack(tuple([ft for ft in features.values()]))
    rank = weight_function(
                    loudness=features['loudness'],
                    peak_num=features['peak_num'],
                    means=features['means'],
                    means_trend=features['means_trend'],
                    peaks_trend=features['peaks_trend'],
                    )
    return rank


def weight_function(*, loudness, peak_num, mean, mean_trend, peaks_trend):
    """ Combine features with simple weighted addition """
    return 0.333*(peak_num + mean - loudness) + mean_trend + peaks_trend
