"""
    Functions used to transform features into class
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


def weight_function(*, loudness, peak_num, means, means_trend, peaks_trend):
    """ Combine features with simple weighted addition """
    return 0.333*(peak_num + means - loudness) + means_trend + peaks_trend


def classify_rank(rank):
    """ Get rank and return class (int) - TODO fix for more classes """
    return np.where(rank>0, 1, 0)

