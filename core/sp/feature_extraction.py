"""
Functions for extracting features from audio sample
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from essentia.standard import MonoLoader
# Local imports
from .signal_processing import get_spectrogram, equalize_spectrum, group_avg
# from .utils import timer

# Import support
__all__ = (
        'get_features', 'get_peaks', 'get_peak_means', 'get_peak_num',
        )


def get_features(filename, *, samplerate=44100, features_per_second=2):
    """
        Process audio from filename and return dictionary with features
    """
    # TODO: load sample from different function
    loader = MonoLoader(filename=filename)
    audio = loader()

    samples = audio.shape[0]
    seconds = samples/samplerate
    minutes = seconds/60

    spectrogram = get_spectrogram(audio)
    log_spectrogram = np.log(spectrogram)
    spectr_eq = equalize_spectrum(log_spectrogram)

    frame_num = spectrogram.shape[1]
    frames_per_second = frame_num // seconds
    frames_per_featureset = int(frames_per_second // features_per_second)

    spectr_combined = group_avg(spectr_eq.T, frames_per_featureset).T

    featureset_num = frame_num // frames_per_featureset

    # Find most prominent peaks
    threshold = 255*0.7         # TODO: get as parameter
    peaks = get_peaks(spectr_combined.T, threshold=threshold)
    features = {}
    features['peak_num'] = get_peak_num(peaks).astype(float)
    features['means'] = get_peak_means(peaks)
    features['loudness'] = np.sum(spectr_combined, axis=0)

    # Normalise features
    for key, ft in features.items():
        features[key] = ft / ft.max()

    # rate of change of peak num
    features['peaks_trend'] = np.gradient(features['peak_num'])
    features['means_trend'] = np.gradient(features['means'])

    return features


def get_peaks(data, *, threshold=1):
    """ Return list with peak indeces above threshold for a numpy array """
    # Create temporary function - find peaks above threshold and get first output
    f = lambda x: find_peaks(x, threshold)[0]
    # Apply temp function on numpy rows
    peaks = list(map(f, data))
    return peaks


def get_peak_means(data):
    """ Get list of peak indices and return their mean """
    # Apply on list components and return result as numpy array
    return np.asarray(list(map(np.mean, data)))


def get_peak_num(data):
    """ Get list of peak indices and return their mean """
    # Apply on list components and return result as numpy array
    return np.asarray(list(map(np.size, data)))


def get_peaks_alt(data, *, threshold=1):
    """
        Alt getPeaks function (depr)
    """
    peaks_idx = []
    feature_peaks_mean = []
    for count, row in enumerate(data.T):
        idx, _ = find_peaks(row, threshold)
        peaks_idx.append(idx)
        feature_peaks_mean.append(np.mean(idx))
    feature_peaks_mean = np.asarray(feature_peaks_mean)

    return {
            'peaks': peaks_idx,
            'means': feature_peaks_mean,
            }



