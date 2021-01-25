"""
    SIGNAL PROCESSING
"""
# import essentia
from essentia.standard import FrameGenerator, Spectrum, Windowing
import numpy as np

from .utils import timer


@timer
def get_spectrogram(audio_data):

    spectrogram = []
    spectrum = Spectrum()
    w = Windowing(type='hann')

    spectrogram = np.array(list(map(
                            lambda x: spectrum(w(x)),
                            list(FrameGenerator(
                                audio_data,
                                frameSize=2048,
                                hopSize=1024,
                                startFromZero=True))
                            ))).T

    spectrogram += np.min(spectrogram[np.nonzero(spectrogram)])
    return spectrogram


@timer
def equalize_spectrum(data):
    # Move minimum value to zero
    data -= data.min()
    # Scale max to 255
    data *= (255.0/data.max())
    # Round to int values
    data = np.rint(data)
    # Create histogram and bins
    hist, bins = np.histogram(data.flatten(), 256, [0, 256])
    # Find Cumulative Distr Function
    cdf = hist.cumsum()
    # Normalise cdf
    cdf_normalized = cdf * hist.max() / cdf.max()
    # Mask nonzero values ( ingore zeros )
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Equalise cdf values
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # Return equalized data values according to cdf mapping
    return cdf[data.astype('uint8')]


@timer
def group_avg(data, N=2):
    result = np.cumsum(data, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result
