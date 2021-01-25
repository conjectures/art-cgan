
import numpy as np
import os
import matplotlib.pyplot as plt

from sp.feature_extraction import get_features
from sp.utils import plot_features
from ftrank.feature_ranking import collect_features
from defs import MEDIA_DIR, DATA_DIR
from cgan import cgan


def main():
    # filename = os.path.join(os.path.join(MEDIA_DIR, 'djrum-blue_violet.wav'))
    filename = os.path.join(os.path.join(MEDIA_DIR, 'dubstep.wav'))

    features = get_features(filename)
    plot_features(features)

    rank = rank_features(features)
    print(f"{type(rank)=}, {len(rank)=}, \n{rank=}")
    # plot
    plt.plot(rank)
    plt.show()

    # Initialize cgan
    cgan = cgan.CGAN()

    # Train


if __name__ == '__main__': main()



#
