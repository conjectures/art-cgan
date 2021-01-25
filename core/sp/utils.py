import time
import functools
import numpy as np
import matplotlib.pyplot as plt


def apply_to_array(data, f):
    """ Apply function to numpy array"""
    return list(map(f, data))


def timer(func):
    """
    Decorated function runtime
    """
    @functools.wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        print(f"{func.__name__!r} executed in {runtime:.4f} seconds")
        return value
    return timer_wrapper


def imshow_two_graphs(graphA, graphB):
    """ TEMP """
    plt.subplot(211)
    plt.imshow(graphA[:, :], aspect='auto', origin='lower', interpolation='none')
    plt.subplot(212)
    plt.imshow(graphB[:, :], aspect='auto', origin='lower', interpolation='none')
    plt.show()


def imshow_three_graphs(graphA, graphB, graphC):
    """ TEMP """
    plt.subplot(311)
    plt.imshow(graphA[:, :], aspect='auto', origin='lower', interpolation='none')
    plt.subplot(312)
    plt.imshow(graphB[:, :], aspect='auto', origin='lower', interpolation='none')
    plt.subplot(313)
    plt.imshow(graphC[:, :], aspect='auto', origin='lower', interpolation='none')
    plt.show()


def plot_features(features):
    """ Plot values from dictionary of features; """
    try:
        keynum = len(features)
        x = next(iter(features.values())).size
        x = np.arange(x)
        # print(f"{x=}")
        color = iter(plt.cm.rainbow(np.linspace(0, 1, keynum)))
        for key, value in features.items():
            plt.plot(x, value, c=next(color), label=key)

        plt.legend()
        plt.show()
    except AttributeError:
        print("Please input dictionary")



