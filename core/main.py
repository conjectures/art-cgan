
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# from sp.feature_extraction import get_features
from sp.utils import plot_features
from ftrank.feature_ranking import rank_features, classify_rank
from defs import MEDIA_DIR, DATA_DIR, RESULTS_DIR
from cgan import cgan, cgan_models, utils


def main():
    # Define vars
    epochs = 100
    batch_size = 1

    # filename = os.path.join(os.path.join(MEDIA_DIR, 'djrum-blue_violet.wav'))
    # filename = os.path.join(os.path.join(MEDIA_DIR, 'dubstep.wav'))

    # features = get_features(filename)
    # plot_features(features)

    # rank = rank_features(features)
    # print(f"{type(rank)=}, {len(rank)=}, \n{rank=}")
    # # plot
    # plt.plot(rank)
    # plt.show()
    # print(rank)
    rank = np.asarray([
        -3.21148580e-01, -6.41094057e-01, -2.08017051e-01,
        2.77437942e-01, 4.48627728e-02, 3.22995367e-01,
        4.23711706e-01,  2.76627046e-01, -9.42335690e-01,
        -1.61024928e-01,  3.37734902e-01,  5.01757204e-04,
        2.02171142e-01, -1.96823792e-01])

    input = classify_rank(rank)

    # Initialize cgan
    # train_files = os.path.join(DATA_DIR, 'unpacked')
    # num_classes = len(train_files)
    print(DATA_DIR)
    print()
    data = utils.get_training_data(DATA_DIR)
    num_classes = data['num_classes']
    train_data = data['train_data']

    gan = cgan.CGAN(num_classes=2, result_dir=RESULTS_DIR)

    start_time = time.perf_counter()
    gan.train(train_data, epochs=epochs, batch_size=batch_size, save_interval=10)

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time}")



    # Train


if __name__ == '__main__':
    main()
