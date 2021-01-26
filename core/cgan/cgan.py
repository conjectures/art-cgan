#from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, GaussianNoise
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam


import os
import matplotlib.pyplot as plt
import sys, time
import numpy as np

import cgan.utils as ut
import cgan.cgan_models as cgm


class CGAN():
    def __init__(self, num_classes=10, *, gweights=None, dweights=None, result_dir=None):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = 2048
        self.result_dir = result_dir

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Load weights if provided
        if dweights:
            self.discriminator.load_weights(dweigths)

        self.discriminator.compile(
                loss=['binary_crossentropy'],
                optimizer=optimizer,
                metrics=['accuracy'],
                )

        # Build the generator
        self.generator = self.build_generator()
        # Load weights if given as arg
        if (gweights):
            self.generator.load_weights(gweights)

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        # Label derived from feature extraction rank
        label = Input(shape=(1,))
        # Image produced by generator
        img = self.generator([noise, label])
        # For combined model only generator is trained
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity and label
        valid = self.discriminator([img, label])
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)
        #self.combined.save(ut.check_filename("models", "dcgan_model", ext="h5"))

    def build_generator(self, model=None):
        if model is None:
            pass
            # model = cgm.generator_model_A(input_dimension=self.latent_dim, channels=self.channels)

        #Embed(combine) noise and labels into model input
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int8')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])

        x = Dense(2048)(model_input)
        x = Reshape((2, 2, 512))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        # 2
        x = Conv2DTranspose(256, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        # 4
        x = Conv2DTranspose(128, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        # 8
        x = Conv2DTranspose(64, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        # 16

        x = Conv2DTranspose(32, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        # 32
        x = Conv2DTranspose(32, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        #64
        x = Conv2DTranspose(32, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        #128
        x = Conv2DTranspose(16, (5, 5,), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.1)(x)
        #256
        x = Conv2DTranspose(3, (5, 5,), padding='same', strides=2)(x)
        img = Activation('tanh')(x)
        # img = model(model_input)

        print("Generator: ")
        model = Model([noise, label], img)
        #model.summary()

        return model

    def build_discriminator(self, model=None):

        if model is None:
            pass
            # Import default model
            # model = cgm.discriminator_model_A(input_shape=self.img_shape)
        img = Input(shape=(self.img_shape))
        #512
        x = GaussianNoise(stddev=0.1)(img)
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #256

        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #128

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #64

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #32

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #16

        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #8

        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #4

        x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        #2

        # img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int8')

        #label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        flat_img = Flatten()(x)

        model_input = multiply([flat_img, label_embedding])

        nn = Dropout(0.3)(model_input)
        validity = Dense(1, activation='sigmoid')(nn)

        model = Model(inputs=[img, label], outputs=validity)

        print("Discriminator: ")
        # model.summary()
        return model

    def train(self, train_data, epochs, batch_size=8, save_interval=50):
        # Load the dataset
        #(X_train,_), (_, _) = mnist.load_data()
        #X_train = np.load("./data/train_data_p00.npy").astype("float16")
        print("Loading data")
        # npzfile =  np.load(train_data)
        # X_train = npzfile['arr_0'].astype("float16")
        # y_train = npzfile['arr_1'].astype("float16")
        X_train, y_train = train_data

        # print("Begin training?")
        # input()
        #transpose
        y_train.reshape(-1,1)
        # Rescale -1 to 1
        #X_train = X_train /127.5 - 1.
        print(f'{X_train.shape=}')
        X_train = (X_train + 0.5) / 127.5

        #X_train = np.expand_dims(X_train, axis=4)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            iter_start_time = time.time()
            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels])
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            #train generator on random labels
            rand_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, rand_labels], valid)
            #train generator twice
            g_loss = self.combined.train_on_batch([noise, rand_labels], valid)
            iter_end_time = time.time()
            per_iter_time = iter_end_time - iter_start_time
            # Plot the progress
            if not (epoch % 100):
                print("{} [D loss: {:.2f}, acc.: {:.2f}%] [G loss: {:.2f}] - t: {:.2f}".format(epoch, d_loss[0], 100*d_loss[1], g_loss, per_iter_time ))
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, self.result_dir)
        #save weights after training

        print("Saving weights..")
        # self.generator.save_weights(ut.check_filename("models", "gweights", ext="h5"))
        # self.discriminator.save_weights(ut.check_filename("models", "dweights", ext="h5"))


    def save_imgs(self, epoch, result_dir):
        # r, c = 6, 8
        label_num = self.num_classes
        print("Saving generated images")
        #print("class number: {}".format(self.num_classes))

        noise = np.random.normal(0, 1, (label_num, self.latent_dim,))

        labels = np.arange(label_num).reshape(-1,1)
        gen_imgs = self.generator.predict([noise,labels])
        print(gen_imgs.shape)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(2)

        # for i in range(r):
        axs[0].imshow(gen_imgs[0, :, :, :])
        axs[1].imshow(gen_imgs[1, :, :, :])
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,:])
        #         axs[i,j].axis('off')
        #         cnt += 1
        outfile = ut.check_filename("gen_{:04d}".format(epoch), result_dir, ext="png")
        fig.savefig(outfile)
        plt.close()

