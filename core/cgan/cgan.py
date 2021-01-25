#from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


import os
import matplotlib.pyplot as plt
import sys, time
import numpy as np
import utils as ut


class CGAN():
    def __init__(self, classnum, gweights=None, dweights=None):
        # Input shape
        self.img_rows = 720
        self.img_cols = 720
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = classnum
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Load weights if provided
        if dweights:
            self.discriminator.load_weights(dweigths)

        self.discriminator.compile(
                loss='binary_crossentropy',
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
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #self.combined.save(ut.check_filename("models", "dcgan_model", ext="h5"))
    def build_generator(self, model=None):
        if model is None:
            # TODO: import default model
            pass

        print("Generator: ")
        model.summary()
        #Embed(combine) noise and labels into model input
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int8')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)

    def build_discriminator(self, model=None):

        if model is None:
            # TODO: import default discr
            pass

        print("Discriminator: ")
        model = Model(inputs=[image_input, label_input], outputs = [main_output])
        model.summary()
        return model

    def train(self, train_data, epochs, batch_size=16, save_interval=50):
        # Load the dataset
        #(X_train,_), (_, _) = mnist.load_data()
        #X_train = np.load("./data/train_data_p00.npy").astype("float16")
        print("Loading data")
        npzfile =  np.load(train_data)
        X_train = npzfile['arr_0'].astype("float16")
        y_train = npzfile['arr_1'].astype("float16")

        # print("Begin training?")
        # input()
        #transpose
        y_train.reshape(-1,1)
        # Rescale -1 to 1
        #X_train = X_train /127.5 - 1.
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
                self.save_imgs(epoch)
        #save weights after training

        print("Saving weights..")
        self.generator.save_weights(ut.check_filename("models", "gweights", ext="h5"))
        self.discriminator.save_weights(ut.check_filename("models", "dweights", ext="h5"))


    def save_imgs(self, epoch):
        r, c = 6, 8
        print("Saving generated images")
        #print("class number: {}".format(self.num_classes))

        noise = np.random.normal(0, 1, (r*c, self.latent_dim))

        labels = np.arange(r*c).reshape(-1,1)
        gen_imgs = self.generator.predict([noise,labels])
        print(gen_imgs.shape)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        outfile = ut.check_filename("results","gen_{:04d}".format(epoch),ext="png")
        fig.savefig(outfile)
        plt.close()

