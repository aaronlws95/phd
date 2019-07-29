#https://arxiv.org/abs/1406.2661
import os
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from keras.utils import plot_model

import matplotlib.pyplot as plt

import sys

import numpy as np

import argparse

import h5py

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Arguments
parser = argparse.ArgumentParser(description='GAN')
#PHASE
parser.add_argument('--phase', dest='phase', default='train', type=str, help='[train, test]')
#SAVE DIR
parser.add_argument('--save_dir', dest='save_dir', default='default', type=str, help='save directory')
#GPU_FRACTION
parser.add_argument('--gpu_fraction', dest='gpu_fraction', default=1, type=float, help='GPU resource allocation [0, 1]')
#GPU
parser.add_argument('--gpu', dest='gpu', default='0', type=str, help='GPU model')
#EPOCHS
parser.add_argument('--epochs', dest='epochs', default=30000, type=int, help='number of epochs to run training')
#BATCH_SIZE
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, help='training batch size')
#SAMPLE_INTERVAL
parser.add_argument('--sample_interval', dest='sample_interval', default=200, type=int, help='number of epochs to sample images')
#SAVE_INTERVAL
parser.add_argument('--save_interval', dest='save_interval', default=5000, type=int, help='number of epochs before saving model')
#MODEL_VERSION
#If model not 0 then load model
parser.add_argument('--model_version', dest='model_version', default='0', type=str, help='model version to load')
args = parser.parse_args()

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def get_session(gpu_fraction, show_log=False):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=show_log))
KTF.set_session(get_session(gpu_fraction = args.gpu_fraction))

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # ---------------------
        #        PATHS
        # ---------------------
        dirname = '/media/aaron/DATA/ubuntu/GAN'

        self.save_dir = os.path.join(dirname, args.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model_path = os.path.join(self.save_dir, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.com_model_path = os.path.join(self.model_path, 'combined')
        if not os.path.exists(self.com_model_path):
            os.makedirs(self.com_model_path)

        self.gen_model_path = os.path.join(self.model_path, 'generator')
        if not os.path.exists(self.gen_model_path):
            os.makedirs(self.gen_model_path)

        self.disc_model_path = os.path.join(self.model_path, 'discriminator')
        if not os.path.exists(self.disc_model_path):
            os.makedirs(self.disc_model_path)

        self.loss_path = os.path.join(self.save_dir, 'loss')
        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)

        self.image_path = os.path.join(self.save_dir, 'images')
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

        self.itmd_path = os.path.join(self.save_dir, 'itmd')
        if not os.path.exists(self.itmd_path):
            os.makedirs(self.itmd_path)

        self.log_path = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        with open(os.path.join(self.disc_model_path, 'disc_model_arch.json'), 'w') as f:
            f.write(self.discriminator.to_json())

        # Build the generator
        self.generator = self.build_generator()

        with open(os.path.join(self.gen_model_path, 'gen_model_arch.json'), 'w') as f:
            f.write(self.generator.to_json())

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,), name='Noise')
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity, name='Combined')
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        plot_model(self.combined, to_file=os.path.join(self.model_path, 'combined.png'), show_shapes=True, show_layer_names=True)

        with open(os.path.join(self.com_model_path, 'com_model_arch.json'), 'w') as f:
            f.write(self.combined.to_json())


    def build_generator(self):

        noise = Input(shape=(self.latent_dim,), name='Noise')

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim, name='dense_gen_1'))
        model.add(LeakyReLU(alpha=0.2, name='lrelu_gen_1'))
        model.add(BatchNormalization(momentum=0.8, name='bnorm_gen_1'))
        model.add(Dense(512, name='dense_gen_2'))
        model.add(LeakyReLU(alpha=0.2, name='lrelu_gen_2'))
        model.add(BatchNormalization(momentum=0.8, name='bnorm_gen_2'))
        model.add(Dense(1024, name='dense_gen_3'))
        model.add(LeakyReLU(alpha=0.2, name='lrelu_gen_3'))
        model.add(BatchNormalization(momentum=0.8, name='bnorm_gen_3'))
        model.add(Dense(np.prod(self.img_shape), activation='tanh', name='dense_gen_4'))
        model.add(Reshape(self.img_shape, name='reshape_1'))
        model.summary()

        img = model(noise)

        plot_model(model, to_file=os.path.join(self.model_path, 'generator.png'), show_shapes=True, show_layer_names=True)

        return Model(noise, img, name='Generator')

    def build_discriminator(self):

        img = Input(shape=self.img_shape, name='Image')

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape, name='flat_disc_1'))
        model.add(Dense(512, name='dense_disc_1'))
        model.add(LeakyReLU(alpha=0.2, name='lrelu_disc_1'))
        model.add(Dense(256, name='dense_disc_2'))
        model.add(LeakyReLU(alpha=0.2, name='lrelu_disc_2'))
        model.add(Dense(1, activation='sigmoid', name='dense_disc_3'))
        model.summary()

        validity = model(img)

        plot_model(model, to_file=os.path.join(self.model_path, 'discriminator.png'), show_shapes=True, show_layer_names=True)

        return Model(img, validity, name='Discriminator')

    def load_models(self):
        # probably only need to load combined
        self.combined.load_weights(os.path.join(self.com_model_path, 'com_model_weights_%s.h5') % args.model_version)
        self.generator.load_weights(os.path.join(self.gen_model_path, 'gen_model_weights_%s.h5') % args.model_version)
        self.discriminator.load_weights(os.path.join(self.disc_model_path, 'disc_model_weights_%s.h5') % args.model_version)

    def save_weights(self, epoch):
        self.generator.save_weights(os.path.join(self.gen_model_path, 'gen_model_weights_%s.h5' % epoch))

        self.discriminator.save_weights(os.path.join(self.disc_model_path, 'disc_model_weights_%s.h5' % epoch))

        self.combined.save_weights(os.path.join(self.com_model_path, 'com_model_weights_%s.h5' % epoch))

    def train(self, epochs, batch_size=128, sample_interval=50, save_interval=5000):

        # Load model
        if args.model_version != '0':
            self.load_models()

            start_epoch = int(args.model_version)+1
            h5f = h5py.File(os.path.join(self.loss_path, 'loss_%s.h5' % args.model_version), 'r')
            d_loss_arr = h5f['d_loss'][:]
            g_loss_arr = h5f['g_loss'][:]
            h5f.close()
        else:
            start_epoch = 0
            d_loss_arr = []
            g_loss_arr = []

        # Load the dataset
        # X_train.shape: (60000, 28, 28)
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.

        # X_train.shape: (60000, 28, 28, 1)
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Set callback for TensorBoard
        callback = TensorBoard(self.log_path, batch_size=batch_size)
        callback.set_model(self.combined)

        for epoch in range(epochs):

            cur_epoch = int(epoch + start_epoch)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            if cur_epoch % sample_interval == 0:
                self.save_discr_itmd(noise, gen_imgs, imgs, cur_epoch)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # THESE ARE EQUAL
            # print(self.discriminator.evaluate(self.generator.predict(noise), valid))
            # print(self.combined.evaluate(noise, valid))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            self.write_log(callback, ['g_loss'], [g_loss], cur_epoch)
            self.write_log(callback, ['d_loss'], [d_loss[0]], cur_epoch)

            # Plot the progress
            print ('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (cur_epoch, d_loss[0], 100*d_loss[1], g_loss))

            d_loss_arr = np.append(d_loss_arr, d_loss[0])
            g_loss_arr = np.append(g_loss_arr, g_loss)

            # If at save interval => save model
            if cur_epoch % save_interval == 0:
                self.save_weights(cur_epoch)

            # If at sample interval => save generated image samples
            if cur_epoch % sample_interval == 0:
                self.sample_images(cur_epoch)

        h5f = h5py.File(os.path.join(self.loss_path, 'loss_%s.h5' % cur_epoch), 'w')
        h5f.create_dataset('d_loss', data=d_loss_arr)
        h5f.create_dataset('g_loss', data=g_loss_arr)
        h5f.close()

        self.save_loss_plt(d_loss_arr, g_loss_arr, cur_epoch)
        self.save_weights(cur_epoch)

    def test(self):
        # Create directory for loading model
        self.generator.load_weights(os.path.join(self.gen_model_path, 'gen_model_weights_%s.h5') % args.model_version)

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.image_path, '%s_gen.png' % args.model_version))
        plt.close()

    def save_discr_itmd(self, noise, gen_imgs, real_imgs, epoch):
            # fig = plt.figure()
            # plt.imshow(noise, cmap='gray')
            # fig.savefig(os.path.join(self.itmd_path, '%d_noise.png' % epoch))
            # plt.close()

            validity = self.discriminator.predict(gen_imgs)

            r = 8
            c = 4
            fig, axs = plt.subplots(r, c)
            fig.tight_layout()
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].set_title('%.4f' % validity[cnt], fontsize=8)
                    axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(self.itmd_path, '%d_gen_img.png' % epoch))
            plt.close()

            validity = self.discriminator.predict(real_imgs)

            r = 8
            c = 4
            fig, axs = plt.subplots(r, c)
            fig.tight_layout()
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].set_title('%.4f' % validity[cnt], fontsize=8)
                    axs[i,j].imshow(real_imgs[cnt, :, :, 0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(os.path.join(self.itmd_path, '%d_real_img.png' % epoch))
            plt.close()

    def save_loss_plt(self, d_loss_arr, g_loss_arr, cur_epoch):
        fig, sps = plt.subplots(2, sharex=True)
        fig.set_size_inches(8, 6)

        big_subplot = fig.add_subplot(111, frameon=False)
        plt.xlabel('Epoch')
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)

        sps[0].plot(d_loss_arr, c='b')
        sps[0].set_title('Discriminator Loss')
        sps[1].plot(g_loss_arr, c='g')
        sps[1].set_title('Generator Loss')

        fig.savefig(os.path.join(self.loss_path, 'loss_%s.png' % cur_epoch))
        plt.close()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.image_path, 'sample_%d.png' % epoch))
        plt.close()

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

gan = GAN()

if args.phase == 'train':
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=args.sample_interval, save_interval=args.save_interval)
if args.phase == 'test':
    gan.test()

