import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU
import datetime
import time
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm

#implementation of adapted pix2pix cGAN
#compare tensorflow tutorial https://www.tensorflow.org/tutorials/generative/pix


class HiCGAN():
    def __init__(self, log_dir: str): 
        super().__init__()

        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 3
        self.INPUT_SIZE = 64
        self.TARGET_CHANNELS = 1
        self.LAMBDA = 100
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.log_dir=log_dir
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)


    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(BatchNormalization())
        result.add(LeakyReLU())
        return result

    @staticmethod
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(BatchNormalization())
        if apply_dropout:
            result.add(Dropout(0.5))
        result.add(ReLU())
        return result


    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[self.INPUT_SIZE,self.INPUT_SIZE,self.INPUT_CHANNELS], name="factorData")

        down_stack = [
            #HiCGAN.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            #HiCGAN.downsample(128, 4), # (bs, 64, 64, 128)
            HiCGAN.downsample(256, 4), # (bs, 32, 32, 256)
            HiCGAN.downsample(512, 4), # (bs, 16, 16, 512)
            HiCGAN.downsample(512, 4), # (bs, 8, 8, 512)
            HiCGAN.downsample(512, 4), # (bs, 4, 4, 512)
            HiCGAN.downsample(512, 4), # (bs, 2, 2, 512)
            HiCGAN.downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            HiCGAN.upsample(512, 4), # (bs, 16, 16, 1024)
            HiCGAN.upsample(256, 4), # (bs, 32, 32, 512)
            #HiCGAN.upsample(128, 4), # (bs, 64, 64, 256)
            #HiCGAN.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.INPUT_SIZE, self.INPUT_SIZE, self.INPUT_CHANNELS], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.INPUT_SIZE, self.INPUT_SIZE, self.TARGET_CHANNELS], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 80 80, 3+1=4)

        down1 = HiCGAN.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = HiCGAN.downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = HiCGAN.downsample(256, 4)(down2) # (bs, 32, 32, 256)
        conv = Conv2D(512, 4, strides=1,
                        kernel_initializer=initializer,
                        use_bias=False)(down3) # (bs, 31, 31, 512)
        batchnorm1 = BatchNormalization()(conv)
        leaky_relu = LeakyReLU()(batchnorm1)
        last = Conv2D(1, 4, strides=1,
                        kernel_initializer=initializer)(leaky_relu) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        with self.summary_writer.as_default(): #pylint: disable=not-context-manager
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

    def generate_images(self, model, test_input, tar, epoch: int):
        prediction = model(test_input, training=True)
        figname = "testpred_epoch_{:05d}.png".format(epoch+1)
        figname = os.path.join(self.log_dir, figname)
        display_list = [test_input["factorData"][0], tar["out_matrixData"][0], prediction[0]]
        titleList = ['Input Image', 'Ground Truth', 'Predicted Image']

        fig1, axs1 = plt.subplots(1,len(display_list), figsize=(15,15))
        for i in range(len(display_list)):
            axs1[i].imshow(display_list[i] * 0.5 + 0.5)
            axs1[i].set_title(titleList[i])
        fig1.savefig(figname)

    def fit(self, train_ds, epochs, test_ds, steps_per_epoch: int):
        for epoch in range(epochs):
            for example_input, example_target in test_ds.take(1):
                self.generate_images(self.generator, example_input, example_target, epoch)

            # Train
            train_pbar = tqdm(train_ds.enumerate(), total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            for _, (input_image, target) in train_pbar:
                self.train_step(input_image["factorData"], target["out_matrixData"], epoch)

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    def plotModels(self, outputpath: str, figuretype: str):
        generatorPlotName = "generatorModel.{:s}".format(figuretype)
        generatorPlotName = os.path.join(outputpath, generatorPlotName)
        discriminatorPlotName = "discriminatorModel.{:s}".format(figuretype)
        discriminatorPlotName = os.path.join(outputpath, discriminatorPlotName)
        tf.keras.utils.plot_model(self.generator, show_shapes=True, to_file=generatorPlotName)
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, to_file=discriminatorPlotName)