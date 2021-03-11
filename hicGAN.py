import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, LeakyReLU, Conv2DTranspose, Dropout, ReLU, Flatten, Dense
import numpy as np
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm
import utils

#implementation of adapted pix2pix cGAN
#compare tensorflow tutorial https://www.tensorflow.org/tutorials/generative/pix


class HiCGAN():
    def __init__(self, log_dir: str, 
                    lambda_pixel: float = 100,
                    lambda_disc: float = 0.5, 
                    loss_type_pixel: str = "L1",
                    tv_weight: float = 1e-10,
                    input_size: int = 256,
                    plot_frequency: int = 20,
                    plot_type: str = "png",
                    learning_rate: float = 2e-5,
                    adam_beta_1: float = 0.5,
                    pretrained_model_path: str = "",
                    embedding_model_type: str = "CNN"): 
        super().__init__()

        self.OUTPUT_CHANNELS = 1
        self.INPUT_CHANNELS = 1
        self.INPUT_SIZE = 256
        if input_size in [64,128,256]:
            self.INPUT_SIZE = input_size
        self.NR_FACTORS = 14
        self.lambda_pixel = lambda_pixel
        self.lambda_disc = lambda_disc
        self.tv_loss_Weight = tv_weight
        self.loss_type_pixel = loss_type_pixel
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta_1, name="Adam_Generator")
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta_1, name="Adam_Discriminator")
        #choose the desired embedding network: DNN (like Farre et al.) or CNN
        if embedding_model_type not in ["DNN", "CNN", "mixed"]:
            msg = "Embedding {:s} not supported".format(embedding_model_type)
            raise NotImplementedError(msg)
        if embedding_model_type == "DNN":
            self.generator_embedding = self.dnn_embedding(pretrained_model_path=pretrained_model_path)
            self.discriminator_embedding = self.dnn_embedding(pretrained_model_path=pretrained_model_path)
        elif embedding_model_type == "mixed":
            self.generator_embedding = self.dnn_embedding(pretrained_model_path=pretrained_model_path)
            self.discriminator_embedding = self.cnn_embedding()
        else:
            self.generator_embedding = self.cnn_embedding()
            self.discriminator_embedding = self.cnn_embedding()         
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.log_dir=log_dir
        
        self.checkpoint_dir = os.path.join(self.log_dir, 'training_checkpoints')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                    discriminator_optimizer=self.discriminator_optimizer,
                                    generator=self.generator,
                                    discriminator=self.discriminator)
        self.plot_type = plot_type
        if self.plot_type not in ["png", "pdf", "svg"]:
            self.plot_type = "png"
            print("plot type {:s} unsupported, changed to png".format(plot_type))
        self.progress_plot_name = os.path.join(self.log_dir, "lossOverEpochs.{:s}".format(self.plot_type))
        self.progress_plot_frequency = plot_frequency
        self.example_plot_frequency = 5

    def cnn_embedding(self, nr_filters_list=[1024,512,512,256,256,128,128,64], kernel_width_list=[4,4,4,4,4,4,4,4], apply_dropout: bool = False):  
        inputs = tf.keras.layers.Input(shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        #add 1D convolutions
        x = inputs
        for i, (nr_filters, kernelWidth) in enumerate(zip(nr_filters_list, kernel_width_list)):
            convParamDict = dict()
            convParamDict["name"] = "conv1D_" + str(i + 1)
            convParamDict["filters"] = nr_filters
            convParamDict["kernel_size"] = kernelWidth
            convParamDict["data_format"]="channels_last"
            convParamDict["kernel_regularizer"]=tf.keras.regularizers.l2(0.01)
            if kernelWidth > 1:
                convParamDict["padding"] = "same"
            x = Conv1D(**convParamDict)(x)
            x = BatchNormalization()(x)
            if apply_dropout:
                x = Dropout(0.5)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        #make the shape of a square matrix
        x = Conv1D(filters=self.INPUT_SIZE, 
                    strides=3, 
                    kernel_size=4, 
                    data_format="channels_last", 
                    activation="sigmoid", 
                    padding="same", name="conv1D_final")(x)
        #ensure the matrix is symmetric, i.e. x = transpose(x)
        x_T = tf.keras.layers.Permute((2,1))(x) #this is the matrix transpose
        x = tf.keras.layers.Add()([x, x_T])
        x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x) #add transpose and divide by 2
        #reshape the matrix into a 2D grayscale image
        x = tf.keras.layers.Reshape((self.INPUT_SIZE,self.INPUT_SIZE,self.INPUT_CHANNELS))(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name="CNN-embedding")
        #model.build(input_shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        #model.summary()
        return model

    def dnn_embedding(self, pretrained_model_path : str = ""):
        inputs = tf.keras.layers.Input(shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        x = Conv1D(filters=1,
                    kernel_size=1,
                    strides=1, 
                    padding="valid",
                    data_format="channels_last",
                    activation="sigmoid")(inputs)
        x = Flatten(name="flatten_1")(x)
        for i, nr_neurons in enumerate([460,881,1690]):
            layerName = "dense_" + str(i+1)
            x = Dense(nr_neurons, activation="relu", kernel_regularizer="l2", name=layerName)(x)
            layerName = "dropout_" + str(i+1)
            x = Dropout(0.1, name=layerName)(x)
        nr_output_neurons = (self.INPUT_SIZE * (self.INPUT_SIZE + 1)) // 2
        x = Dense(nr_output_neurons, activation="relu",kernel_regularizer="l2", name="dense_out")(x)
        dnn_model = tf.keras.Model(inputs=inputs, outputs=x)
        if pretrained_model_path != "":
            try:
                dnn_model.load_weights(pretrained_model_path)
                print("model weights successfully loaded")
            except Exception as e:
                msg = str(e)
                msg += "\nCould not load the weights of pre-trained model"
                print(msg)
        inputs2 = tf.keras.layers.Input(shape=(3*self.INPUT_SIZE, self.NR_FACTORS))
        x = dnn_model(inputs2)
        #place the upper triangular part from dnn model into full matrix
        x = CustomReshapeLayer(self.INPUT_SIZE)(x)
        #symmetrize the output
        x_T = tf.keras.layers.Permute((2,1))(x)
        diag = tf.keras.layers.Lambda(lambda z: -1*tf.linalg.band_part(z, 0, 0))(x)
        x = tf.keras.layers.Add()([x, x_T, diag])
        out = tf.keras.layers.Reshape((self.INPUT_SIZE, self.INPUT_SIZE, self.INPUT_CHANNELS))(x)
        dnn_embedding = tf.keras.Model(inputs=inputs2, outputs=out, name="DNN-embedding")
        return dnn_embedding

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(BatchNormalization())
        result.add(LeakyReLU(alpha=0.2))
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
        inputs = tf.keras.layers.Input(shape=[3*self.INPUT_SIZE,self.NR_FACTORS], name="factorData")

        twoD_conversion = self.generator_embedding
        #the downsampling part of the network, defined for 256x256 images
        down_stack = [
            HiCGAN.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            HiCGAN.downsample(128, 4), # (bs, 64, 64, 128)
            HiCGAN.downsample(256, 4), # (bs, 32, 32, 256)
            HiCGAN.downsample(512, 4), # (bs, 16, 16, 512)
            HiCGAN.downsample(512, 4), # (bs, 8, 8, 512)
            HiCGAN.downsample(512, 4), # (bs, 4, 4, 512)
            HiCGAN.downsample(512, 4), # (bs, 2, 2, 512)
            HiCGAN.downsample(512, 4, apply_batchnorm=False), # (bs, 1, 1, 512)
        ]
        #if the input images are smaller, leave out some layers accordingly
        if self.INPUT_SIZE < 256:
            down_stack = down_stack[:-2] + down_stack[-1:]
        if self.INPUT_SIZE < 128:
            down_stack = down_stack[:-2] + down_stack[-1:]

        #the upsampling portion of the generator, designed for 256x256 images
        up_stack = [
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            HiCGAN.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            HiCGAN.upsample(512, 4), # (bs, 16, 16, 1024)
            HiCGAN.upsample(256, 4), # (bs, 32, 32, 512)
            HiCGAN.upsample(128, 4), # (bs, 64, 64, 256)
            HiCGAN.upsample(64, 4), # (bs, 128, 128, 128)
        ]
        #for smaller images, take layers away, otherwise downsampling won't work
        if self.INPUT_SIZE < 256:
            up_stack = up_stack[:2] + up_stack[3:]
        if self.INPUT_SIZE < 128:
            up_stack = up_stack[:2] + up_stack[3:]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer) # (bs, 256, 256, 3)

        x = inputs
        x = twoD_conversion(x)

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
        #enforce symmetry
        x_T = tf.keras.layers.Permute((2,1,3))(x)
        x = tf.keras.layers.Add()([x, x_T])
        x = tf.keras.layers.Lambda(lambda z: 0.5*z)(x)
        x = tf.keras.layers.Activation("sigmoid")(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # mean squared error or mean absolute error
        if self.loss_type_pixel == "L1":
            pixel_loss = tf.reduce_mean(tf.abs(target - gen_output))
        else: 
            pixel_loss = tf.reduce_mean(tf.square(target - gen_output))
        tv_loss = tf.reduce_mean(tf.image.total_variation(gen_output))
        total_gen_loss = self.lambda_pixel * pixel_loss + self.lambda_disc * gan_loss + self.tv_loss_Weight * tv_loss
        return total_gen_loss, gan_loss, pixel_loss


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[3*self.INPUT_SIZE, self.NR_FACTORS], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.INPUT_SIZE, self.INPUT_SIZE, self.OUTPUT_CHANNELS], name='target_image')
        twoD_conversion = self.discriminator_embedding
        #x = Flatten()(inp)
        #x = Dense(units = self.INPUT_SIZE*(self.INPUT_SIZE+1)//2)(x)
        #x = tf.keras.layers.LeakyReLU()(x)
        #x = CustomReshapeLayer(self.INPUT_SIZE)(x)
        #x_T = tf.keras.layers.Permute((2,1))(x)
        #diag = tf.keras.layers.Lambda(lambda z: -1*tf.linalg.band_part(z, 0, 0))(x)
        #x = tf.keras.layers.Add()([x, x_T, diag])
        #x = tf.keras.layers.Reshape((self.INPUT_SIZE, self.INPUT_SIZE, self.INPUT_CHANNELS))(x)
        #x = tf.keras.layers.concatenate([x, tar])
        #Patch-GAN (Isola et al.)
        d = twoD_conversion(inp)
        d = tf.keras.layers.Concatenate()([d, tar])
        if self.INPUT_SIZE > 64:
            #downsample and symmetrize 1 
            d = HiCGAN.downsample(64, 4, False)(d) # (bs, inp.size/2, inp.size/2, 64)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
            #downsample and symmetrize 2
            d = HiCGAN.downsample(128, 4)(d)# (bs, inp.size/4, inp.size/4, 128)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        else:    
            #downsample and symmetrize 3
            d = HiCGAN.downsample(256, 4)(d)
            d_T = tf.keras.layers.Permute((2,1,3))(d)
            d = tf.keras.layers.Add()([d, d_T])
            d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        #downsample and symmetrize 4
        d = HiCGAN.downsample(256, 4)(d) # (bs, inp.size/8, inp.size/8, 256)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        d = Conv2D(512, 4, strides=1, padding="same", kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 512)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, 4, strides=1, padding="same",
                        kernel_initializer=initializer)(d) #(bs, inp.size/8, inp.size/8, 1)
        d_T = tf.keras.layers.Permute((2,1,3))(d)
        d = tf.keras.layers.Add()([d, d_T])
        d = tf.keras.layers.Lambda(lambda z: 0.5*z)(d)
        #d = tf.keras.layers.Activation("sigmoid")(d)
        return tf.keras.Model(inputs=[inp, tar], outputs=d)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss, real_loss, generated_loss


    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss, disc_real_loss, disc_gen_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        return gen_total_loss, disc_loss, disc_real_loss, disc_gen_loss

    @tf.function
    def validationStep(self, input_image, target, epoch):
        gen_output = self.generator(input_image, training=True)

        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

        gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss, _, _ = self.discriminator_loss(disc_real_output, disc_generated_output)

        return gen_total_loss, disc_loss

    def generate_images(self, model, test_input, tar, epoch: int):
        prediction = model(test_input, training=True)
        pred_mse = tf.reduce_mean(tf.square( tar["out_matrixData"][0], prediction[0] ))
        figname = "testpred_epoch_{:05d}.png".format(epoch)
        figname = os.path.join(self.log_dir, figname)
        display_list = [test_input["factorData"][0], tar["out_matrixData"][0], prediction[0]]
        titleList = ['Input Image', 'Ground Truth', 'Predicted Image (MSE: {:.5f})'.format(pred_mse)]

        fig1, axs1 = plt.subplots(1,len(display_list), figsize=(15,15))
        for i in range(len(display_list)):
            axs1[i].imshow(display_list[i] * 0.5 + 0.5)
            axs1[i].set_title(titleList[i])
        fig1.savefig(figname)
        plt.close(fig1)
        del fig1, axs1

    def fit(self, train_ds, epochs, test_ds, steps_per_epoch: int):
        gen_loss_train = []
        gen_loss_val = []
        disc_loss_train =[]
        disc_loss_real_train = []
        disc_loss_gen_train = []
        disc_loss_val = []
        for epoch in range(epochs):
            #generate sample output
            if epoch % self.example_plot_frequency == 0:
                for example_input, example_target in test_ds.take(1):
                    self.generate_images(self.generator, example_input, example_target, epoch)
            # Train
            train_pbar = tqdm(train_ds.enumerate(), total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            gen_loss_batches = []
            disc_loss_batches = []
            disc_real_loss_batches = []
            disc_gen_loss_batches = []
            for _, (input_image, target) in train_pbar:
                gen_loss, disc_loss, disc_real_loss, disc_gen_loss = self.train_step(input_image["factorData"], target["out_matrixData"], epoch)
                gen_loss_batches.append(gen_loss)
                disc_loss_batches.append(disc_loss)
                disc_real_loss_batches.append(disc_real_loss)
                disc_gen_loss_batches.append(disc_gen_loss)
                if epoch == 0:
                    train_pbar.set_postfix( {"loss": "{:.4f}".format(gen_loss)} )
                else:
                    train_pbar.set_postfix( {"train loss": "{:.4f}".format(gen_loss),
                                             "val loss": "{:.4f}".format(gen_loss_val[-1])} )
            gen_loss_train.append(np.mean(gen_loss_batches))
            disc_loss_train.append(np.mean(disc_loss_batches))
            disc_loss_real_train.append(np.mean(disc_real_loss_batches))
            disc_loss_gen_train.append(np.mean(disc_gen_loss_batches))
            del gen_loss_batches, disc_loss_batches, disc_real_loss_batches, disc_gen_loss_batches
            # Validation
            gen_loss_batches = []
            disc_loss_batches = []
            for input_image, target in test_ds:
                gen_loss, disc_loss = self.validationStep(input_image["factorData"], target["out_matrixData"], epoch)
                gen_loss_batches.append(gen_loss)
                disc_loss_batches.append(disc_loss)
            gen_loss_val.append(np.mean(gen_loss_batches))
            disc_loss_val.append(np.mean(disc_loss_batches))
            del gen_loss_batches, disc_loss_batches, train_pbar
            
            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % self.progress_plot_frequency == 0:
                #self.checkpoint.save(file_prefix = self.checkpoint_prefix)
                #plot loss
                utils.plotLoss(pGeneratorLossValueLists=[gen_loss_train, gen_loss_val],
                              pDiscLossValueLists=[disc_loss_train, disc_loss_real_train, disc_loss_gen_train, disc_loss_val],
                              pGeneratorLossNameList=["training", "validation"],
                              pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                              pFilename=self.progress_plot_name,
                              useLogscaleList=[True, False])
                np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=gen_loss_train, 
                                    genLossVal=gen_loss_val, 
                                    discLossTrain=disc_loss_train, 
                                    discLossVal=disc_loss_val)
                self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.h5".format(epoch)), save_format="h5")
                self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.h5".format(epoch)), save_format="h5")
            

        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        utils.plotLoss(pGeneratorLossValueLists=[gen_loss_train, gen_loss_val],
                       pDiscLossValueLists=[disc_loss_train, disc_loss_real_train, disc_loss_gen_train, disc_loss_val],
                       pGeneratorLossNameList=["training", "validation"],
                       pDiscLossNameList=["train total", "train real", "train gen.", "valid. total"],
                       pFilename=self.progress_plot_name,
                       useLogscaleList=[True, False])
        np.savez(os.path.join(self.log_dir, "lossValues_{:05d}.npz".format(epoch)), 
                                    genLossTrain=gen_loss_train, 
                                    genLossVal=gen_loss_val, 
                                    discLossTrain=disc_loss_train, 
                                    discLossVal=disc_loss_val)
        self.generator.save(filepath=os.path.join(self.log_dir, "generator_{:05d}.h5".format(epoch)), save_format="h5")
        self.discriminator.save(filepath=os.path.join(self.log_dir, "discriminator_{:05d}.h5".format(epoch)), save_format="h5")

    def plotModels(self, outputpath: str, figuretype: str):
        generatorPlotName = "generatorModel.{:s}".format(figuretype)
        generatorPlotName = os.path.join(outputpath, generatorPlotName)
        discriminatorPlotName = "discriminatorModel.{:s}".format(figuretype)
        discriminatorPlotName = os.path.join(outputpath, discriminatorPlotName)
        generatorEmbeddingPlotName = "generatorEmbeddingModel.{:s}".format(figuretype)
        generatorEmbeddingPlotName = os.path.join(outputpath, generatorEmbeddingPlotName)
        discriminatorEmbeddingPlotName = "discriminatorEmbeddingModel.{:s}".format(figuretype)
        discriminatorEmbeddingPlotName = os.path.join(outputpath, discriminatorEmbeddingPlotName)
        tf.keras.utils.plot_model(self.generator, show_shapes=True, to_file=generatorPlotName)
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, to_file=discriminatorPlotName)
        tf.keras.utils.plot_model(self.generator_embedding, show_shapes=True, to_file=generatorEmbeddingPlotName)
        tf.keras.utils.plot_model(self.discriminator_embedding, show_shapes=True, to_file=discriminatorEmbeddingPlotName)

    def predict(self, test_ds, steps_per_record):
        predictedArray = []
        for batch in tqdm(test_ds, desc="Predicting", total=steps_per_record):
            predBatch = self.predictionStep(input_batch=batch).numpy()
            for i in range(predBatch.shape[0]):
                predictedArray.append(predBatch[i][:,:,0])        
        predictedArray = np.array(predictedArray)
        return predictedArray
    
    @tf.function
    def predictionStep(self, input_batch, training=True):
        return self.generator(input_batch, training=training)
  
    
    def loadGenerator(self, trainedModelPath: str):
        '''
            load a trained generator model for prediction
        '''
        try:
            trainedModel = tf.keras.models.load_model(filepath=trainedModelPath, 
                                                  custom_objects={"CustomReshapeLayer": CustomReshapeLayer(self.INPUT_SIZE)})
            self.generator = trainedModel
        except Exception as e:
            msg = str(e)
            msg += "\nError: failed to load trained model"
            raise ValueError(msg)

class CustomReshapeLayer(tf.keras.layers.Layer):
    '''
    reshape a 1D tensor such that it represents 
    the upper triangular part of a square 2D matrix with shape (matsize, matsize)
    #example: 
     [1,2,3,4,5,6] => [[1,2,3],
                       [0,4,5],
                       [0,0,6]]
    '''
    def __init__(self, matsize, **kwargs):
        super(CustomReshapeLayer, self).__init__(**kwargs)
        self.matsize = matsize
        self.triu_indices = [ [x,y] for x,y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1]) ]

    def call(self, inputs):      
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)
        
    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputVec, 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

    def get_config(self):
        return {"matsize": self.matsize}