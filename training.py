import csv
import os
import click
import numpy as np
import tensorflow as tf
import hicGAN
import dataContainer
import records


@click.option("--trainMatrices", "-tm", required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True), multiple=True,
              help="Cooler matrices for training. Use this option multiple times to specify more than one matrix. First matrix belongs to first trainChromPath")
@click.option("--trainChroms", "-tchroms", required=True,
              type=str, 
              help="Train chromosomes. Must be present in all train matrices. Specify multiple chroms separated by spaces, e.g. '10 11 12'.")
@click.option("--trainChromPaths", "-tcp", required=True,
              type=click.Path(exists=True, file_okay=False, readable=True), multiple=True,
              help="Path where chromatin factors for training reside (bigwig files). Use this option multiple times to specify more than one path. First path belongs to first train matrix")
@click.option("--valMatrices", "-vm", required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True), multiple=True,
              help="Cooler matrices for validation. Use this option multiple times to specify more than one matrix")
@click.option("--valChroms", "-vchroms", required=True,
              type=str,
              help="Validation chromosomes. Must be present in all validation matrices. Specify multiple chroms separated by spaces, e.g. '1 2 3'.")
@click.option("--valChromPaths", "-vcp", required=True,
              type=click.Path(exists=True, file_okay=False, readable=True), multiple=True,
              help="Path where chromatin factors for validation reside (bigwig files). Use this option multiple times to specify more than one path. First path belongs to first validation matrix etc.")
@click.option("--windowsize", "-ws", required=True,
              type=click.IntRange(min=64), 
              default=64, show_default=True,
              help="Windowsize for submatrices. Must be integer >80 and (much) smaller than smallest chrom")
@click.option("--outfolder", "-o", required=True,
              type=click.Path(exists=True, writable=True, file_okay=False), 
              help="Folder where trained model and diverse outputs will be stored")
@click.option("--epochs", "-ep", required=True,
              type=click.IntRange(min=1), 
              default=2, show_default=True)
@click.option("--lossWeightPixel", "-lwp", required=False,
              type=click.FloatRange(min=1e-10), 
              default=10000., show_default=True, 
              help="loss weight for L1/L2 error of generator")
@click.option("--lossTypePixel", "-ltp", required=False,
             type=click.Choice(["L1", "L2"]), 
             default="L2", show_default=True,
             help="Type of loss to use per-pixel")
@click.option("--lossWeightTv", "-lvt", required=False,
             type=click.FloatRange(min=0.0),
             default=1e-10, show_default=True,
             help="loss weight for TV loss of generator")
@click.command()
def training(trainmatrices, 
             trainchroms, 
             trainchrompaths, 
             valmatrices, 
             valchroms, 
             valchrompaths,
             windowsize,
             outfolder,
             epochs,
             lossweightpixel,
             losstypepixel,
             lossweighttv):
    
    paramDict = locals().copy()

    #few constants
    recordsize = 2000
    debugstate = None
    figuretype = "png"
    EPOCHS = epochs
    batchsize = 32

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    trainChromNameList = trainchroms.replace(",","")
    trainChromNameList = trainChromNameList.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    valChromNameList = valchroms.replace(",","")
    valChromNameList = valChromNameList.rstrip().split(" ")
    valChromNameList = [x.lstrip("chr") for x in valChromNameList]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList

    #ensure there are as many matrices as chromatin paths
    if len(trainmatrices) != len(trainchrompaths):
        msg = "Number of train matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(trainmatrices), len(trainchrompaths))
        raise SystemExit(msg)
    if len(valmatrices) != len(valchrompaths):
        msg = "Number of validation matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(valmatrices), len(valchrompaths))
        raise SystemExit(msg)

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    for chrom in trainChromNameList:
        for matrix, chromatinpath in zip(trainmatrices, trainchrompaths):
            container = dataContainer.DataContainer(chromosome=chrom,
                                                    matrixfilepath=matrix,
                                                    chromatinFolder=chromatinpath)
            traindataContainerList.append(container)

    #prepare the validation data containers. No data is loaded yet.
    valdataContainerList = []
    for chrom in valChromNameList:
        for matrix, chromatinpath in zip(valmatrices, valchrompaths):
            container = dataContainer.DataContainer(chromosome=chrom,
                                                    matrixfilepath=matrix,
                                                    chromatinFolder=chromatinpath)
            valdataContainerList.append(container)

    #define the load params for the containers
    loadParams = {"scaleFeatures": True,
                  "clampFeatures": False,
                  "scaleTargets": True,
                  "windowsize": windowsize,
                  "flankingsize": windowsize,
                  "maxdist": None}
    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = traindataContainerList[0]
    tfRecordFilenames = []
    nr_samples_list = []
    for container in traindataContainerList + valdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outfolder,
                                                        pRecordSize=recordsize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                         outpath=outfolder,
                                         figuretype=figuretype)
            container.saveMatrix(outputpath=outfolder, index=idx)
        nr_samples_list.append(container.getNumberSamples())
        container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    valdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]

    #different binsizes are ok, as long as no sequence is used
    #not clear which binsize to use for prediction when they differ during training.
    #For now, store the max. 
    binsize = max([container.binsize for container in traindataContainerList])
    paramDict["binsize"] = binsize
    #because of compatibility checks above, 
    #the following properties are the same with all containers,
    #so just use data from first container
    nr_factors = container0.nr_factors
    paramDict["nr_factors"] = nr_factors
    for i in range(nr_factors):
        paramDict["chromFactor_" + str(i)] = container0.factorNames[i]
    nr_trainingSamples = sum(nr_samples_list[0:len(traindataContainerList)])
    storedFeaturesDict = container0.storedFeatures

    #save the training parameters to a file before starting to train
    #(allows recovering the parameters even if training is aborted
    # and only intermediate models are available)
    parameterFile = os.path.join(outfolder, "trainParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)

    #build the input streams for training
    shuffleBufferSize = 3*recordsize
    trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchsize, drop_remainder=False)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    #build the input streams for validation
    validationDs = tf.data.TFRecordDataset(valdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
    
    hicGanModel = hicGAN.HiCGAN(log_dir=outfolder, lambda_pixel=lossweightpixel, loss_type_pixel=losstypepixel, tv_weight=lossweighttv)
    hicGanModel.plotModels(outputpath=outfolder, figuretype=figuretype)
    hicGanModel.fit(train_ds=trainDs, epochs=EPOCHS, test_ds=validationDs, steps_per_epoch=int( np.floor(nr_trainingSamples / batchsize) ))

if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter