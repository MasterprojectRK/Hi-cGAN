import click
import csv
import os
import numpy as np
import tensorflow as tf

from dataContainer import ImprovementDataContainer
import records
import hicGAN
import utils

@click.option("--trainMatrices", "-tm", required=True,
              multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--targetMatrices", "-targm", required=True,
              multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--trainChroms", "-tchroms", required=True,
              type=str)
@click.option("--valChroms", "-vchroms", required=True,
             type=str)
@click.option("--testMatrix", "-testm", required=True,
             type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option("--testChroms", "-tcs", required=True,
             type=str)
@click.option("--epochs", "-ep", required=True,
             type=click.IntRange(min=2), default=2)
@click.option("--outputpath", "-o", required=True)
@click.command
def improve(trainmatrices,
            targetmatrices,
            trainchroms,
            valchroms,
            testmatrix,
            testchroms,
            epochs,
            outputpath):
    batchsize = 4
    recordsize = 2000
    figuretype = "png"
    multiplier = 1000
    windowsize = 256
    paramDict = locals().copy()

    #remove spaces, commas and "chr" from the train and val chromosome lists
    #ensure each chrom name is used only once, but allow the same chrom for train and validation
    #sort the lists and write to param dict
    trainChromNameList = " ".join(trainchroms.replace(","," ").split()) #remove commas, multiple spaces, trailing spaces
    trainChromNameList = trainChromNameList.split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList
    valChromNameList = " ".join(valchroms.replace(","," ").split())
    valChromNameList = valChromNameList.split(" ")
    valChromNameList = [x.lstrip("chr") for x in valChromNameList]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList
    testChromNameList = " ".join(testchroms.replace(","," ").split())
    testChromNameList = testChromNameList.split(" ")
    testChromNameList = [x.lstrip("chr") for x in testChromNameList]
    testChromNameList = sorted(list(set(testChromNameList)))
    paramDict["testChromNameList"] = testChromNameList

    #check if number of train and target matrices is the same
    if len(targetmatrices) != len(trainmatrices):
        msg = "Error. Number of train- and target matrices must be equal"
        raise ValueError(msg)

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    for chrom in trainChromNameList:
        for trainmatrix, targetmatrix in zip(trainmatrices, targetmatrices):
            container = ImprovementDataContainer(chromosome=chrom,
                                                 trainmatrix_filepath=trainmatrix,
                                                 targetmatrix_filepath=targetmatrix)
            traindataContainerList.append(container)

    #prepare the validation data containers. No data is loaded yet.
    valdataContainerList = []
    for chrom in valChromNameList:
        for validationmatrix, targetmatrix in zip(trainmatrices, targetmatrices):
            container = ImprovementDataContainer(chromosome=chrom,
                                                 trainmatrix_filepath=validationmatrix,
                                                 targetmatrix_filepath=targetmatrix)
            valdataContainerList.append(container) 

    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    tfRecordFilenames = []
    nr_samples_list = []
    for container in traindataContainerList + valdataContainerList:
        container.loadData()
    
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outputpath,
                                                        pRecordSize=recordsize))
        nr_samples_list.append(container.getNumberSamples())
        #container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    valdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]

    nr_trainingSamples = sum(nr_samples_list[0:len(traindataContainerList)])
    storedFeaturesDict = traindataContainerList[0].storedFeatures

    parameterFile = os.path.join(outputpath, "improveParams.csv")    
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
    trainDs = trainDs.batch(batchsize, drop_remainder=True)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    #build the input streams for validation
    validationDs = tf.data.TFRecordDataset(valdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
    
    steps_per_epoch = int( np.floor(nr_trainingSamples / batchsize) )

    hicGanModel = hicGAN.ImproveGAN(log_dir=outputpath)
    hicGanModel.plotModels(outputpath=outputpath, figuretype=figuretype)
    hicGanModel.fit(train_ds=trainDs, epochs=epochs, test_ds=validationDs, steps_per_epoch=steps_per_epoch)

    #######
    #######try to improve the testmatrix / testchroms now
    #######
    #prepare the validation data containers. No data is loaded yet.
    testdataContainerList = []
    for chrom in testChromNameList:
        container = ImprovementDataContainer(chromosome=chrom,
                                            trainmatrix_filepath=testmatrix,
                                            targetmatrix_filepath=testmatrix) #inefficient, to be improved
        testdataContainerList.append(container) 

    #now load the data and write TFRecords, one container at a time.
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    testRecordFilenames = []
    nr_samples_list = []
    for container in testdataContainerList:
        container.loadData()
        testRecordFilenames.append(container.writeTFRecord(pOutfolder=outputpath,
                                                        pRecordSize=recordsize))
        nr_samples_list.append(container.getNumberSamples())

    predList = []
    for record, container, nr_samples in zip(testRecordFilenames, testdataContainerList, nr_samples_list):
        storedFeaturesDict = container.storedFeatures
        testDs = tf.data.TFRecordDataset(record, 
                                            num_parallel_reads=None,
                                            compression_type="GZIP")
        testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testDs = testDs.map(lambda x, y: x) #drop the target matrices (they are for evaluation)
        testDs = testDs.batch(batchsize, drop_remainder=False) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
        testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)
        predArray = hicGanModel.predict(test_ds=testDs, steps_per_record=nr_samples)
        triu_indices = np.triu_indices(windowsize)
        predArray = np.array( [np.array(x[triu_indices]) for x in predArray] )
        predList.append(predArray)
    predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowsize, pFlankingSize=windowsize) for x in predList]
    predList = [utils.scaleArray(x) * multiplier for x in predList]

    matrixname = os.path.join(outputpath, "improvedMatrix.cool")
    utils.writeCooler(pMatrixList=predList, 
                      pBinSizeInt=testdataContainerList[0].binsize_train, 
                      pOutfile=matrixname, 
                      pChromosomeList=testChromNameList)

    for tfRecordfile in traindataRecords + valdataRecords + testRecordFilenames:
        if os.path.exists(tfRecordfile):
            os.remove(tfRecordfile)

if __name__ == "__main__":
    improve() #pylint: disable=no-value-for-parameter