import click
import numpy as np
import os
import csv
import tensorflow as tf
import dataContainer
import records
import hicGAN
import utils

@click.option("--trainedModel", "-trm", required=True,
              type=click.Path(exists=True, readable=True, dir_okay=False),
              help="Trained generator model to predict from")
@click.option("--testChromPath", "-tcp", required=True,
              type=click.Path(exists=True, readable=True, file_okay=False),
              help="Path where test data (bigwig files) resides")
@click.option("--testChroms", "-tchroms", required=True,
              type=str,
              help="Chromosomes for testing. Must be available in all bigwig files")
@click.option("--outfolder", "-o", required=False,
              type=click.Path(exists=True, writable=True, file_okay=False),
              default="./", show_default=True,
              help="Output path for predicted coolers")
@click.option("--multiplier", "-mul", required=False,
             type=click.IntRange(min=1), 
             default=10, show_default=True)
@click.option("--binsize", "-b", required=True,
              type=click.IntRange(min=1000), 
              help="bin size for binning the chromatin features")
@click.option("--batchsize", "-bs", required=False,
              type=click.IntRange(min=1),
              default=32, show_default=True,
              help="batchsize for predicting")
@click.option("--windowsize", "-ws", required=True,
              type=click.Choice(choices=["64", "128", "256"]),
              help="windowsize for predicting; must be the same as in trained model. Supported values are 64, 128 and 256")
@click.command()
def prediction(trainedmodel, 
                testchrompath,
                testchroms,
                outfolder,
                multiplier,
                binsize,
                batchsize,
                windowsize
                ):
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowsize = int(windowsize)
    flankingsize = windowsize

    paramDict = locals().copy()
        
    #extract chromosome names from the input
    chromNameList = testchroms.replace(",", " ").rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixfilepath=None,
                                                  chromatinFolder=testchrompath,
                                                  binsize=binsize)) 
    #define the load params for the containers
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowsize": windowsize,
                  "flankingsize": flankingsize,
                  "maxdist": maxdist}
    #now load the data and write TFRecords, one container at a time.
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = testdataContainerList[0]
    nr_factors = container0.nr_factors
    tfRecordFilenames = []
    sampleSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outfolder,
                                                        pRecordSize=None)[0]) #list with 1 entry
        sampleSizeList.append( int( np.ceil(container.getNumberSamples() / batchsize) ) )
    
    nr_factors = container0.nr_factors
    #data is no longer needed, unload it
    for container in testdataContainerList:
        container.unloadData() 

    trained_GAN = hicGAN.HiCGAN(log_dir=outfolder, number_factors=nr_factors)
    trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
    predList = []
    for record, container, nr_samples in zip(tfRecordFilenames, testdataContainerList, sampleSizeList):
        storedFeaturesDict = container.storedFeatures
        testDs = tf.data.TFRecordDataset(record, 
                                            num_parallel_reads=None,
                                            compression_type="GZIP")
        testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testDs = testDs.batch(batchsize, drop_remainder=False) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
        #if validationmatrix is not None:
        #    testDs = testDs.map(lambda x, y: x) #drop the target matrices (they are for evaluation)
        testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)
        predArray = trained_GAN.predict(test_ds=testDs, steps_per_record=nr_samples)
        triu_indices = np.triu_indices(windowsize)
        predArray = np.array( [np.array(x[triu_indices]) for x in predArray] )
        predList.append(predArray)
    predList = [utils.rebuildMatrix(pArrayOfTriangles=x, pWindowSize=windowsize, pFlankingSize=windowsize) for x in predList]
    predList = [utils.scaleArray(x) * multiplier for x in predList]

    matrixname = os.path.join(outfolder, "predMatrix.cool")
    utils.writeCooler(pMatrixList=predList, 
                      pBinSizeInt=binsize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

    parameterFile = os.path.join(outfolder, "predParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)
    
    for tfrecordfile in tfRecordFilenames:
        if os.path.exists(tfrecordfile):
            os.remove(tfrecordfile)

if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter