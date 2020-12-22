import click
import numpy as np
import os
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
              type=click.Path(exists=True, writable=True),
              default="./", show_default=True,
              help="Output path for predicted coolers")
@click.option("--multiplier", "-mul", required=False,
             type=click.IntRange(min=1), 
             default=10, show_default=True)
@click.command()
def prediction(trainedmodel, 
                testchrompath,
                testchroms,
                outfolder,
                multiplier
                ):
    binSizeInt = 25000
    scalefactors = True
    clampfactors = False
    scalematrix = True
    windowsize = 64
    flankingsize = windowsize
    maxdist = None
    batchSizeInt = 32

    
    #extract chromosome names from the input
    chromNameList = testchroms.replace(",", " ").rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixfilepath=None,
                                                  chromatinFolder=testchrompath,
                                                  binsize=binSizeInt)) 
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
    tfRecordFilenames = []
    sampleSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outfolder,
                                                        pRecordSize=None)[0]) #list with 1 entry
        sampleSizeList.append( int( np.ceil(container.getNumberSamples() / batchSizeInt) ) )
        container.unloadData() 

    trained_GAN = hicGAN.HiCGAN(log_dir=outfolder)
    trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
    predList = []
    for record, container, nr_samples in zip(tfRecordFilenames, testdataContainerList, sampleSizeList):
        storedFeaturesDict = container.storedFeatures
        testDs = tf.data.TFRecordDataset(record, 
                                            num_parallel_reads=None,
                                            compression_type="GZIP")
        testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testDs = testDs.batch(batchSizeInt, drop_remainder=False) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
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
                      pBinSizeInt=binSizeInt, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)


if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter