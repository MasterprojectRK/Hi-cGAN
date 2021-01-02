import utils
import records
import os
import numpy as np
from Bio import SeqIO
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm
import pandas as pd

class DataContainer():
    def __init__(self, chromosome, matrixfilepath, chromatinFolder, binsize=None):
        self.chromosome = str(chromosome)
        self.matrixfilepath = matrixfilepath
        self.chromatinFolder = chromatinFolder
        self.FactorDataArray = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.binsize = None
        if matrixfilepath is None: #otherwise it will be defined by the Hi-C matrix itself upon loading
            self.binsize = binsize
        self.factorNames = None
        self.prefixDict_factors = None
        self.prefixDict_matrix = None
        self.prefixDict_sequence = None
        self.chromSize_factors = None
        self.chromSize_matrix = None
        self.storedFeatures = None
        self.storedFiles = None
        self.windowsize = None
        self.flankingsize = None
        self.maxdist = None
        self.data_loaded = False

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return
        #ensure that binsizes for matrix (if given) and factors match
        if self.binsize is None:
            msg = "No binsize given; use a Hi-C matrix or explicitly specify binsize for the container"   
            raise TypeError(msg)
        ###load data for a specific chromsome
        #get the names of the bigwigfiles
        bigwigFileList = utils.getBigwigFileList(self.chromatinFolder)
        bigwigFileList = sorted(bigwigFileList)
        if len(bigwigFileList) is None:
            msg = "Warning: folder {:s} does not contain any bigwig files"
            msg = msg.format(self.chromatinFolder)
            print(msg)
            return
        #check the chromosome name prefixes (e.g. "" or "chr") and sizes
        chromSizeList = []
        prefixDict_factors = dict()
        for bigwigFile in bigwigFileList:
            try:
                prefixDict_factors[bigwigFile] = utils.getChromPrefixBigwig(bigwigFile)
                chromname = prefixDict_factors[bigwigFile] + self.chromosome
                chromSizeList.append( utils.getChromSizesFromBigwig(bigwigFile)[chromname] )
            except Exception as e:
                msg = str(e) + "\n"
                msg += "Could not load data from bigwigfile {}".format(bigwigFile) 
                raise IOError(msg)
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise IOError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            chromSize_factors = min(chromSizeList)
        else:
            chromSize_factors = chromSizeList[0]
        #the chromosome lengths of matrices and bigwig files must be equal
        if self.chromSize_matrix is not None \
                and self.chromSize_matrix != chromSize_factors:
            msg = "Chrom lengths not equal between matrix and bigwig files\n"
            msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, chromSize_factors)
            raise IOError(msg)
        #load the data into memory now
        self.factorNames = [os.path.splitext(os.path.basename(name))[0] for name in bigwigFileList]
        self.nr_factors = len(self.factorNames)
        self.prefixDict_factors = prefixDict_factors
        self.chromSize_factors = chromSize_factors
        nr_bins = int( np.ceil(self.chromSize_factors / self.binsize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        msg = "Loaded {:d} chromatin features from folder {:s}\n"
        msg = msg.format(self.nr_factors, self.chromatinFolder)
        featLoadedMsgList = [] #pretty printing for features loaded
        for i, bigwigFile in enumerate(bigwigFileList):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            tmpArray = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                pBinSizeInt=self.binsize,
                                                pChromStr=chromname,
                                                pChromSize=self.chromSize_factors)
            if clampFeatures:
                tmpArray = utils.clampArray(tmpArray)
            if scaleFeatures:
                tmpArray = utils.scaleArray(tmpArray)
            self.FactorDataArray[i] = tmpArray
            nr_nonzero_abs = np.count_nonzero(tmpArray)
            nr_nonzero_perc = nr_nonzero_abs / tmpArray.size * 100
            msg2 = "{:s} - min. {:.3f} - max. {:.3f} - nnz. {:d} ({:.2f}%)"
            msg2 = msg2.format(bigwigFile, tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
            featLoadedMsgList.append(msg2)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
        print(msg + "\n".join(featLoadedMsgList))
            
    def __loadMatrixData(self, scaleMatrix=False):
        #load Hi-C matrix from cooler file
        if self.matrixfilepath is None:
            return
        try:
            prefixDict_matrix = {self.matrixfilepath: utils.getChromPrefixCooler(self.matrixfilepath)}
            chromname = prefixDict_matrix[self.matrixfilepath] + self.chromosome
            chromsize_matrix = utils.getChromSizesFromCooler(self.matrixfilepath)[chromname]
            sparseHiCMatrix, binsize = utils.getMatrixFromCooler(self.matrixfilepath, chromname)
        except:
            msg = "Error: Could not load data from Hi-C matrix {:s}"
            msg = msg.format(self.matrixfilepath)
            raise IOError(msg)
        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)       
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise IOError(msg)
        self.chromSize_matrix = chromsize_matrix
        #ensure that binsizes for matrix and factors (if given) match
        if self.binsize is None or self.binsize == binsize:
            self.binsize = binsize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binsize is not None and self.binsize != binsize:
            msg = "Matrix has wrong binsize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binsize, self.binsize)
            raise IOError(msg)
        msg = "Loaded cooler matrix {:s}\n".format(self.matrixfilepath)
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:d} -- max. {:d} -- nnz. {:d}"
        msg = msg.format(self.chromosome, self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], int(self.sparseHiCMatrix.min()), int(self.sparseHiCMatrix.max()), self.sparseHiCMatrix.getnnz() )
        print(msg)
    
    def __unloadFactorData(self):
        #unload chromatin factor data to save memory, but do not touch metadata 
        self.FactorDataArray = None
        
    def __unloadMatrixData(self):
        #unload matrix data to save memory, but do not touch metadata
        self.sparseHiCMatrix = None

    def unloadData(self):
        #unload all data to save memory, but do not touch metadata
        self.__unloadFactorData
        self.__unloadMatrixData
        self.windowsize = None
        self.flankingsize = None
        self.maxdist = None
        self.data_loaded = False
    
    def loadData(self, windowsize, flankingsize=None, maxdist=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False):
        if not isinstance(windowsize, int):
            msg = "windowsize must be integer"
            raise TypeError(msg)
        if isinstance(maxdist, int):
            maxdist = np.clip(maxdist, a_min=1, a_max=self.windowsize)
        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.windowsize = windowsize
        self.flankingsize = flankingsize
        self.maxdist = maxdist
        self.data_loaded = True

    def checkCompatibility(self, containerIterable):
        ret = []
        try:
           for container in containerIterable:
               ret.append(self.__checkCompatibility(container))
        except:
            ret = [self.__checkCompatibility(containerIterable)]
        return np.all(ret)
        
    def __checkCompatibility(self, container):
        if not isinstance(container, DataContainer):
            return False
        if not self.data_loaded or not container.data_loaded:
            return False
        #check if the same kind of data is available for all containers
        factorsOK = type(self.FactorDataArray) == type(container.FactorDataArray)
        matrixOK = type(self.sparseHiCMatrix) == type(container.sparseHiCMatrix)
        #check if windowsize, flankingsize and maxdist match
        windowsizeOK = self.windowsize == container.windowsize
        flankingsizeOK = self.flankingsize == container.flankingsize
        maxdistOK = self.maxdist == container.maxdist
        #sanity check loading of bigwig files
        if self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the numbers and names of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        factorsOK = factorsOK and (self.factorNames == container.factorNames)
        return factorsOK and matrixOK and windowsizeOK and flankingsizeOK and maxdistOK
        
    def writeTFRecord(self, pOutfolder, pRecordSize=None):
        '''
        Write a dataset to disk in tensorflow TFRecord format
        
        Parameters:
            pOutfolder (str): directory where TFRecords will be written
            pRecordsize (int): split the TFRecords into multiple files containing approximately this number of samples
        
        Returns:
            list of filenames written
        '''

        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to write"
            print(msg)
            return None
        nr_samples = self.getNumberSamples()
        #adjust record size (yields smaller files and reduces memory load)
        recordsize = nr_samples
        if pRecordSize is not None and pRecordSize < recordsize:
            recordsize = pRecordSize
        #compute number of record files, number of samples 
        #in each file and corresponding indices
        nr_files = int( np.ceil(nr_samples/recordsize) )
        target_ct = int( np.floor(nr_samples/nr_files) )
        samples_per_file = [target_ct]*(nr_files-1) + [nr_samples-(nr_files-1)*target_ct]
        sample_indices = [sum(samples_per_file[0:i]) for i in range(len(samples_per_file)+1)] 
        #write the single files
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        recordfiles = [os.path.join(pOutfolder, "{:s}_{:s}_{:03d}.tfrecord".format(folderName, str(self.chromosome), i + 1)) for i in range(nr_files)]
        for recordfile, firstIndex, lastIndex in tqdm(zip(recordfiles, sample_indices, sample_indices[1:]), desc="Storing TFRecord files", total=len(recordfiles)):
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(pFirstIndex=firstIndex, 
                                                                        pLastIndex=lastIndex, 
                                                                        pOutfolder=pOutfolder)
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles

    def getNumberSamples(self):
        if not self.data_loaded:
            return None
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [self.windowsize+2*self.flankingsize, self.windowsize+2*self.flankingsize, (self.windowsize+2*self.flankingsize)*self.binsize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        #check if all features have the same number of samples
        if len(set( [x for x in nr_samples_list if x>0] )) != 1:
            msg = "Error: sample binning / DNA sequence encoding went wrong"
            raise RuntimeError(msg)
        return max(nr_samples_list)

    def __getMatrixData(self, idx):
        if self.matrixfilepath is None:
            return None # this can't work
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th matrix starts flankingsize away from the boundary
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if flankingsize is None:
            flankingsize = windowsize
            self.flankingsize = windowsize
        startInd = idx + flankingsize
        stopInd = startInd + windowsize
        trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()
        trainmatrix = np.array(np.nan_to_num(trainmatrix))
        trainmatrix = np.expand_dims(trainmatrix, axis=-1) #make Hi-C (sub-)matrix an RGB image
        return trainmatrix
    
    def __getFactorData(self, idx):
        if self.chromatinFolder is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th feature matrix starts at position 0
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if flankingsize is None:
            flankingsize = windowsize
            self.flankingsize = windowsize
        startIdx = idx
        endIdx = startIdx + 2*flankingsize + windowsize
        factorArray = self.FactorDataArray[startIdx:endIdx]
        factorArray = np.expand_dims(factorArray, axis=-1)
        return factorArray

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        factorArray = self.__getFactorData(idx)
        matrixArray = self.__getMatrixData(idx)
        if matrixArray is not None:
            matrixArray = matrixArray.astype("float32")
        return {"factorData": factorArray.astype("float32"), 
                "out_matrixData": matrixArray}
        
    def plotFeatureAtIndex(self, idx, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or integer in 0..{:d}".format(idx, self.FactorDataArray.shape[0]-1)
            raise ValueError(msg)
        if isinstance(idx, int):
            factorArray = self.__getFactorData(idx)
            startBin = idx
        else:
            factorArray = self.FactorDataArray 
            startBin = None
        for plotType in ["box", "line"]:   
            utils.plotChromatinFactors(pFactorArray=factorArray, 
                                        pFeatureNameList=self.factorNames,
                                        pChromatinFolder=self.chromatinFolder,
                                        pChrom=self.chromosome,
                                        pBinsize=self.binsize,
                                        pStartbin=startBin,
                                        pOutputPath=outpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)
    
    def plotFeaturesAtPosition(self, position, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(position, int) and position > self.chromSize_factors:
            msg = "Error: Invalid position {:d}; must be in 0..{:d}"
            msg = msg.format(position, self.chromSize_factors)
            raise ValueError(msg)
        #compute the bin index from the position
        elif isinstance(position, int):
            idx = int(np.floor(position / self.binsize))
        else:
            idx = None
        return self.plotFeatureAtIndex(idx=idx,
                                        outpath=outpath,
                                        figuretype=figuretype)

    def saveMatrix(self, outputpath, index=None):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to save"
            print(msg)
            return
        sparseMatrix = None
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if not isinstance(flankingsize, int):
            flankingsize = windowsize
        if isinstance(self.maxdist, int) and self.maxdist < windowsize and self.maxdist > 0:
            maxdist = self.maxdist
        else:
            maxdist = windowsize
        if isinstance(index, int) and index < self.getNumberSamples():
            tmpMat = np.zeros(shape=(windowsize, windowsize))
            indices = np.mask_indices(windowsize, utils.maskFunc, k=maxdist)
            tmpMat[indices] = self.__getMatrixData(idx=index)
            sparseMatrix = csr_matrix(tmpMat)
        else:
            sparseMatrix = self.sparseHiCMatrix
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        filename = "matrix_{:s}_chr{:s}_{:s}".format(folderName, str(self.chromosome), str(index))
        filename = os.path.join(outputpath, filename)
        save_npz(file=filename, matrix=sparseMatrix)

    def __prepareWriteoutDict(self, pFirstIndex, pLastIndex, pOutfolder):
        if not self.data_loaded:
            msg = "Error: no data loaded, nothing to prepare"
            raise RuntimeError(msg)
        data = [ self.getSampleData(idx=i) for i in range(pFirstIndex, pLastIndex) ]
        recordDict = dict()
        storedFeaturesDict = dict()
        if len(data) < 1:
            msg = "Error: No data to write"
            raise RuntimeError(msg)
        for key in data[0]:
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}
        return recordDict, storedFeaturesDict


class ImprovementDataContainer():
    def __init__(self, chromosome: str, trainmatrix_filepath: str, targetmatrix_filepath: str):
        self.chromosome = chromosome
        self.trainmatrix_filepath = trainmatrix_filepath
        self.targetmatrix_filepath = targetmatrix_filepath
        self.windowsize = 256
        self.sparse_trainmatrix = None
        self.sparse_targetmatrix = None
        self.chromsize_train = 0
        self.chromsize_target = 0
        self.binsize_train = 0
        self.binsize_target = 0
        self.data_loaded = False
        self.storedFeatures = None
        self.storedFiles = None

    def __loadMatrixData(self, targetStr):
        #load Hi-C matrix from cooler file
        if targetStr == "train":
            matrixfilepath = self.trainmatrix_filepath
            sparseHiCMatrix = [self.sparse_trainmatrix]
            chromsize = [self.chromsize_train]
            binsize = [self.binsize_train]
        else:
            matrixfilepath = self.targetmatrix_filepath
            sparseHiCMatrix = [self.sparse_targetmatrix]
            chromsize = [self.chromsize_target]
            binsize = [self.binsize_target]
        try:
            prefixDict_matrix = {matrixfilepath: utils.getChromPrefixCooler(matrixfilepath)}
            chromname = prefixDict_matrix[matrixfilepath] + self.chromosome
            chromsize[0] = utils.getChromSizesFromCooler(matrixfilepath)[chromname]
            sparseHiCMatrix[0], binsize[0] = utils.getMatrixFromCooler(matrixfilepath, chromname)
        except:
            msg = "Error: Could not load data from Hi-C matrix {:s}"
            msg = msg.format(matrixfilepath)
            raise IOError(msg)
        #scale to 0..1
        sparseHiCMatrix[0] = utils.scaleArray(sparseHiCMatrix[0])       
            
        msg = "Loaded cooler matrix {:s}\n".format(matrixfilepath)
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:d} -- max. {:d} -- nnz. {:d}"
        msg = msg.format(self.chromosome, sparseHiCMatrix[0].shape[0], sparseHiCMatrix[0].shape[1], int(sparseHiCMatrix[0].min()), int(sparseHiCMatrix[0].max()), sparseHiCMatrix[0].getnnz() )
        print(msg)

    def __checkOk(self):
        binsizeOk = (self.binsize_train == self.binsize_target)
        chromsizeOk = (self.chromsize_train == self.chromsize_target)
        return (binsizeOk and chromsizeOk)

    def loadData(self):
        self.__loadMatrixData("train")
        self.__loadMatrixData("target")
        self.__checkOk()
        self.data_loaded = True
    
    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        trainmatrix = self.__getMatrixData(idx, self.sparse_trainmatrix)
        targetmatrix = self.__getMatrixData(idx, self.sparse_targetmatrix)

        return {"factorData": trainmatrix, 
                "out_matrixData": targetmatrix}
        
    def getNumberSamples(self):
        if not self.data_loaded:
            return 0
        fullsize = self.sparse_targetmatrix.shape[0]
        available_size = fullsize - 2* self.windowsize
        nr_samples = available_size - self.windowsize + 1
        if nr_samples <= 0:
            msg = "Matrix is too small for given windowsize"
            raise ValueError(msg)
        return nr_samples 

    def __getMatrixData(self, idx, matrixObj: csr_matrix):
        startInd = idx + self.windowsize
        stopInd = startInd + self.windowsize
        trainmatrix = matrixObj[startInd:stopInd,startInd:stopInd].todense()
        trainmatrix = np.array(np.nan_to_num(trainmatrix))
        trainmatrix = np.expand_dims(trainmatrix, axis=-1).astype("float32") #make Hi-C (sub-)matrix an RGB image
        return trainmatrix

    def __prepareWriteoutDict(self, first_index, last_index):
        if not self.data_loaded:
            msg = "Error: no data loaded, nothing to prepare"
            raise RuntimeError(msg)
        data = [ self.getSampleData(idx=i) for i in range(first_index, last_index) ]
        recordDict = dict()
        storedFeaturesDict = dict()
        if len(data) < 1:
            msg = "Error: No data to write"
            raise RuntimeError(msg)
        for key in data[0]:
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}
        return recordDict, storedFeaturesDict
    
    def writeTFRecord(self, pOutfolder, pRecordSize=None):
        '''
        Write a dataset to disk in tensorflow TFRecord format
        
        Parameters:
            pOutfolder (str): directory where TFRecords will be written
            pRecordsize (int): split the TFRecords into multiple files containing approximately this number of samples
        
        Returns:
            list of filenames written
        '''

        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to write"
            print(msg)
            return None
        nr_samples = self.getNumberSamples()
        #adjust record size (yields smaller files and reduces memory load)
        recordsize = nr_samples
        if pRecordSize is not None and pRecordSize < recordsize:
            recordsize = pRecordSize
        #compute number of record files, number of samples 
        #in each file and corresponding indices
        nr_files = int( np.ceil(nr_samples/recordsize) )
        target_ct = int( np.floor(nr_samples/nr_files) )
        samples_per_file = [target_ct]*(nr_files-1) + [nr_samples-(nr_files-1)*target_ct]
        sample_indices = [sum(samples_per_file[0:i]) for i in range(len(samples_per_file)+1)] 
        #write the single files
        recordfiles = [os.path.join(pOutfolder, "{:s}_{:s}_{:03d}.tfrecord".format("test", str(self.chromosome), i + 1)) for i in range(nr_files)]
        for recordfile, firstIndex, lastIndex in tqdm(zip(recordfiles, sample_indices, sample_indices[1:]), desc="Storing TFRecord files", total=len(recordfiles)):
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(first_index=firstIndex, 
                                                                        last_index=lastIndex)
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles