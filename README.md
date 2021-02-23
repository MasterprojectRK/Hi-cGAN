# Hi-cGAN

Hi-cGAN is a conditional generative adversarial network 
designed to predict Hi-C contact matrices from one-dimensional
chromatin feature data, e. g. from ChIP-seq experiments.

## Installation

Simply `git clone` this repository into an empty folder of your choice.
It is recommended to use conda or another package manager to install
the following dependencies into an empty environment:
dependency | tested version
-----------|---------------
click | 7.1.2
cooler | 0.8.10
graphviz | 2.42.3
matplotlib | 3.3.2
numpy | 1.19.4
pandas | 1.1.4
pybigwig | 0.3.17
pydot | 1.4.1
python | 3.7.8
scikit-learn | 0.23.2
scipy | 1.5.3
tensorflow-gpu | 2.2.0   
tqdm | 4.50.2

Other versions *might* work, but are untested and will usually cause dependency
conflicts. Using tensorflow without GPU support is possible, but will be very slow and is thus not recommended.

## Usage
Hi-cGAN consists of two python scripts, training.py and predict.py,
which will be explained below.

### Training
This script will train the cGAN Generator and Discriminator
by alternately updating their weights using the Adam optimizer
on a joint loss function (L1/L2 loss, Discriminator Loss, TV Loss).  

Synopsis: `python training.py [parameters and options]`  
Parameters / Options:  
* --trainmatrices, -tm [required]  
Hi-C matrices for training. Must be in cooler format. Use this option multiple times to specify more than one matrix (e.g. `-tm matrix1.cool -tm matrix2.cool`). First matrix belongs to first training chromatin feature path and so on, see below.
* --trainChroms, -tchroms [required]  
Chromosomes for training. Specify without leading "chr" and separated by spaces,
e.g. "1 3 5 11". These chromosomes must be present in all train matrices.
Currently, only integer values are supported.
* --trainChromPaths, -tcp [required]  
Path where chromatin features for training reside.
The program will look for bigwig files in this folder, subfolders are not considered.
Specify one trainChromPath for each training matrix, in the desired order,
see above.
Note that the chromatin features for training and prediction must have the same names. 
* --valMatrices, -vm, [required]  
Hi-C matrices for validation. Must be in cooler format. Use this option multiple times to specify more than one matrix.
* --valChroms, -vchroms [required]  
Same as trainChroms, just for validation
* --valChromPaths, -vcp [required]  
Same as trainChromPaths, just for validation
* --windowsize, -ws [required]  
Windowsize for submatrices in sliding window approach. 64, 128 and 256 are supported.
Default: 64.
* --outfolder, -o [required]  
Folder where output will be stored.
Must be writable and have several 100s of MB of free storage space.
* --epochs, -ep [required]  
Number of epochs for training. 
* --batchsize, -bs [required]  
Batch size for training. Choose integer between 1 and 256. 
Mind the memory limits of your GPU; in a test environment with 15GB GPU memory, batchsizes 32,4,2 were safely within limits for windowsizes 64,128,256, respectively.
* --lossWeightPixel, -lwp  
Loss weight for the L1 or L2 loss in the generator. 
Floating point value, default: 100.
* --lossWeightDisc, -lwd  
loss weight for the discriminator error, floating point value, default: 0.5
* --lossTypePixel, -ltp  
Type of per-pixel loss to use for the generator; choose from L1 (mean abs. error) or L2 (mean squared error). Default: L1.
* --lossWeightTv, -lvt  
loss weight for Total-Variation-loss of generator; higher value - more smoothing.
Default: 1e-10.
* --learningRate, -lr  
Learning rate for the Adam optimizers (Generator and Discriminator).
Default: 2e-5.
* --beta1, -b1  
beta1 parameter for the Adam optimizers (Generator and Discriminator). Default 0.5.
* --flipsamples, -fs  
Flip training matrices and chromatin features (data augmentation). Default: False.
* --pretrainedIntroModel, -ptm  
Undocumented, developer use only.
* --figuretype, -ft  
Figure type for all plots, choose from png, pdf, svg. Default: png.
* --recordsize, -rs  
Approx. size (number of samples) of the tfRecords used in the data pipeline for training. Can be tweaked to balance the load between RAM / GPU / CPU. Default: 2000.  

Returns: 
* The following files will be stored in the chosen output path (option `-o`) 
* Trained models of generator and discriminator in h5py format, stored in output path (every 20 epochs and after completion).
* Sample images of generated Hi-C matrices (every 5 epochs).
* Parameter file in csv format for reference.
* Tensorflow TFRecord files containing serialized train samples. Do not touch these files while the program is running, they should be open for reading anyway and will be deleted automatically upon completion.


### Predict
This script will predict Hi-C matrices using chromatin features and a trained generator model as input.  

Synopsis: `python predict.py [parameters and options]`  
Parameters / Options:  
* --trainedModel, -trm [required]
Trained generator model to predict from, h5py format.
Generated by training.py above.
* --testChromPath, -tcp [required]  
Same as trainChromPaths, just for testing / prediction.
The number and names of bigwig files in this path must be the same as for training.
* --testChroms, -tchroms [required]  
Chromosomes for testing (to be predicted). Must be available in all bigwig files.
Input format is the same as above, e.g. "8 12 21"
* --outfolder, -o  
Output path for predicted Hi-C matrices (in cooler format). Default: current path
* --multiplier, -mul  
Multiplier for better visualization of results. 
Integer value greater equal 1, default: 1000.  

Returns:  
* Predicted matrix in cooler format, defined for the specified test chromosomes.  
* Parameter file in csv format for reference.  

### Example usage
t.b.d.

## Notes
### Creating bigwig files for chromatin features from BAM alignment files
If bigwig files of the chromatin features are not available,
it is possible to use `bamCoverage` [[link]](https://github.com/deeptools/deepTools/blob/master/docs/content/tools/bamCoverage.rst) to convert alignments in .bam format to bigwig
for example as shown below.
```
# creating a bigwig file from the bam file BAMFILE (which ends in ".bam")
OUTFILE="${BAMFILE%bam}bigwig"
hg19SIZE="2685511504"
COMMAND="--numberOfProcessors 10 --bam ${BAMFILE}"
COMMAND="${COMMAND} --outFileName $ {OUTFILE}"
COMMAND="${COMMAND} --outFileFormat bigwig"
COMMAND="${COMMAND} --binSize 5000 --normalizeUsing RPGC"
COMMAND="${COMMAND} --effectiveGenomeSize $ {hg19SIZE}"
COMMAND="${COMMAND} --scaleFactor 1.0 --extendReads 200"
COMMAND="${COMMAND} --minMappingQuality 30"
bamCoverage ${COMMAND}
```

If data for more than one replicate is available,
it is possible to merge replicates by first converting to bigwig as shown above  and then taking the mean across replicates using `bigwigCompare` from deeptools suite [[link]](https://github.com/deeptools/deepTools) for example like so:
```
#REPLICATE1 and REPLICATE2 are bigwig files
COMMAND="-b1 ${REPLICATE1} -b2 ${REPLICATE2}"
COMMAND="${COMMAND} -o ${OUTFILE} -of bigwig"
COMMAND="${COMMAND} --operation mean -bs 5000"
COMMAND="${COMMAND} -p 10 -v"
bigwigCompare ${COMMAND}
```

### Creating cooler files
Cooler offers a bunch of tools for converting Hi-C matrices from other formats
into cooler format, e.g. `hic2cool`. Check https://github.com/open2c/cooler