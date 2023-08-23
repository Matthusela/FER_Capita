# FER_Capita
## Facial Expression Recognition - Capita Project

Project is focussed on developing a FER model around the following dataset: 
     https://www.kaggle.com/datasets/debanga/facial-expression-recognition-challenge


dataset_cleaning.py deals with cleaning the kaggle dataset in preparation for modelling in FER_Model1.py, most of FER_Model1.py should work with a different data set as long the data is contained in a pandas dataframe with image pixel info stored in a numpy array/matrix with one image being represented per row. Obviously some formatting will have to take place with other data but hopefully wouldn't be too grueling to do that.

Currently model validation accuracy is peaking at 50-60% which is underwhelming but likely due to low quality of training images 48x48 pixels is not a sufficiently detailed image

###### In general I should remove RGB option, however, do not want to accidentally break rest of code in process so have chosen not to yet.

### Code Flow:
1) All code is in python
2) There are two sections to this project - the first 6 files which are concerned with loading facial expression data and fitting TensorFlow models to this data, the last file is the most exciting and runs fitted models live on video feed from computer camera and attempts to correctly identify emotion - if any - present in video feed.
The first 6 files are as follows:
1) simply defined two functions which calculates number of images and image dimension
2) takes the Kaggle CSV and converts it into a pandas dataframe - from there the string of pixel values is converted into a numerical matrix and the dataframe tidied up a small amount, the Kaggle CSV has no blanks or duplicates so there is no code in this file to deal with that
3) this file deals with the augmentation of certain classes and balances out the number of instances of each class - this is retrospective attempt to improve later model fitting
4) defines some important functions which take dataframe (augmented or not) and produces test and training sets, also does any formatting required for RGB and one-hot labelling
5) the most complicated file, all the TensorFlow work is done here - much of the file is to do with setting up CNN models for expression recognition. As did not have great success in training individual models the architecture is loosely randomised in a hope to find several decent models which can be used as an ensemble later on. Last function of this file sets up a process for generating an ensemble of models at different learning rates and trains each model on a different partition of the data using functions from previous file. if augmentation is implemented as it should be then this only ensures that there is a class balance in the entire dataframe - the sampling into test and training sets does not ensure a class balance - however the resulting discrepency in classes here is likely to be small.
6) Final file of this section just applies the final function of the previous file.




To download all the modules used and matching versions run following pip command from python command line in folder where project is kept

pip install -r requirementsFER.txt 
the requirements file works for me but when asking friends to check the code flow it has thrown up a few problems and some of the modules have to be imported manually - all the modules are fairly standard

### Instructions: 
1) Training images can be found from link above. Download the CSV from the link above and store this CSV in a folder with the rest of this GitHub's contents. ALways make sure working in this folder.
2) install the requirements from requirements file - fairly standard modules - run: (without quotes) "pip install -r requirementsFER.txt" in a python command line ideally anaconda command line.
3) Python files are numbered in the order of their sequence 
4) Files 1 to 6 should work hopefully by just running MakeModels.py  - can either be run in spyder or command line (see below for command line command).
5) RandomModels creates a bunch of CNNs to classify facial expression - individually these models are not particularly good so the hope is that running them as an ensemble will improve accuracy.
6) The MakeModelsCLI.py file should be run in anaconda command line - navigate to this project directory and run the command: python MakeModelsCLI.py --OneHot
7) Pressing enter should run the model - fitting the models will take several hours. After python MakeModelsCLI.py there are two options: can either add the --OneHot key like above and can also add --RGB key: adding --RGB will train the models to learn images in RGB rather than greyscale - not including this command will keep models in greyscale, --OneHot tag sets the class variables to be represented in a One hot format which should improve accuracy. AVOID --RGB PROBABLY BROKEN WHEN USING THESE MODELS WITH LATER FILES
8) As models take a long time to train there are some preloaded models in this repository stored in: AugmentModels - as GitHub has a file size limit can only upload a few files with few model parameters - more parameter models are too large - all tensorflow models should have .h5 extension
9) These models can be used in the TKinterScriptButton.py file - make sure models are stored in a folder and that folder name is correctly put into the script.
10) Running the TKinterScriptButton.py should launch a face detection window - press start detection and it should display front camera video feed!


### Progress
#### State at 05/07/23: 
Command line model training and command line launching of tkinter detection window running well. Initial hopes were to have the files configurable so that in command line one could specify whether the GreyScale images taken from Kaggle were converted into RGB format, and also whether to represent the target vector through one-hot representation or just as 0,1,2,3,etc. The choice of one-hot and RGB are the arguments of the Model_files(Bool,Bool) function - this is the function that begins model fitting and brings together model training process. While this function works with all Boolean arguments whether these arguments are actually causing the changes to be made has not yet been tested. Also do not think the Tkinter script works for models trained as RGB models. However this is not a huge issue as code does happily run for one-hot target variables and GreyScale images and this is the most logical choice: emotions are not ranked in order of their similarity therefore one-hot representation is sensible, and to me it seems like taking GreyScale images to RGB is unnecessary and cannot be properly justified. 

#### 25/07/23:
Restructured the dataset_cleaning file as was having issues with the emotion_dictionary when code was being executed multiple times without restarting kernel. Also introduced df_functions file which takes some functions off the dataset_cleaning file and tries to avoid the emotion_dictionary issues

#### 15/08/23: 
Finally dealt with class imbalance of emotions - had previously combined the anger and disgust emotion which had been the only change made to address the class imbalance issue. Knew that it was definitely necessary to address this issue as it became apparent at points that certain models were not learning any features from the images but just 'learnt' to guess the most common emotion in the dataset everytime and due to a large class imbalance this created a false sense of model accuracy improvement. Had previously tried some form of image augmentation which had just caused Spyder to crash so tried a very simple form of augmentation which was to just reflect/flip a number of images from the underrepresented classes. Brought all classes down to the mean of the classes (with anger and disgust combined) then from the overrepresented classes randomly sampled the mean number of images from each class and discarded other images, with underrepresented classes augmented (by flipping) a number of images to make up the difference with the mean. This work is Augmentation file and this brought all classes up to roughly ~6000 images. Noticed good improvement in model training after making this change.

#### 23/08/23:
Up to now the TKinter detection script had been taking the entire image of the video feed and trying to predict an emotion on this - this was clearly an error as the code would predict emotions when just faced with a blank wall. Restructured the code to use an algorithm provided by the CV2 library which detects faces. Using this algorithm can then isolate the patch of video feed which displays a face (code assumes there is just one face in the image but could be altered to detect emotions in multiple faces) and then the facial expression models can be run on just the patches of video feed containing faces. This has increased the accuracy of the video feed detection greatly.
