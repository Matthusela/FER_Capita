# FER_Capita
Facial Expression Recognition - Capita Project

Project is initially focussed on developing a FER model around the following dataset: 
     https://www.kaggle.com/datasets/debanga/facial-expression-recognition-challenge


dataset_cleaning.py deals with cleaning the kaggle dataset in preparation for modelling in FER_Model1.py, most of FER_Model1.py should work with a different data set as long the data is contained in a pandas dataframe with image pixel info stored in a numpy array/matrix with one image being represented per row. Obviously some formatting will have to take place with other data but hopefully wouldn't be too grueling to do that.

Currently model validation accuracy is peaking at 50% which is underwhelming

To download all the modules used and matching versions run following pip command from python command line in folder where project is kept
pip install -r requirementsFER.txt 
