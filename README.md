# NLP Final Project Code
## Directory Structure: 
The Baseline folder has all the code necessary for obtaining any of the baseline results. 
The BERT folder has the BERT Jupyter Notebook. 
The BiLSTM code is in the notebook in the BiLSTM Folder. 
The Word2Vec code is in the same as the BiLSTM notebook. 
## Instillation Instructions
There is a requirements.txt file that can be used to recreate our environments for this project. Depending on what is run, some packages might not be used. You can use pip to install the packages from this requirements.txt file. 
## Preprocessing Code 
Preprocessing code is found in the "CS4650_Disaster_Tweet_Notebook" jupyter notebook under the BiLSTM folder. The jupyter cells are self-explanatory, and the subheadings explain what each cell does. Following their instuctions will generate new csv files depending on the required use.
## Baseline Code
If you go into the folder, you can run the code using the following command: (Note that this will default Logistic Regression classifier and unigram features.)
```
python main.py
```
To run the code with a different feature for example a full Ngram feature, go into the utils.py file, find the init method for the CustomFeature class, and change the n value to whatever Ngram feature you desire. Then proceed to run the following command: 
```
python main.py -f customized
```
Note that the code for Baseline results was largly adapted from the Homework code. 


## Main Results Code
BERT folder Contains Jupyter notebook to run BERT classification on tweets. Requires preprocessed csv BiLSTM folder. BiLSTM code is found in the "CS4650_Disaster_Tweet_Notebook" jupyter notebook under the BiLSTM folder

