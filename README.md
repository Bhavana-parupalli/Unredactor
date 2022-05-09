# cs5293sp22-project3
## Author: Bhavana Parupalli
## Email: parupallibhavana123@ou.edu
## Packages installed
```bash 
import pandas as pd
import requests
import io
import csv
import re
import glob
import io
import os
import pdb
import sys
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics
```
## unredactor.py
### get_entity(text)
The get entity function takes one text file at a time from doextraction() method and the sentences present inside the text are word tokenized and the entities of human names are appended by combining the entire name of a person using strings and removing spaces at the end of each name using rstrip. Followed by the the appended human names list is looped and the respective name in the text document is replaced with u"\u2588". Finally, returns the appended names list along with redacted document.
### doextraction(glob_text)
The doextraction method takes the text files path as argument, opens and reads each text file at a time and pass the text file to get_entity() function and for every text file it gets the redacted document and human names list. And then appends the human names and redacted document in to a new list and returns the new lists. Both get_entity() and doextracton() function are used to get the redacted documents along with names list, followed by creating a dataframe for train, validation, test and writing the extracted names and redacted sentences into the data frame and converting the dataframe to .tsv file.
### extracting_unredacted_file(url)
The extracting_unredacted_file function takes the raw unredacted.tsv file path present in the github as argument. Then after it downloads the content present inside the raw unredacted.tsv file. Finally, reads the .tsv file and is assigned to a dataframe and returns the dataframe. 
### training_features(df1, x)
The training_features function takes the data frame returned from the extracting_unredacted_file function along with the string named "training" as arguments. Then creates a dictionary as well as two lists along with a variable 'characters=0'. Followed by the data frame is looped through and if the second column in the data frame is equals to 'training' then from that the third column from that particular row which is name of the person is appended into the names_list. Moreover, the sentence present at the particular row in fourth column is assigned to a variable named text and the text, using a for loop the sentence is looped and if block character is found then the characters count is incremented. Then i created a dict with features, i considered only two features, one is character count of the name and the other is number of words present in the sentence and appending the dictionary into a list named training_list. Finally, returning the names_list along with the training_list.
### testing_features(df1, x)
The testing_features function takes the data frame along with the string named 'testing' as arguments. The testing_features() function is same as training_features() function. It involves same steps except in the training_features() function rows that contain training in the second column are taken, but in testing_features() function if the column contain testing then that row is considered.
### model_training_and_testing(train_features, test_features, train_names, test_names)
