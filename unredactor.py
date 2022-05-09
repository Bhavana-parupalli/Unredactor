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

def get_entity(text):
    human_names=[]
    """Prints the entity inside of the text."""
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                person_name=""
                for c in chunk.leaves():
                    person_name=person_name + c[0]
                    person_name=person_name + " "
                person_name=person_name.rstrip()
                human_names.append(person_name)
    doc=text
    for i in human_names:
        doc=doc.replace(i,u"\u2588" * len(i))
    return human_names, doc

def doextraction(glob_text):
    names = []
    docs = []
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            human_names, doc = get_entity(text)
            names.append(human_names)
            docs.append(doc)
    return names, docs

def get_entity(text):
    human_names=[]
    """Prints the entity inside of the text."""
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                #print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                person_name=""
                for c in chunk.leaves():
                    person_name=person_name + c[0]
                    person_name=person_name + " "
                person_name=person_name.rstrip()
                human_names.append(person_name)
    return human_names

def extracting_unredacted_file(url):
    download = requests.get(url).content

    df1 = pd.read_csv(url, delimiter='\t', encoding='utf-8', header=None, on_bad_lines='skip', quoting=csv.QUOTE_NONE)
    return df1

def training_features(df1, x):
    training_dict={}
    training_list=[]
    names_list=[]
    for i in range(0, df1.index.size):
        if df1.iloc[i][1]==x:
            characters=0
            name=df1.iloc[i][2]
            text=df1.iloc[i][3]
            word_count=len(word_tokenize(text))
            for k in text:
                if k=='█':
                    characters+=1
            names_list.append(name)
            training_dict={'Number_of_characters_in_Person_name' : characters,
                            'Number_of_words': word_count}
            training_list.append(training_dict)
    return training_list, names_list

def testing_features(df1, x):
    testing_dict={}
    testing_list=[]
    test_names=[]
    for i in range(0, df1.index.size):
        if df1.iloc[i][1]==x:
            characters=0
            name=df1.iloc[i][2]
            text = df1.iloc[i][3]
            word_count=len(word_tokenize(text))
            for k in text:
                if k=='█':
                    characters+=1
            test_names.append(name)
            testing_dict={'Number_of_characters_in_Person_name' : characters,
                          'Number_of_words': word_count}
            testing_list.append(testing_dict)
    return testing_list, test_names


def model_training_and_testing(train_features, test_features, train_names, test_names):
    feature_array = []
    target_variable = []
    for dict in train_features:
        feature_array.append(dict)
    for i in train_names:
        target_variable.append(i)

    DV = DictVectorizer(sparse=False)
    feature_array = DV.fit_transform(feature_array)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(feature_array, target_variable)

    f_array = []
    test = []
    for dict in test_features:
        f_array.append(dict)
    for i in test_names:
        test.append(i)

    DV = DictVectorizer(sparse=False)
    f = DV.fit_transform(f_array)

    predict = model.predict(f)

    print("Precision:", precision_score(test, predict, average="micro"))
    print("Recall:", recall_score(test, predict, average="micro"))
    print("f1-score:", f1_score(test, predict, average="micro"))


def main():

    df1=extracting_unredacted_file("https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv")
    train_features, train_names = training_features(df1, 'training')
    test_features, test_names = testing_features(df1, 'testing')
    model_training_and_testing(train_features, test_features, train_names, test_names)

main()
