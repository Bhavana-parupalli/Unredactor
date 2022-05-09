import unredactor as p3
import pytest
import pandas
import numpy

def test_extracting_unredacted_file():
    url="https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    dataframe=p3.extracting_unredacted_file(url)
    assert type(dataframe)==pandas.core.frame.DataFrame

def test_training_features():
    url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    dataframe = p3.extracting_unredacted_file(url)
    training_list, names_list=p3.training_features(dataframe, "training")
    assert type(training_list)==list and type(names_list)==list

def test_testing_features():
    url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    dataframe = p3.extracting_unredacted_file(url)
    testing_list, test_names = p3.testing_features(dataframe, "testing")
    assert type(testing_list) == list and type(test_names) == list

def test_model_training_and_testing():
    url = "https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv"
    dataframe = p3.extracting_unredacted_file(url)
    training_list, names_list = p3.training_features(dataframe, "training")
    testing_list, test_names = p3.testing_features(dataframe, "testing")
    predictions=p3.model_training_and_testing(training_list, testing_list, names_list, test_names)
    assert type(predictions)==numpy.ndarray



