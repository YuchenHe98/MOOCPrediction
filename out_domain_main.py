#Please create a new folder named "all" to store all txt files of threads in the original folder.
from bs4 import BeautifulSoup, Tag
from collections import OrderedDict
from pathlib import Path
import os
import sqlite3
from sqlite3 import Error
import argparse
import nltk
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import metrics 
from sklearn.metrics import classification_report
from graph_constructor import get_graph
import networkx as nx
import matplotlib.pyplot as plt
from one_course_vector import get_vector_for_one_course

course_id_list = []
course_vector_list = []

#html-css-javascript LgWwihnoEeWDtQoum3sFeQ
#python_database eQJvsjn9EeWJaxK5AT4frw
#python_network Y4DUPDpQEeWO-Qq6rEZAow
#python 7A1yFTaREeWWBQrVFXqd1w
#hybrid-mobile-development -gcU5xn4EeWwrBKfKrqlSQ
#machine-learning Gtv4Xb1-EeS-ViIACwYKVQ
#learning-how-to-learn GdeNrll1EeSROyIACtiVvg
#angular-js 52blABnqEeW9dA4X94-nLQ
#server-side-development ngZrURn5EeWwrBKfKrqlSQ

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def read_forum_list(course_id):

    forum_id_list = []
    path = course_id + "/" + "forum.txt"
    file = open(path, "r")
    num_forum = int(file.readline())
    for i in range (0, num_forum):
        forum_id = file.readline()
        forum_id = forum_id.rstrip(os.linesep)
        forum_id_list.append(forum_id)
        
    return forum_id_list


def main():
    
    course_vector_list = []
    course_result_list = []
    course_index_list = []
    index = 0
    database = input("Please input your database name: ")
    course_id_list = []
    path = "courseidlist.txt"
    file = open(path, "r")
    num_course = int(file.readline())
    for i in range (0, num_course):
        course_id = file.readline()
        course_id = course_id.rstrip(os.linesep)
        course_id_list.append(course_id)
        course_index_list.append(index)
        index += 1
    
    course_index_list = np.asarray(course_index_list)

    kf = KFold(int(num_course), n_folds = 2, shuffle=False, random_state = 18)
    print(kf)
    #print(to_train)
    for train_index, test_index in kf:
        
        train_set, test_set = course_index_list[train_index], course_index_list[test_index]
        
    for course_id in course_id_list:
        forum_id_list = read_forum_list(course_id)
        print("now")
        target_vector_list = get_vector_for_one_course(database, course_id, forum_id_list)
        course_vector_list.append(target_vector_list[0])
        course_result_list.append(target_vector_list[3])
        print(target_vector_list[3])

       

        '''
        sys_mat_train, sys_mat_test = course_vector_list[train_index], course_vector_list[test_index]
        sys_result_train, sys_result_test = course_result_list[train_index], course_vector_list[test_index]
        '''
        
    vec_train = []
    for index in train_set:
        each_mat = course_vector_list[index]
        for each_vec in each_mat:
            vec_train.append(each_vec)
            
    result_train = []
    for index in train_set:
        each_mat = course_result_list[index]
        for each_vec in each_mat:
            result_train.append(each_vec)
            
    vec_test = []
    for index in test_set:
        each_mat = course_vector_list[index]
        for each_vec in each_mat:
            vec_test.append(each_vec)
            
    result_test = []
    for index in test_set:
        each_mat = course_result_list[index]
        for each_vec in each_mat:
            result_test.append(each_vec)
            
    #LogReg = LogisticRegression(class_weight = 'balanced')
    #LogReg.fit(x_train, y_train)
    #y_pred = LogReg.predict(x_test)
    
    vec_train = np.asarray(vec_train)
    result_train = np.asarray(result_train)
    #print(vec_train.shape(0))
    #print(vec_train.shape(1))
    #print(result_train.shape(0))

    vec_test = np.asarray(vec_test)
    result_test = np.asarray(result_test)

    LogReg = LogisticRegression(class_weight = 'balanced')
    LogReg.fit(vec_train, result_train)
    result_pred = LogReg.predict(vec_test)
    with open('EDM.txt', 'w') as f:
        print(classification_report(result_test, result_pred), file = f)
    print(classification_report(result_test, presult_pred))
    
    


if __name__ == "__main__":
    main()
