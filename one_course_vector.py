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

def tf_idf(docs, queries, tokenizer):
    """
    performs TF-IDF vectorization for documents and queries
    Parameters
        ----------
        docs : list
            list of documents
        queries : list
            list of queries
        tokenizer : custom tokenizer function
    Returns
    -------
    tfs : sparse array,
        tfidf vectors for documents. Each row corresponds to a document.
    tfs_query: sparse array,
        tfidf vectors for queries. Each row corresponds to a query.
    dictionary: list
        sorted dictionary
    """
    model = str.maketrans(dict.fromkeys(string.punctuation))
    processed_docs = [d.lower().translate(model) for d in docs]
    tfidf = TfidfVectorizer(stop_words='english', tokenizer=tokenizer)
    tfs = tfidf.fit_transform(processed_docs)
    tfs_query = tfidf.transform(queries)
    return tfs, tfs_query, tfidf


def tokenize_text(docs):
    """
    custom tokenization function given a list of documents
    Parameters
        ----------
        docs : string
            a document
    Returns
    -------
    stems : list
        list of tokens
    """

    text = ''
    for d in docs:
        text += '' + d
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems



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


def select_info(conn):

    cur = conn.cursor()
    cur.execute("SELECT instReplied, title, content, threadId FROM thread")
 
    rows = cur.fetchall()
    return rows


def get_vectors(texts):
    descriptions = []
    descriptions.append('')
    vec_docs, vec_queries, tfidf_model = tf_idf(texts, descriptions, tokenize_text)
    return vec_docs


def get_vector_for_one_course(database, course_id, forum_id_list):
    
    list_to_return = []
    #
    user_table = {}
    # create a database connection
    conn = create_connection(database)
            
    texts = []
    ids = []
    isreplied= []
    thread_id_dic = {}
    id_text_length_dic ={}
    comment_dic = {}
    post_dic = {}
    feature_dict = {}
    forum_type_dic = {}
    forum_graph_dic = {}
    post_set_dic = {}
    num_of_post_dic = {}
    num_of_comment_dic = {}
    num_of_sen_dic = {}
    num_of_comment_for_post_dic = {}
    num_of_url_dic = {}
    num_of_timeref_dic = {}
    num_of_votes_dic = {}
    is_replied = {}
    id_to_index = {}
    forum_of_thread_dic = {}
    starter_dic = {}
    locker = {}
    
    
    num_forum = len(forum_id_list)
    i = 0
    for forum_id in forum_id_list:
        
        forum_type_dic[forum_id] = i
        i += 1
        forum_graph_dic[forum_id] = get_graph(forum_id, database)
    
    print(forum_id_list)
    limit_message = "WHERE courseid = \'"
    limit_message += course_id
    limit_message += "\'"
    if int(num_forum) >= 1:
        limit_message += " AND (forumid = \'"
        limit_message += forum_id_list[0]
        limit_message += "\'"
    
    for i in range(1, int(len(forum_id_list))):
        limit_message += " OR forumid = \'"
        limit_message += forum_id_list[i]
        limit_message += "\'"
    
    if int(num_forum) >= 1:
        limit_message += ")"
    
    
    
    with conn:
        cur = conn.cursor()
        thread_message = "SELECT id, inst_replied, forumid, starter FROM thread "
        thread_message += limit_message
        thread_message += "ORDER BY posted_time"
        print(thread_message)
        cur.execute(thread_message)
        rows = cur.fetchall()
        for each_thread in rows:
            starter = each_thread[3]
            if starter in user_table and (user_table[starter] == 'instructor' or user_table[starter] == 'staff'):
                continue
            threadid = each_thread[0]
            if threadid in locker and locker[threadid] == True:
                continue
            locker[threadid] = False
            ids.append(each_thread[0])
            forum_of_thread_dic[threadid] = each_thread[2]
            is_replied[threadid] = each_thread[1]
            post_set_dic[threadid] = []
            num_of_post_dic[threadid] = 0
            num_of_comment_dic[threadid] = 0
            num_of_votes_dic[threadid] = 0
            starter_dic[threadid] = each_thread[3]
            isreplied.append(each_thread[1])
        
        post_message = "SELECT thread_id,id, votes, user FROM post "
        post_message += limit_message
        post_message += 'ORDER BY post_time'
        cur.execute(post_message)
        rows = cur.fetchall()
        for each_post in rows:
            starter = each_post[3]
            if starter in user_table and (user_table[starter] == 'instructor' or user_table[starter] == 'staff'):
                locker[each_post[0]] = True
                locker[each_post[1]] = True
                continue
            if each_post[0] in locker and locker[each_post[0]] == True:
                continue
            locker[threadid] = False
            post_set_dic[each_post[0]].append(each_post[1])
            num_of_post_dic[each_post[0]] = num_of_post_dic[each_post[0]] + 1
            num_of_comment_for_post_dic[each_post[1]] = 0
            num_of_votes_dic[each_post[0]] = num_of_votes_dic[each_post[0]] + each_post[2]

        
        comment_message = "SELECT thread_id, post_id, user FROM comment "
        comment_message += limit_message
        post_message += 'ORDER BY post_time'

        cur.execute(comment_message)
        rows = cur.fetchall()
        for each_comment in rows:
            starter = each_comment[2]
            if starter in user_table and (user_table[starter] == 'instructor' or user_table[starter] == 'staff'):
                locker[each_post[0]] = True
                locker[each_post[1]] = True
                continue
            if (each_post[0] in locker and locker[each_post[0]] == True) or (each_post[1] in locker and locker[each_post[1]] == True):
                continue
            num_of_comment_dic[each_comment[0]] = num_of_comment_dic[each_comment[0]] + 1
            num_of_comment_for_post_dic[each_post[1]] =             num_of_comment_for_post_dic[each_post[1]] + 1

    for oneid in ids:
        filename = course_id + "/" + str(oneid) + '.txt'
        f = open(filename, 'r', errors='ignore')
        content = f.read()
        
        num_url = content.count('a href')             
        num_timeref = content.count('<TIMEREF>')
        num_sen = content.count('.')
        num_of_url_dic[oneid] = num_url
        num_of_timeref_dic[oneid] = num_timeref
        num_of_sen_dic[oneid] = num_sen
        
        soup = BeautifulSoup(content, "lxml")

        for tag in soup.find_all('code'):
            tag.replaceWith('')
            
        content = re.sub(r'^https?:\/\/.*[\r\n]*', '', content, flags=re.MULTILINE)

        
        content = soup.get_text()
        TAG_RE = re.compile(r'<[^>]+>')
        content = TAG_RE.sub('', content)

        texts.append(content)

        id_text_length_dic[oneid] = len(content.split())
    
    i = 0
    for oneid in ids:
        id_to_index[oneid] = i
        i = i + 1
        
    #rubbish
    vectors = get_vectors(texts)
    x = vectors.toarray()
    
    x = np.asarray(x)      
    y = np.asarray(isreplied)
    data_length = len(ids)

    '''
    kf = KFold(data_length, n_folds = 5, shuffle=False, random_state = 18)
    print(kf)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    LogReg = LogisticRegression(class_weight = 'balanced')
    LogReg.fit(x_train, y_train)
    y_pred = LogReg.predict(x_test)
    with open('tfidf.txt', 'w') as f:
        print(classification_report(y_test, y_pred), file = f)
    print(classification_report(y_test, y_pred))
    '''
    

    tmp_vectors = []
    training_vectors = []

    
    x = vectors.toarray()
    # EDM 15
    for oneid in ids:
        index = id_to_index[oneid]
        origin_list = x[index].tolist()
        li = []
        forum_feature = forum_type_dic[forum_of_thread_dic[oneid]]
        li.append(forum_feature)
        numpost = num_of_post_dic[oneid]
        li.append(numpost)
        numcomment = num_of_comment_dic[oneid]
        li.append(numcomment)
        li.append(numpost + numcomment)
        # will add the average # of comments per post
        summ = 0
        for postid in post_set_dic[oneid]:
            summ = summ + num_of_comment_for_post_dic[postid]
        avr = summ / len(post_set_dic[oneid])
        li.append(avr)
        numurl = num_of_url_dic[oneid]
        li.append(numurl)
        numsen = num_of_sen_dic[oneid]
        li.append(numsen)
        numvotes = num_of_votes_dic[oneid]
        li.append(numvotes)
        numtimeref = num_of_timeref_dic[oneid]
        li.append(numtimeref)
        new_list = origin_list + li
        
        tmp_vectors.append(new_list)

        
    list_to_return.append(tmp_vectors)    
  

    sna_only_vectors = []
    all_vectors = []
    
    
    
    for index in range(data_length):
        li = []
        oneid = ids[index]
        origin_list = x[index].tolist()
        starter = starter_dic[oneid]
        forum_id = forum_of_thread_dic[oneid]
        forum_index = forum_type_dic[forum_id]
        for i in range(0, 4):
            current_forum_index = forum_index - i
            if current_forum_index <= 0:
                li.append(0)
                continue
            else:
                g = forum_graph_dic[forum_id_list[current_forum_index]]
                cent = nx.degree_centrality(g)
        
                #modified
                if starter in cent:
                #   x_as_cen.append(cent[starter])
                    li.append(cent[starter] * 50)
                else:
                    #x_as_cen.append(0)
                    li.append(0)
            
        new_list = origin_list + li
        
        sna_only_vectors.append(li)
        all_vectors.append(new_list)
        
    list_to_return.append(sna_only_vectors)
    list_to_return.append(all_vectors)
    list_to_return.append(isreplied)
    return list_to_return

    