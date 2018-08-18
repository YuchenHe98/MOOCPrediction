#Please create a new folder named "all" to store all txt files of threads in the original folder.
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
import networkx as nx


user_table = {} #user_id - user-title

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

# undirected graph from one forum
def get_graph(forum_id, database):
    
    # create the user table
    graph = nx.Graph()
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        cur.execute("SELECT id, user_title FROM user")
        rows = cur.fetchall()
        for each_user in rows:
            user_id = each_user[0]
            user_title = each_user[1]
            user_table[user_id] = user_title
            
    #course id 1: GplkvRnqEeW9dA4X94-nLQ
    #course id 2: 
    #bug: user 3163925
    # create a database connection
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        message_to_be_executed = "SELECT id, thread_id, user, post_time FROM post WHERE forumid = \""
        message_to_be_executed += forum_id
        message_to_be_executed += "\" ORDER BY post_time"
        cur.execute(message_to_be_executed)
        rows = cur.fetchall()
        for each_post in rows:
            post_id = each_post[0]
            thread_id_of_this_post = each_post[1]
            post_user = each_post[2]
            post_time = each_post[3]
            
            # whether a post is created by an instructor or an user
            if(user_table[post_user] == 'Instructor' or user_table[post_user] == 'Staff'):
                continue
            
            # initiates the list of posts under a thread
            #if thread_id_of_this_post not in thread_post_table:
            #    thread_post_table[thread_id_of_this_post] = []
              
            # retrieve all comments under that post
            message_for_comment = "SELECT id, post_id, user, post_time FROM comment WHERE post_id = \""
            message_for_comment += post_id
            message_for_comment += "\" ORDER BY post_time"
            cur.execute(message_for_comment)
            all_comments = cur.fetchall()

            for each_comment in all_comments:
                comment_user = each_comment[2]
                comment_time = each_comment[3]
                # if there's an instructor commenting, truncate
                if(comment_user in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    continue
                if comment_user == post_user:
                    continue
                graph.add_edge(comment_user, post_user, weight = comment_time)
            
            message_for_thread = "SELECT id, starter FROM thread WHERE id = \""
            message_for_thread += thread_id_of_this_post
            message_for_thread += "\" ORDER BY id"
            cur.execute(message_for_thread)
            rows = cur.fetchall()
            for each_thread in rows:
                starter = each_thread[1]
                if(starter in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    break
                if post_user == starter:
                    continue
                graph.add_edge(post_user, starter, weight = post_time)
            
    return graph


# undirected graph from a list of forums
'''
def get_graph_from_list(forum_list, database):
    
    # create the user table
    graph = nx.Graph()
    conn = create_connection(database)
    limit_message = ''
    current_forum_num = 0
    for forum_id in forum_list:
        limit_message += 'forumid = '
        limit_message += forum_id
        if current_forum != len(forum_list) - 1:
            limit_message += ' OR '
        else:
            limit_message += ' '
    with conn:
        cur = conn.cursor() 
        cur.execute("SELECT id, user_title FROM user")
        rows = cur.fetchall()
        for each_user in rows:
            user_id = each_user[0]
            user_title = each_user[1]
            user_table[user_id] = user_title
            
    #course id 1: GplkvRnqEeW9dA4X94-nLQ
    #course id 2: 
    #bug: user 3163925
    # create a database connection
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        message_to_be_executed = 'SELECT id, thread_id, user FROM post WHERE '
        message_to_be_executed += forum_id
        message_to_be_executed += "ORDER BY post_time"
        cur.execute(message_to_be_executed)
        rows = cur.fetchall()
        for each_post in rows:
            post_id = each_post[0]
            thread_id_of_this_post = each_post[1]
            post_user = each_post[2]
            
            # whether a post is created by an instructor or an user
            if(user_table[post_user] == 'Instructor' or user_table[post_user] == 'Staff'):
                continue
            
            # initiates the list of posts under a thread
            #if thread_id_of_this_post not in thread_post_table:
            #    thread_post_table[thread_id_of_this_post] = []
              
            # retrieve all comments under that post
            message_for_comment = "SELECT id, post_id, user FROM comment WHERE post_id = \""
            message_for_comment += post_id
            message_for_comment += "\" ORDER BY post_time"
            cur.execute(message_for_comment)
            all_comments = cur.fetchall()

            for each_comment in all_comments:
                comment_user = each_comment[2]
                # if there's an instructor commenting, truncate
                if(comment_user in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    continue
                graph.add_edge(comment_user, post_user)
            
            message_for_thread = "SELECT id, starter FROM thread WHERE id = \""
            message_for_thread += thread_id_of_this_post
            message_for_thread += "\" ORDER BY id"
            cur.execute(message_for_thread)
            rows = cur.fetchall()
            for each_thread in rows:
                starter = each_thread[1]
                if(starter in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    break
                graph.add_edge(post_user, starter)
            
    return graph

'''
# directed graph from one forum
def get_digraph(forum_id, database):
    
    # create the user table
    graph = nx.DiGraph()
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        cur.execute("SELECT id, user_title FROM user")
        rows = cur.fetchall()
        for each_user in rows:
            user_id = each_user[0]
            user_title = each_user[1]
            user_table[user_id] = user_title
            
    #course id 1: GplkvRnqEeW9dA4X94-nLQ
    #course id 2: 
    #bug: user 3163925
    # create a database connection
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        message_to_be_executed = "SELECT id, thread_id, user, post_time FROM post WHERE forumid = \""
        message_to_be_executed += forum_id
        message_to_be_executed += "\" ORDER BY post_time"
        cur.execute(message_to_be_executed)
        rows = cur.fetchall()
        for each_post in rows:
            post_id = each_post[0]
            thread_id_of_this_post = each_post[1]
            post_user = each_post[2]
            post_time = each_post[3]
            
            # whether a post is created by an instructor or an user
            if(user_table[post_user] == 'Instructor' or user_table[post_user] == 'Staff'):
                continue
            
            # initiates the list of posts under a thread
            #if thread_id_of_this_post not in thread_post_table:
            #    thread_post_table[thread_id_of_this_post] = []
              
            # retrieve all comments under that post
            message_for_comment = "SELECT id, post_id, user, post_time FROM comment WHERE post_id = \""
            message_for_comment += post_id
            message_for_comment += "\" ORDER BY post_time"
            cur.execute(message_for_comment)
            all_comments = cur.fetchall()

            for each_comment in all_comments:
                comment_user = each_comment[2]
                comment_time = each_comment[3]
                # if there's an instructor commenting, truncate
                if(comment_user in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    continue
                if comment_user == post_user:
                    continue
                graph.add_edge(comment_user, post_user, weight = comment_time)
            
            message_for_thread = "SELECT id, starter FROM thread WHERE id = \""
            message_for_thread += thread_id_of_this_post
            message_for_thread += "\" ORDER BY id"
            cur.execute(message_for_thread)
            rows = cur.fetchall()
            for each_thread in rows:
                starter = each_thread[1]
                if(starter in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    break
                if post_user == starter:
                    continue
                graph.add_edge(post_user, starter, weight = post_time)
                
    return graph

def get_graph_from_list(forum_list, database):
    
    # create the user table
    graph = nx.Graph()
    conn = create_connection(database)
    limit_message = ''
    current_forum_num = 0
    for forum_id in forum_list:
        limit_message += 'forumid = '
        limit_message += forum_id
        if current_forum != len(forum_list) - 1:
            limit_message += ' OR '
        else:
            limit_message += ' '
    with conn:
        cur = conn.cursor() 
        cur.execute("SELECT id, user_title FROM user")
        rows = cur.fetchall()
        for each_user in rows:
            user_id = each_user[0]
            user_title = each_user[1]
            user_table[user_id] = user_title
            
    #course id 1: GplkvRnqEeW9dA4X94-nLQ
    #course id 2: 
    #bug: user 3163925
    # create a database connection
    conn = create_connection(database)
    with conn:
        cur = conn.cursor() 
        message_to_be_executed = 'SELECT id, thread_id, user FROM post WHERE '
        message_to_be_executed += forum_id
        message_to_be_executed += "ORDER BY post_time"
        cur.execute(message_to_be_executed)
        rows = cur.fetchall()
        for each_post in rows:
            post_id = each_post[0]
            thread_id_of_this_post = each_post[1]
            post_user = each_post[2]
            
            # whether a post is created by an instructor or an user
            if(user_table[post_user] == 'Instructor' or user_table[post_user] == 'Staff'):
                continue
            
            # initiates the list of posts under a thread
            #if thread_id_of_this_post not in thread_post_table:
            #    thread_post_table[thread_id_of_this_post] = []
              
            # retrieve all comments under that post
            message_for_comment = "SELECT id, post_id, user FROM comment WHERE post_id = \""
            message_for_comment += post_id
            message_for_comment += "\" ORDER BY post_time"
            cur.execute(message_for_comment)
            all_comments = cur.fetchall()

            for each_comment in all_comments:
                comment_user = each_comment[2]
                # if there's an instructor commenting, truncate
                if(comment_user in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    continue
                graph.add_edge(comment_user, post_user)
            
            message_for_thread = "SELECT id, starter FROM thread WHERE id = \""
            message_for_thread += thread_id_of_this_post
            message_for_thread += "\" ORDER BY id"
            cur.execute(message_for_thread)
            rows = cur.fetchall()
            for each_thread in rows:
                starter = each_thread[1]
                if(starter in user_table and (user_table[starter] == 'Instructor' or user_table[starter] == 'Staff' )):
                    break
                graph.add_edge(post_user, starter)
            
    return graph


def get_chronological_clustering_coef(time, graph, user):
    
    new_graph = nx.Graph()
    weight_edges_list = list(graph.edges_iter(data='weight', default=1))
    weight_edges_list.sort(key=lambda x: x[2])
    for weight_edge in weight_edges_list:
        if weight_edge[2] <= time:
            new_graph.add_edge(weight_edge[0], weight_edge[1])
    
    clu_dic = nx.clustering(new_graph)
    if user not in clu_dic:
        return 0
    else:
        return clu_dic[user]
    
    
    
def get_chronological_pgrank(time, digraph, user):
    
    new_digraph = nx.DiGraph()
    weight_edges_list = list(digraph.edges_iter(data='weight', default=1))
    weight_edges_list.sort(key=lambda x: x[2])
    for weight_edge in weight_edges_list:
        if weight_edge[2] <= time:
            new_digraph.add_edge(weight_edge[0], weight_edge[1])
    
    pg_dic = nx.pagerank(new_digraph)
    if user not in pg_dic:
        return 0
    else:
        return pg_dic[user]

    
                