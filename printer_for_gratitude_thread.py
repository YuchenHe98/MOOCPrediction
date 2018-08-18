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
import util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


thread_text_dic = {} #id - text
post_text_dic = {} #id - text
thread_post_table = {} #thread_id - list of post_id under that thread
user_table = {} #user_id - user-title
needs_truncation = {}
thread_intervened_time_dic = {}
database = ""

def is_gratitude_message(text):
    
    
    cleanr = re.compile('<.*?>')
    cleaned_text = re.sub(cleanr, '', text)

    if len(cleaned_text.split()) <= 15 and ('thank' in cleaned_text or 'Thank' in cleaned_text):
        return True
    
    else:
        return False
    
def work():
    
    # create the user table
    database = input("please input your database name: ")
    conn = util.create_connection(database)
    with conn:
        cur = conn.cursor() 
        cur.execute("SELECT id, user_title FROM user")
        rows = cur.fetchall()
        for each_user in rows:
            user_id = each_user[0]
            user_title = each_user[1]
            user_table[user_id] = user_title
            

    # create a database connection
    conn = util.create_connection(database)
    with conn:
        cur = conn.cursor() 
        course_id = input("please input your course_id: ")
        message_to_be_executed = "SELECT id, title, inst_replied, starter FROM thread WHERE courseid = \""
        message_to_be_executed += course_id
        message_to_be_executed += "\" ORDER BY posted_time"
        cur.execute(message_to_be_executed)
        threads = cur.fetchall()
        for each_thread in threads:
            text_to_be_written = ""
            thread_id = each_thread[0]
            thread_title = each_thread[1]
            thread_intervention = each_thread[2]
            thread_starter = each_thread[3]
            thread_intervened_time_dic[thread_id] = -1
            if(thread_starter in user_table and (user_table[thread_starter] == 'Instructor' or user_table[thread_starter] == 'Staff')):           
                continue
                
            text_to_be_written += thread_title
            
            # Detect the earliest intervention.
            if thread_intervention == 1:
                post_message = "SELECT user, post_time, post_text FROM post WHERE thread_id = \""
                post_message += thread_id
                post_message += "\" ORDER BY post_time"
                cur.execute(post_message)
                all_posts = cur.fetchall()
                for each_post in all_posts:
                    
                    poster = each_post[0]
                    post_time = each_post[1]
                    post_text = each_post[2]
                    
                    if poster in user_table and (user_table[poster] == 'Instructor' or user_table[poster] == 'Staff' or is_gratitude_message(post_text)) and thread_intervened_time_dic[thread_id] == -1:
                        
                        thread_intervened_time_dic[thread_id] = post_time
                        
                comment_message = "SELECT user, post_time, comment_text FROM comment WHERE thread_id = \""
                comment_message += thread_id
                comment_message += "\" ORDER BY post_time"
                cur.execute(comment_message)
                all_comments = cur.fetchall()
                for each_comment in all_comments:
                    
                    commenter = each_comment[0]
                    comment_time = each_comment[1]
                    comment_text = each_comment[2]
                    
                    if commenter in user_table and (user_table[commenter] == 'Instructor' or user_table[commenter] == 'Staff' or is_gratitude_message(comment_text)):
                        
                        if thread_intervened_time_dic[thread_id] == -1 or comment_time < thread_intervened_time_dic[thread_id]:
                            
                            thread_intervened_time_dic[thread_id] = comment_time
                            

            # Adding Texts.
            post_message = "SELECT id, post_text, user, post_time FROM post WHERE thread_id = \""
            post_message += thread_id
            post_message += "\" ORDER BY post_time"
            cur.execute(post_message)
            all_posts = cur.fetchall()
    
            for each_post in all_posts:
                post_id = each_post[0]
                post_text = each_post[1]
                poster = each_post[2]
                post_time = each_post[3]
                
                if thread_intervention == 1 and thread_intervened_time_dic[thread_id] != -1 and post_time >= thread_intervened_time_dic[thread_id]:
                    break
                else:
                    text_to_be_written += '\n\n'
                    text_to_be_written += post_text
                    comment_message = "SELECT id, comment_text, user, post_time FROM comment WHERE post_id = \""
                    comment_message += post_id
                    comment_message += "\" ORDER BY post_time"
                    cur.execute(comment_message)
                    all_comments = cur.fetchall()

                    for each_comment in all_comments:
                        comment_id = each_comment[0]
                        comment_text = each_comment[1]
                        commenter = each_comment[2]
                        comment_time = each_comment[3]
                        if thread_intervention == 1 and thread_intervened_time_dic[thread_id] != -1 and comment_time >= thread_intervened_time_dic[thread_id]:
                            break
                        else:
                            text_to_be_written += '\n\n'
                            text_to_be_written += comment_text

            file_name = str(thread_id) + '.txt'
            path = 'gratitude_text/' + course_id + "/" + file_name
            with open(path, 'w') as f:
                print(text_to_be_written, file = f)
                
        #print(nx.degree_centrality(graph))
                
if __name__ == "__main__":
    work()