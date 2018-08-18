import os
import argparse
import nltk
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import seaborn as sb
from bs4 import BeautifulSoup, Tag


def preprocess_text(raw_text):
    
    soup = BeautifulSoup(raw_text, "lxml")

    for tag in soup.find_all('code'):
        tag.replaceWith('')
        
    for tag in soup.find_all('a'):
        tag.replaceWith('')

    #content = re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)


    content = soup.get_text()
    TAG_RE = re.compile(r'<[^>]+>')
    content = TAG_RE.sub('', content)
    return content
