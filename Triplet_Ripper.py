#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Name                       :   Triplet_Ripper.py
# Author                     :   K. SrePadmashiny
# Model Reviewer  :          :  
# Algo                       :  
# Date                       :  
# Purpose                    :                            
# Data                       :                    :   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ### *Load Library & Init*
import time
import pandas as pd
import pandas_profiling as pp
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
from collections import namedtuple
import string
import numpy as np
from keybert import KeyBERT
import IPython
from pyvis.network import Network
 

# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import spacy
import textacy
 
# ###  *Triplet Extraction Model*

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
 

# ### *Text PreProcess*
# text preprocessing functions
def clean_text(text):
    '''Make  lowercase, remove text in square brackets,remove links,remove punctuation, and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
 
def text_preprocessing(text):
    """  Cleaning and parsing the text. """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text
 
# ### *Triplet Ripper Functions*


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations
class KB():
    def __init__(self):
        self.relations = []
 
    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", tail"])


    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)
 

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)
 
    def print(self):
        K=0
        tempDF = pd.DataFrame(columns=['Subject_Property','Predicate','Object_Property'])
        for r in self.relations:
            #Subject_keywords = kw_model.extract_keywords(r.get("head"))
            #print(Subject_keywords)
            #if Subject_keywords[0][0] is not None :
            #    Subject_keywords = Subject_keywords[0][0]
            #else :
            #    Subject_keywords = ""         
            #Object_keywords = kw_model.extract_keywords(r.get("tail"))
            #print(r.get("tail"))
            #print(Object_keywords)
            #if len(Object_keywords) > 0:
            #        Object_keywords = Object_keywords[0][0]
            #else :
            #    Object_keywords = r.get("tail")   
            tempDF.loc[K, :] = [r.get("head"),
                                r.get("type"),
                                r.get("tail"),
                                " ",
                                " "]
                           
            K = K+1
        return tempDF
def from_small_text_to_kb(text, verbose=False):
    kb = KB()
 
    # Tokenizer text
    model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                           return_tensors='pt')
    if verbose:
        print(f"Num tokens: {len(model_inputs['input_ids'][0])}")
 
    # Generate
    gen_kwargs = {
        "max_length": 216,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": 5
    }

    generated_tokens = model.generate(
        **model_inputs,
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

 

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)
    return kb
 
# ### *Data Loading & Text Pre Processing*

df = pd.DataFrame()
df = pd.read_csv('data.csv', sep=',',low_memory=False, encoding = "ISO-8859-1"  , na_values=' ')

start = time.time()

df["summary_description"] = df["summary"] + ". " + df["description"]
df['summary_description'] = df['summary_description'].apply(str).apply(lambda x: text_preprocessing(x))
#df['summary'] = df['summary'].apply(str).apply(lambda x: text_preprocessing(x))
#df['description'] = df['description'].apply(str).apply(lambda x: text_preprocessing(x))

end = time.time()
print('Text PreProcesing %f .',  (end - start)/60)
df.columns


# ### *Looping Out Columns*
#tv_txt_cols = ['summary', 'description', 'Technical_Root_Cause']
tv_txt_cols = ['summary_description', 'Technical_Root_Cause'] 

start = time.time()

DF_neo4J = pd.DataFrame(columns=['Subject_Property','Predicate','Object_Property', 'Code', 'DataSource', 'TripletSource', "Y_N")

for tv_col in tv_txt_cols:
    for index, row in df.iterrows():
            kb = from_small_text_to_kb(str(df[tv_col][index]), verbose=False)
            tmp = kb.print()
            tmp['DataSource'] = "nex16"
            tmp['Code'] = str(df['issuekey'][index])
            tmp['TripletSource'] = tv_col           
            if tmp['Subject_Property'] in str(df['summary_description'][index]):
                tmp['Y_N'] = "Y"
            else:
                tmp['Y_N'] = "N"
            if tmp is not None :
                DF_neo4J = pd.concat([DF_neo4J, tmp] , ignore_index=True)
end = time.time()
print('Tripplet Extraction in  % Mins.', (end - start)/60)

DF_neo4J
DF_neo4J.to_csv('latest.csv', index=False)

#import pickle
#
#with open('C:\\KnowLedgeGraph\\nex\\triplet_latest.pkl', 'wb') as f:
#    pickle.dump(DF_neo4J, f)