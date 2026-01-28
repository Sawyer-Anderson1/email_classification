# imports
from flask import Flask, jsonify, request

from email import policy
from email.parser import Parser
import html2text
import io

import numpy as np
import pandas as pd

import nltk
import re
from string import punctuation

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

from urllib.parse import urlparse
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from transformers import pipeline
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.naive_bayes import BernoulliNB

app = Flask(__name__)

@app.route('/parse_eml_file', methods=['POST'])
def parse_eml_file():
    # read the http request (get the data and decode from utf-8 )
    raw_eml_string_data = request.data.decode('utf-8')

    # then parse the raw string to get subject, from, and for body
    msg = Parser(policy=policy.default).parse(io.StringIO(raw_eml_string_data))

    # get the subject and from
    subject = msg['subject']
    sender = msg['from']

    # then get body and account for if its multipart or not
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                break
            elif content_type =='text/html':
                html_content = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                body = html2text.html2text(html_content)
    else:
        body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')

    # then create the dictionary that will become the key features to be used in the AI model
    whole_message = "From: " + sender + "\nSubject: " + subject + "\n\n" + body
    email_dict = {"Message": whole_message, "Sender": sender, "Subject": subject, "Body": body}

    input = pd.DataFrame(email_dict)

    # call the function(s) that prepare and send the data into the model
    y_pred, y_prob = prep_data(input)
    
    # return the y_pred and y_prob
    return jsonify({'prediction': y_pred.tolist(), 'probability': y_prob.tolist()})

def prep_data(df):
    exclude_punctuation = set(punctuation)
    stp_words = nltk.corpus.stopwords.words('english')

    # Extract some more features
    df['Sentence_count'] = 0
    df['Word_count'] = 0

    index = 0
    for message in df['Message']:
        sentence_count, _ = sentence_metrics([message])
        word_count, _ = word_email_metrics([message], sentence_count)

        df.at[index, 'Sentence_count'] = sentence_count
        df.at[index, 'Word_count'] = word_count

        index += 1

    # cleaning subject and body
    # use .split() instead of nltk.word_tokenize to preserve urls (but will later replace url with a token 'url')
    # new features that will be created
    # create the lemmatizer object
    lemmatizer = WordNetLemmatizer()

    df['urls_found'] = 0

    token_pattern = re.compile(r'https?://\S+|www\.\S+|\S+')

    # columns to clean
    columns_to_clean = ['Subject', 'Body']

    for col in columns_to_clean:
        for idx, raw_text in df[col].fillna('').astype(str).items():
            text = raw_text.lower()

            # get tokens
            tokens = token_pattern.findall(text)

            url_count = sum(1 for t in tokens if is_url(t))
            df.at[idx, 'urls_found'] = url_count

            # substitute urls with 'url'
            tokens = ['url' if is_url(t) else t for t in tokens]

            # do PoS tagging-lemmatization
            # get Pos tag
            tagged = nltk.pos_tag(tokens)
            lemma_tokens = []

            for word, tag in tagged:
                new_tag = pos_tagger(tag)
                lemma = lemmatizer.lemmatize(word, new_tag)
                lemma_tokens.append(lemma)

            # remove stop words and punctuation
            text_clean = [
                word for word in lemma_tokens 
                if word and (word not in stp_words) and (not all(ch in exclude_punctuation for ch in word))
            ]

            df.at[idx, col] = ' '.join(text_clean).strip()

    # sentiment analysis   
    classifier = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
    df['Sentiment'] = classifier(df['Message'].tolist())

    for idx, sentiment_dict in df['Sentiment'].items():
        df.at[idx, 'Sentiment'] = getAnalysis(sentiment_dict)

    # then call the function that vectorizes and sends in the data
    y_pred, y_prob = pass_to_model(df)
    return y_pred, y_prob

def sentence_metrics(sentence_array):
    total_sentences = 0
    total_messages = len(sentence_array)

    for text in sentence_array:
        sentence_in_text = nltk.sent_tokenize(text)
        total_sentences += len(sentence_in_text)
    
    avg_sentence_per_message = total_sentences / total_messages

    return total_sentences, avg_sentence_per_message

def word_email_metrics(sentence_array, total_sentences):
    total_words = 0
    total_sentences = total_sentences

    for text in sentence_array:
        words_in_text = nltk.word_tokenize(text)

        total_words += len(words_in_text)

    avg_words_per_sentence = total_words / total_sentences
    return total_words, avg_words_per_sentence 

def is_url(url_string):
    try:
        result = urlparse(url_string)

        # check for url schem and network location
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def getAnalysis(sentiment_result_dict):
    if sentiment_result_dict['label'] == 'Very Negative':
        return 0
    elif sentiment_result_dict['label'] == 'Negative':
        return 1
    elif sentiment_result_dict['label'] == 'Neutral':
        return 2
    elif sentiment_result_dict['label'] == 'Positive':
        return 3
    elif sentiment_result_dict['label'] == 'Very Positive':
        return 4
    
def pass_to_model(df):
    X_num = df.drop(columns=['Category']).select_dtypes(include=np.number)
    X_text = "Sender: " + df['Sender'].astype(str) + " Subject: " + df['Subject'].astype(str) + " Email Body: " + df['Body'].astype(str)
    y = df['Category']

    # load the tfidf
    tfidf = joblib.load("tfidf_vectorizer.joblib")

    # fit onto vectorizor
    X_tfidf = tfidf.transform(X_text)

    # load the scaler and scale numeric data
    scaler = joblib.load("standard_scaler.joblib")

    # fit onto scaler
    X_num_sc = scaler.transform(X_text)

    # combine 
    X = hstack([X_tfidf, X_num_sc])

    # load the model
    model = joblib.load("nb_smote_model.joblib")

    # then get the prediction
    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)

    return y_pred, y_prob