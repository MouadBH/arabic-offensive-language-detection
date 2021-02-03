import re
import string
import numpy as np
import pandas as pd
import nltk

# import spacy
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.utils import shuffle

from emoji import UNICODE_EMOJI
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

arb_stopwords = set(nltk.corpus.stopwords.words("arabic"))
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

ar_stopwords_file = open('lib/ar_stopword.txt', 'r', encoding='utf-8') 
ar_stopword = ar_stopwords_file.read().splitlines()

################## Data Processing for Arabic Text

def take_a_shower(line):
    if (isinstance(line, float)):
        return None
    line.replace('\n', ' ')
    line = remove_emails(line)
    line = remove_urls(line)
    nline = [w if '@' not in w else 'USERIDX' for w in line.split()]
    line = ' '.join(nline)
    line = re.sub(r'[a-zA-Z?]', '',line).strip() # remove no arabic characters
    line = line.replace('RT', '').replace('<LF>', '').replace('<br />','').replace('&quot;', '').replace('<url>', '').replace('@User', '').replace('USERIDX', '')

    # add spaces between punc,
    line = line.translate(str.maketrans({key: " {0} ".format(key) for key in punctuations_list}))

    # then remove punc,
    translator = str.maketrans('', '', punctuations_list)
    line = line.translate(translator)
    
    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    line = re.sub(p_longation, subst, line)

    
    # remove stopwords
    line = remove_stopwords(line)
    line = extra_remove_ar_stopword(line, ar_stopword)
    
    # normalize & remove diacritis
    line=remove_diacritics(normalize_arabic(line))
    
    #replace number
    nline = [word if not hasDigits(word) else '<NUM>' for word in line.split()]
    line = ' '.join(nline)
    return line

def remove_urls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return text

def remove_emails(text):
    text = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", "",  text, flags=re.MULTILINE)
    return text

def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space_with_emojis(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def removeConsecutiveSameNum(v):
    st = []
    lines=[]

    # Start traversing the sequence
    for i in range(len(v)):

        # Push the current string if the stack
        # is empty
        if (len(st) == 0):
            st.append(v[i])
            lines.append(v[i])
        else:
            Str = st[-1]

            # compare the current string with stack top
            # if equal, pop the top
            if (Str == v[i] and Str == '<NUM>'):
                st.pop()

                # Otherwise push the current string
            else:
                lines.append(v[i])
                st.pop()
                # st.append(v[i])

                # Return stack size
    return lines

def hasDigits(s):
    return any( 48 <= ord(char) <= 57  or 1632 <= ord(char) <= 1641 for char in s)


#Text Normalization
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_stopwords(text):
    filtered_sentence = [w for w in text.split() if not w in arb_stopwords]
    return ' '.join(filtered_sentence)

def extra_remove_ar_stopword(text, ar_stopwords):
    text_tokens = word_tokenize(text)
    return " ".join([word for word in text_tokens if not word in ar_stopwords])


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


# Clean/Normalize Arabic Text Based on aravec
def clean_content_aravec(line):

    if (isinstance(line, float)):
        return None
    line.replace('\n', ' ')
    line = remove_emails(line)
    line = remove_urls(line)
    line = line.replace('@User', '').replace('RT', '').replace('<LF>', '')

    # Check if # or @ is there with word

    # add spaces between punc,
    line = line.translate(str.maketrans({key: " {0} ".format(key) for key in punctuations_list}))

    # then remove punc,
    # line = line.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    translator = str.maketrans('', '', punctuations_list)
    line = line.translate(translator)

    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    line = re.sub(p_tashkeel, "", line)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    line = re.sub(p_longation, subst, line)

    line = line.replace('وو', 'و')
    line = line.replace('يي', 'ي')
    line = line.replace('اا', 'ا')

    for i in range(0, len(search)):
        line = line.replace(search[i], replace[i])

    # trim
    line = line.strip()

    return line


def map_labels_off(lab):
    label_maps = {"Non-Offensive": "NOT_OFF", "Offensive": "OFF"}
    if lab in label_maps:
        return label_maps[lab]
    else:
        return lab