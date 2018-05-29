import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence

INPUT_CONV_DIR = './conv_sample/'
INPUT_THRE_DIR = './thresh_data/'
INPUT_NUCC_DIR = './nucc/'
INPUT_TEST_DIR = './test/'
INPUT_TEXT_DIR = './text/'
INPUT_VALI_DIR = './validation/'

ENC_CONFIG = 'utf-8'

def get_all_files(in_dir):
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            yield os.path.join(root, file)
            
def read_document(path, encoding):
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()
    
def corpus_to_sentences(corpus, encoding):
    docs = [read_document(x, encoding) for x in corpus]
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        yield split_into_words(doc, name)
        
def trim_doc(doc):
    lines = doc.splitlines()
    valid_lines = []
    for line in lines:
        if line == '':
            continue
        if line.startswith('<doc') or line.startswith('</doc'):
            continue
        if "colspan" in line or "|||||" in line:
                continue
        if '＠'in line:
            continue
        if line.startswith('％'):
            continue
        if line.startswith('F'):
            line = line[5:]
        if line.startswith('＃'):
            line = line[1:]
        if line.startswith('M'):
            line = line[5:] 
        #print(line)
        valid_lines.append(line)
    
    return ''.join(valid_lines)

def split_into_words(doc, name=''):
    mecab = MeCab.Tagger("-Ochasen")
    valid_doc = trim_doc(doc)
    lines = mecab.parse(valid_doc).splitlines()
    
    words = []
    for line in lines:
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])

def run(input_dir, encoding):
    corpus = list(get_all_files(input_dir))
    sentences = list(corpus_to_sentences(corpus, encoding))
    return sentences


if __name__ == "__main__":
    sentences = run(INPUT_VALI_DIR, ENC_CONFIG)
#     valid_sentences = run(INPUT_VALI_DIR, ENC_CONFIG)
    print(len(sentences))
#     print(sentences[0].words)
#     print(len(valid_sentences))
#     for i in range(len(sentences)):
#     i = 0
#     print(sentences[i].tags)
#     print(sentences[i].words)