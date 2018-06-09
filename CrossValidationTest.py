import os
import numpy as np
import matplotlib.pyplot as plt
from gensim import models
from sklearn import metrics
from scipy import spatial

CLVA_PATH = './clossvali/'
NUCC_PATH = './nucc/'
LOAD_MODEL = 'doc2vec_houhan(DBoW-QDup-WikiOnly).model'
model = models.Doc2Vec.load(LOAD_MODEL)

def search_dir(PATH):
    filepathes = []
    for root, dirs, files in os.walk(PATH): 
        for file in files:
            docpath = os.path.join(root, file) 
            filepathes.append(docpath)
    return filepathes

clva_files = search_dir(CLVA_PATH)
nucc_files = search_dir(NUCC_PATH)

k = 10
n_samples = len(clva_files)
fold_size = n_samples // k

for fold in range(k):
    comp_files = []
    vali_files = []
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[fold * fold_size:(fold+1)*fold_size] = True
    for i in range(len(test_mask)):
        if test_mask[i] == True:
            comp_files.append(clva_files[i])
        else:
            vali_files.append(clva_files[i])
    vali_files += nucc_files
    
    clva_sentences = []
    test_sentences = []
    for i in range(len(vali_sentences)):
        for j in range(len(comp_files)):
            if os.path.basename(vali_sentences[i].tags[0]) ==  \
            os.path.basename(comp_files[j]):
                clva_sentences.append(vali_sentences[i])
                
    for i in range(len(vali_sentences)):
        for j in range(len(vali_files)):
            if os.path.basename(vali_sentences[i].tags[0]) ==  \
            os.path.basename(vali_files[j]):
                test_sentences.append(vali_sentences[i])
    
    labels = []
    values = []
    for i in range(len(test_sentences)):
        sum_sim = 0
        if 'data' in test_sentences[i].tags[0]:
            label = 1
            labels.append(label)
        else:
            label = 2
            labels.append(label)
        test_vector = model.infer_vector(test_sentences[i].words)
        for j in range(len(clva_sentences)):
            clva_vector = model.infer_vector(clva_sentences[j].words)
            doc_dis = 1 - spatial.distance.cosine(test_vector, clva_vector) 
            sum_sim += doc_dis
        ave_sim = sum_sim / len(clva_sentences)
        values.append(ave_sim)
    y = np.array(labels)
    scores = np.array(values)
#     if fold == 1:
#     for i in range(len(test_sentences)):
#         print(test_sentences[i].tags, ":", scores[i], "->", y[i])
#     print()

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
plt.show()