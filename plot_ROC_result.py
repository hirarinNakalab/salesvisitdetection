import numpy as np
import matplotlib.pyplot as plt
from gensim import models
from sklearn import metrics
from scipy import spatial


LOAD_MODEL = './doc2vec_houhan.model'
HOUHAN_PATH = './text/HH/'

exec(open('./parseValidFiles.py').read())

model = models.Doc2Vec.load(LOAD_MODEL)

labels = []
values = []
for root, dirs, files in os.walk(HOUHAN_PATH):    
    for i in range(len(sentences)):
        sum_sim = 0
        vali_vector = model.infer_vector(sentences[i].words)
        for file in files:
            docpath = os.path.join(root, file)     
            docu_vector = model.docvecs[docpath]
            doc_dis = spatial.distance.cosine(vali_vector, docu_vector) 
            sum_sim += doc_dis
        ave_sim = sum_sim / len(files)
        if 'data' in sentences[i].tags[0]:
            label = 1
            labels.append(label)
        else:
            label = 2
            labels.append(label)
        values.append(ave_sim)
        
y = np.array(labels)
scores = np.array(values)

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' %auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)