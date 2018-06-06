import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim import models
from sklearn import metrics
from scipy import spatial


LOAD_MODEL = 'doc2vec_houhan(saigen-size300).model'
HOUHAN_PATH = './text/HH/'

# exec(open('./parseValidFiles.py').read())

model = models.Doc2Vec.load(LOAD_MODEL)

labels = []
values = []
for root, dirs, files in os.walk(HOUHAN_PATH):    
    for i in range(len(vali_sentences)):
        sum_sim = 0
        vali_vector = model.infer_vector(vali_sentences[i].words)
        for file in files:
            docpath = os.path.join(root, file)     
            docu_vector = model.docvecs[docpath]
            doc_dis = 1 - spatial.distance.cosine(vali_vector, docu_vector) 
            sum_sim += doc_dis
        ave_sim = sum_sim / len(files)
        if 'data' in vali_sentences[i].tags[0]:
            label = 1
            labels.append(label)
        else:
            label = 2
            labels.append(label)
        values.append(ave_sim)
        
y = np.array(labels)
scores = np.array(values)

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2, drop_intermediate=False)
print(thresholds)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='cosine similarity of PV-DM (area = %.2f)' %auc)
plt.legend()
plt.title('Sales Visit - Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
plt.savefig('roc.png')