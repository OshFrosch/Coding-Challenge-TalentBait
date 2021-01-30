import sys
import pickle

X_text = sys.argv[1:]

with open('model/classifier','rb') as file:
    clf = pickle.load(file)
    
with open('model/feature_extractor','rb') as file:
    tfidf = pickle.load(file)
    
X_vec = tfidf.transform(X_text)
y_pred = clf.predict(X_vec)

id_p = {0: 'none', 1:'tech', 2:'soft'}

for y in y_pred:
    print(id_p[y])