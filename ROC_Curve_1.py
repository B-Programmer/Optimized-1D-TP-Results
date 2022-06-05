from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

y_true = np.array([0,1,0,1]) # true labels
y_probas = np.array([0.1, 0.4, 0.35, 0.8]) # predicted results
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=0)

# Print ROC curve
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)