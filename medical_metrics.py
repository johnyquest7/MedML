#Importing required libraries

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from prettytable import PrettyTable

def med_metrics(real_labels, predicted_labels, X_test,model, classNames = ['Benign','Malignant']):
  # Function to calcuate and print commonly used metrics in the medical literature 

  confusion = confusion_matrix(real_labels, predicted_labels)
  TP = confusion[1,1] # true positive 
  TN = confusion[0,0] # true negatives
  FP = confusion[0,1] # false positives
  FN = confusion[1,0] # false negatives

  # Calculating metrics
  accuracy_init = accuracy_score(real_labels, predicted_labels)
  accuracy = format(accuracy_score(real_labels, predicted_labels), '.3f') 
  sensitivity = format(TP / float(TP+FN), '.3f')
  specificity = format(TN / float(TN+FP),'.3f')
  ppv = format(TP / float(TP+FP),'.3f')
  npv = format(TN / float(TN+ FN), '.3f')
  roc_score = format(roc_auc_score(real_labels, predicted_labels),'.3f')
  f1score = format((2*TP) / float(2*TP + FP + FN),'.3f')
  mathew_cor = format(matthews_corrcoef(real_labels, predicted_labels),'.3f')
  nnm_init = 1 / (1- accuracy_init)
  NNM = format(nnm_init,'.3f')
  FPR = format(FP / (FP + TN),'.3f')
  FDR = format(FP / (FP + TP),'.3f')
  FNR = format(FN / (FN + TP),'.3f')

  # Plotting confusion matrix
  plt.clf()
  plt.imshow(confusion, interpolation='nearest', cmap="Pastel1")
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  tick_marks = np.arange(len(classNames))
  plt.xticks(tick_marks, classNames, rotation=45)
  plt.yticks(tick_marks, classNames)
  s = [['TN','FP'], ['FN', 'TP']]
 
  for i in range(2):
      for j in range(2):
          plt.text(j,i, str(s[i][j])+" = "+str(confusion[i][j]))
  plt.show()

  # Creating Pretty Table 
  metrics_table = PrettyTable()
  metrics_table.field_names = ["Metrics", "Value"]
  metrics_table.add_row(["Accuracy", accuracy])
  metrics_table.add_row(["Sensitivity", sensitivity])
  metrics_table.add_row(["Specificity", specificity])
  metrics_table.add_row(["Positive predictive value", ppv])
  metrics_table.add_row(["Negative predictive value", npv])
  metrics_table.add_row(["AUC ROC", roc_score])
  metrics_table.add_row(["Mathews correlation coefficient", mathew_cor])
  metrics_table.add_row(["F1 score", f1score])
  metrics_table.add_row(["Number Needed to Mis-diagnose", NNM])
  metrics_table.add_row(["False positive rate", FPR])
  metrics_table.add_row(["False discovery rate", FDR])
  metrics_table.add_row(["False negative rate", FNR])
  print(metrics_table)

  # Draw receiver operating curve
  draw_AUC(real_labels, predicted_labels, X_test, model)

  return

def draw_AUC(real_labels, predicted_labels, X_test, model):

  # Funtion to draw receiver operating curve 
  model_roc_auc = roc_auc_score(real_labels, predicted_labels)
  fpr, tpr, thresholds = roc_curve(real_labels, model.predict_proba(X_test)[:,1])
  plt.figure()
  plt.plot(fpr, tpr, label='AUC for this model (area = %0.2f)' % model_roc_auc)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()
  return

