from sklearn import svm, datasets
from sklearn.model_selection import KFold
from sklearn import model_selection, metrics
from svm_test import loadData
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 高斯核函数最优threshold设置为0.7，线性核函数最优threshold设置为0.4


def showData(data):
    for data_single in data:
        print(data_single)


def plot_precision_recall_vs_threshold(accuracy, precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="recall")
    plt.plot(thresholds, accuracy[::], "r", label="accuracy")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


clf = svm.SVC(kernel='rbf', gamma=2, C=1)

# clf = svm.SVC(kernel='linear')

load_data = loadData.get_data_target('train_data.txt')
eyeHeight = load_data[0]
eyeWidth = load_data[1]
resultLabel = load_data[2]

scaler = preprocessing.StandardScaler()

eyeHeight = scaler.fit_transform(np.array(eyeHeight))


# 混淆矩阵
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(clf, eyeHeight, resultLabel, cv=10)

cm = confusion_matrix(resultLabel, y_train_pred)

print("normal confusion matrix is :\n", cm)

y_scores = cross_val_predict(clf, eyeHeight, resultLabel, method='decision_function', cv=10)

normal_accuracy = model_selection.cross_val_score(clf, np.array(eyeHeight), resultLabel, cv=10, scoring='accuracy')

normal_precision = model_selection.cross_val_score(clf, np.array(eyeHeight), resultLabel, cv=10, scoring='precision')

precisions, recalls, thresholds = precision_recall_curve(resultLabel, y_scores)

accuracy_score_list = []
for threshold in thresholds:
    y_scores_threshold = y_scores > threshold
    accuracy_score_list.append(accuracy_score(resultLabel, y_scores_threshold))

threshold = 1.31
y_scores_threshold = y_scores > threshold
accuracy_optimal = accuracy_score(resultLabel, y_scores_threshold)
precision_optimal = precision_score(resultLabel, y_scores_threshold)

plot_precision_recall_vs_threshold(accuracy_score_list, precisions, recalls, thresholds)

plt.plot(recalls, precisions)
plt.xlabel("recalls")
plt.ylabel("precisions")
plt.show()

print("normal accuracy mean is :", normal_accuracy.sum() / len(normal_accuracy))
print("normal precision mean is :", normal_precision.sum() / len(normal_precision))


cm = confusion_matrix(resultLabel, y_scores_threshold)
print("optimal confusion matrix is :\n", cm)
print("accuracy_optimal mean is :", accuracy_optimal)
print("precision_optimal mean is :", precision_optimal)
