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
from sklearn.metrics import confusion_matrix


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

# clf = svm.SVC(kernel='rbf', gamma=2, C=1)

clf = svm.LinearSVC()

load_data = loadData.get_data_target('train_data.txt')
eyeHeight = load_data[0]
eyeWidth = load_data[1]
resultLabel = load_data[2]

scaler = preprocessing.StandardScaler()

eyeHeight = scaler.fit_transform(np.array(eyeHeight))

file_para_scaler = open("scaler_para.txt",'w', encoding='utf-8')

file_para_scaler.write("scaler means on train dataset:\n")
file_para_scaler.write(str(scaler.mean_))

file_para_scaler.write("\n\nscaler vars on train dataset:\n")
file_para_scaler.write(str(np.sqrt(scaler.var_)))

resultTest = clf.fit(eyeHeight, resultLabel)

w_para = clf.coef_
intercept_b = clf.intercept_

file_para_scaler.write("\n\nw-para is:\n")
file_para_scaler.write(str(w_para))

file_para_scaler.write("\n\nintercept_b is:\n")
file_para_scaler.write(str(intercept_b))

load_data = loadData.get_data_target('test_data.txt')
eyeHeightTest = load_data[0]
eyeWidthTest = load_data[1]
resultLabelTest = load_data[2]

# eyeHeightTest = scaler.fit_transform(np.array(eyeHeightTest))

print(scaler.var_)
file_para_scaler.write("\n\n使用计算公式: (x[i] - mean[i])/sqrt(var[i]) 块归一化测试集前十组数据：\n")
temp_data_eye10 = (eyeHeightTest[:10] - scaler.mean_)/np.sqrt(scaler.var_)
file_para_scaler.write(str(temp_data_eye10))

eyeHeightTest = scaler.transform(np.array(eyeHeightTest))

file_para_scaler.write("\n\n\n使用现成模块归一化测试集前十组数据：\n")
file_para_scaler.write(str(eyeHeightTest[:10]))

resultPredictTest = clf.predict(eyeHeightTest)

result_cal = np.dot(w_para, np.array(eyeHeightTest).T) + intercept_b

resultPredictTrain = clf.predict(eyeHeight)

accuracy_normal_test = accuracy_score(resultLabelTest, resultPredictTest)

accuracy_normal_train = accuracy_score(resultLabel, resultPredictTrain)

precision_normal_test = precision_score(resultLabelTest, resultPredictTest)

precision_normal_train = precision_score(resultLabel, resultPredictTrain)

cm_test = confusion_matrix(resultLabelTest, resultPredictTest)

cm_train = confusion_matrix(resultLabel, resultPredictTrain)

print("\033[32m-------------------------------------------------------------\033[0m")

print("normal accuracy on train set is :", accuracy_normal_train)

print("normal precision on train set is :", precision_normal_train)

print("confusion matrix on train :\n", cm_train)

print("normal accuracy on test set is :", accuracy_normal_test)

print("normal precision on test set is :", precision_normal_test)

print("confusion matrix on test :\n", cm_test)

print("\033[32m-------------------------------------------------------------\033[0m")

score_train = clf.decision_function(eyeHeight)

score_test = clf.decision_function(eyeHeightTest)

threshold = 0.7

score_train = score_train > threshold

score_test = score_test > threshold

result_cal = result_cal > threshold

accuracy_train_optimal = accuracy_score(resultLabel, score_train)
precision_train_optimal = precision_score(resultLabel, score_train)
cm_train_optimal = confusion_matrix(resultLabel, score_train)

print("optimal accuracy on train set is :", accuracy_train_optimal)

print("optimal precision on train set is :", precision_train_optimal)

print("confusion matrix on train optimal is :\n", cm_train_optimal)

accuracy_test_optimal = accuracy_score(resultLabelTest, score_test)
precision_test_optimal = precision_score(resultLabelTest, score_test)
cm_test_optimal = confusion_matrix(resultLabelTest, score_test)

print("optimal accuracy on test set is :", accuracy_test_optimal)

print("optimal precision on test set is :", precision_test_optimal)

print("confusion matrix on test optimal is :\n", cm_test_optimal)

accuracy_test_cal_optimal = accuracy_score(resultLabelTest, result_cal[0])
#
print("optimal calculate accuracy on test set is :", accuracy_test_cal_optimal)

from sklearn.externals import joblib

joblib.dump(clf, "my_model.pkl")

file_para_scaler.close()