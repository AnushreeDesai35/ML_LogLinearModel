import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Question 3b - i
y_correct = ['A', 'A', 'A', 'B', 'B', 'A', 'B']
y_pred = ['A', 'A', 'B', 'A', 'B', 'B', 'A']
accuracy = accuracy_score(y_correct, y_pred)
# print('Accuracy: ', accuracy)

# Question 3b - ii
y_correct = [1, 1, 1, 0, 0, 1, 0, 2, 2, 2]
y_pred = [1, 1, 0, 1, 0, 0, 1, 1, 0, 2]
f1Score = f1_score(y_correct, y_pred)
# print('f1ScoreBinary: ', f1Score)

# Question 3b - iii
f1ScoreNone = f1_score(y_correct, y_pred, average = None)
# print('f1ScoreNone', f1ScoreNone)
f1ScoreMicro = f1_score(y_correct, y_pred, average = 'micro')
# print('f1ScoreMicro', f1ScoreMicro)
f1ScoreMacro = f1_score(y_correct, y_pred, average = 'macro')
# print('f1ScoreMacro', f1ScoreMacro)
f1ScoreWeighted = f1_score(y_correct, y_pred, average = 'weighted')
# print('f1ScoreWeighted', f1ScoreWeighted)

# Question 3b - iv
f1ScoreClass0 = f1_score(y_correct, y_pred, pos_label=0, average = 'binary')
# print('f1ScoreClass0: ', f1ScoreClass0)
f1ScoreClass1 = f1_score(y_correct, y_pred, pos_label=1, average = 'binary')
# print('f1ScoreClass1: ', f1ScoreClass1)

y_correct = [1, 1, 1, 0, 0, 1, 0]
y_pred = [1, 1, 0, 1, 0, 0, 1, 1]
#Question 3b - v
f1ScoreLabel_0 = f1_score(y_correct, y_pred, labels=[0], average='macro')
print('f1ScoreLabel_0: ', f1ScoreLabel_0)
f1ScoreLabel_12 = f1_score(y_correct, y_pred, labels=[1,2], average='micro')
print('f1ScoreLabel_12: ',f1ScoreLabel_12)
f1ScoreLabel_01 = f1_score(y_correct, y_pred, labels=[0, 1], average = 'macro')
print('f1ScoreLabel_01: ', f1ScoreLabel_01)
f1ScoreLabel_012 = f1_score(y_correct, y_pred, labels=[0, 1, 2], average = 'macro')
print('f1ScoreLabel_012: ', f1ScoreLabel_012)


