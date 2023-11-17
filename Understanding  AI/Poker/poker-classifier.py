#임포트
import pandas as pd
from sklearn.model_selection import train_test_split

import pickle

from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

#데이터셋 불러오기
df_train = pd.read_csv('poker-hand-training.csv')
df_test = pd.read_csv('poker-hand-testing.csv')

#전체 데이터셋에서 일단 카드 덱 3개만(model3)

X = df_train.iloc[:, :-5]
y = df_train.iloc[:, -1]

X_test = df_test.iloc[:,:-5]
y_test = df_test.iloc[:, -1]

encoder3 = OneHotEncoder(drop = 'first', sparse= False)

X = pd.DataFrame(encoder3.fit_transform(X.iloc[:, :]))

X_test = pd.DataFrame(encoder3.fit_transform(X_test.iloc[:,:]))

#Cross_validation
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, random_state = 42)

#Multilayer perceptron with 85 neurons as input and 10 neurons as output
model3 = MLPClassifier(hidden_layer_sizes=(50,40,10),
                      solver='adam',
                      activation='relu',
                      alpha = 0.001,
                      max_iter = 400)

model3.fit(X_train,y_train)

predictions = model3.predict(X_val)
print("model3 검증 세트 accuracy score:", accuracy_score(y_val, predictions))

predictions = model3.predict(X_test)
print("model3 테스트 세트 accuracy score:", accuracy_score(y_test, predictions))

# save the model to disk
filename = 'poker-model3.sav'
pickle.dump(model3, open(filename, 'wb'))

# Save the encoder to disk
filename_encoder3 = 'encoder_model3.sav'
pickle.dump(encoder3, open(filename_encoder3, 'wb'))

#-------------------------------------------------------------
#전체 데이터셋에서 일단 카드 덱 4개만(model3)

X = df_train.iloc[:, :-3]
y = df_train.iloc[:, -1]

X_test = df_test.iloc[:,:-3]
y_test = df_test.iloc[:, -1]


encoder4 = OneHotEncoder(drop = 'first', sparse= False)

X = pd.DataFrame(encoder4.fit_transform(X.iloc[:, :]))

X_test = pd.DataFrame(encoder4.fit_transform(X_test.iloc[:,:]))

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, random_state = 42)

#Multilayer perceptron with 85 neurons as input and 10 neurons as output
model4 = MLPClassifier(hidden_layer_sizes=(50,40,10),
                      solver='adam',
                      activation='relu',
                      alpha = 0.001,
                      max_iter = 400)

model4.fit(X_train,y_train)

predictions = model4.predict(X_val)
print("model4 검증 세트 accuracy score:", accuracy_score(y_val, predictions))

predictions = model4.predict(X_test)
print("model4 테스트 세트 accuracy score:", accuracy_score(y_test, predictions))


# save the model to disk
filename = 'poker-model4.sav'
pickle.dump(model4, open(filename, 'wb'))

# Save the encoder to disk
filename_encoder = 'encoder_model4.sav'
pickle.dump(encoder4, open(filename_encoder, 'wb'))




