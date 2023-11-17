## ————————————————————————
import pickle
import pandas as pd

# load the model from disk
filename3 = 'poker-model3.sav'
model3 = pickle.load(open(filename3, 'rb'))

filename4 = 'poker-model4.sav'
model4 = pickle.load(open(filename4, 'rb'))

filename_encoder3 = 'encoder_model3.sav'
encoder3 = pickle.load(open(filename_encoder3, 'rb'))

filename_encoder4 = 'encoder_model4.sav'
encoder4 = pickle.load(open(filename_encoder4, 'rb'))


# given parameters:
#   hand: list of tuples (suit, rank); length is 3 or 4
#   opposite_score: the score of opposite player
#
# return:
#   True if you want to call,
#   False if you want to fold
def predict_call(hand, opposite_score):
    inp = list()
    for suit, rank in hand:
        inp.extend([suit, rank])

    if len(hand) == 3:
        test_input_encoded = encoder3.transform([inp])
        X_test = pd.DataFrame(
            test_input_encoded,
            columns=encoder3.get_feature_names_out())
        y_pred = model3.predict_proba(X_test)

    else:
        test_input_encoded = encoder4.transform([inp])
        X_test = pd.DataFrame(
            test_input_encoded,
            columns=encoder4.get_feature_names_out())
        y_pred = model4.predict_proba(X_test)

    # compute expectation
    exp = sum([i * p for i, p in enumerate(y_pred[0])])
    # print("expected score = ", exp)
    if exp >= opposite_score:
        return True
    return False


## ————————————————————————

import pickle
import pandas as pd
import numpy as np
import random

# 테스트 데이터 로드
test = pd.read_csv('poker-hand-testing.csv')

# 10라운드의 돈을 추적하기 위한 배열 초기화
money = [1000] * 10

# 10라운드 반복
for i in range(10):
    # 테스트 데이터에서 1000개의 행을 무작위로 샘플링
    samp = np.random.randint(0, len(test), size=1000)
    X_test = list(test.iloc[samp, :-1].to_numpy())
    y_test = list(test.iloc[samp, -1].to_numpy())

    # zip함수를 사용하여 각 패와 그에 대៖하는 점수를 한 쌍으로 묶어서 반복
    for test_hand, test_score in zip(X_test, y_test):

        # test_hand에는 패에 대한 정보가 들어있는데, 패(모양-랭크).
        # 얘를 들어 [2,10,3,5,1,7]이렇게 되어있으면 첫 번째 카드는 모양이 2, 랭크10 이렇게..
        # 이 정보를 튜플의 리스트로 변환하여 test_input에 저장
        test_input = [(suit, rank) for suit, rank in zip(test_hand[::2], test_hand[1::2])]  # [1,2] 이런 형태임

        # 상대방의 점수를 무작위로 선택(0부터 len(y_test)-1사이의 무작위 정수 반환)
        oppo_score = y_test[random.randrange(len(y_test))]

        if not predict_call(test_input[:3], oppo_score):  # 카드 3개
            money[i] -= 1
            continue

        if not predict_call(test_input[:4], oppo_score):  # 카드 4개
            money[i] -= 2
            continue

        if test_score > oppo_score:
            money[i] += 3

        elif test_score < oppo_score:
            money[i] -= 3

print('average of money= {}'.format(round(np.mean(money), 3)))