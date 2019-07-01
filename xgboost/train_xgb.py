# -*- coding:utf-8 -*-

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from utils import get_xgb_data

if __name__ == '__main__':
    params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',  # 回归问题
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    }
    num_round = 3

    train_data = get_xgb_data('../data/xgb_train.txt')
    test_data = get_xgb_data('../data/xgb_test.txt')

    model = xgb.train(params, train_data, num_round)

    model.save_model('../data/xgb.m')

    preds = model.predict(test_data)

    # show the importance of feature
    plot_importance(model)
    plt.show()