# -*- coding:utf-8 -*-

import sys
sys.path.append('..')
import mgnn_m.config_train as args
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    params = {
    'booster': 'gbtree',
    # 'objective': 'reg:gamma',  # 回归问题
    'objective': 'reg:linear',  # 回归问题
    'gamma': 0.05,
    'max_depth': 5,
    'lambda': 1,
    'subsample': 0.9,
    'colsample_bytree': 1.0,
    'min_child_weight': 1,
    'silent': 0,
    'eta': 0.3,
    'seed': 1000,
    'nthread': 1,
    }
    num_round = 20

    train_data = xgb.DMatrix(args.xgb_train)

    print(train_data)
    model = xgb.train(params, train_data, num_round)

    model.save_model(args.xgb_model)

    preds = model.predict(train_data)
    # for item in preds:
    #     print(item)
    print(preds)

    # # show the importance of feature
    # plot_importance(model)
    # plt.show()