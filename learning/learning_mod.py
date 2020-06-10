import pandas as pd
import numpy as np
from constant.constant_data_make import *
from constant.constant_learning import *
import matplotlib.pyplot as plt
from learning.ffn import FFN
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor


# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array((y_pred))
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# KNN
def KNN_reg(train_feature, train_label, test_feature, test_label):
    print(len(train_feature), len(test_feature), len(train_label), len(test_label))
    # KNN
    k_range = np.arange(5, int(len(train_feature)*4/5), 5)
    neigh_dict = {'n_neighbors': k_range}
    my_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    optimize = GridSearchCV(KNeighborsRegressor(weights='distance'), neigh_dict, scoring=my_scorer, cv=5)
    optimize.fit(train_feature, train_label)
    print("best : ", optimize.best_params_)
    score = optimize.score(test_feature, test_label)
    print("test score : ", score)
    return optimize.predict(test_feature), optimize.predict(train_feature), optimize.best_params_


# MLP
def MLP(train_feature, train_label, test_feature, test_label, epoch=2000, unit=30, hidden=5, s=None, s3=0):
    print(len(train_feature), len(test_feature), len(train_label), len(test_label))

    model_F = FFN(train_feature.shape[1], train_feature, train_label, test_feature, test_label, epoch, unit, hidden, s, check_seed=s3)
    score, test_pred, m = model_F.run()
    train_pred = m.predict(x=train_feature)
    return score, test_pred, train_pred, m


# Decision Tree
def decision_tree_reg(train_feature, train_label, test_feature, test_label):
    print(len(train_feature), len(test_feature), len(train_label), len(test_label))

    my_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    optimize = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid={}, scoring=my_scorer, cv=5)
    optimize.fit(train_feature, train_label)
    print("best : ", optimize.best_params_)
    score = optimize.score(test_feature, test_label)
    print("test score : ", score)
    return optimize.predict(test_feature), optimize.predict(train_feature), optimize.best_params_


# random forest _ not used
def random_forest(train_feature, train_label, test_feature, test_label, label_list):
    mseOos = []
    nTreeList = range(50, 500, 10)
    train_label = np.array(train_label[label_list])
    test_label = np.array(test_label[label_list])
    for iTrees in nTreeList:
        depth = None
        maxFeat = 4  # 조정해볼 것
        model = ensemble.RandomForestRegressor(n_estimators=iTrees,
                                                     max_depth=depth, max_features='auto',
                                                     oob_score=False, random_state=531)
        model.fit(train_feature, train_label)
        # 데이터 세트에 대한 MSE 누적
        prediction = model.predict(test_feature)
        print(mean_absolute_percentage_error(test_label, prediction))
        mseOos.append(mean_absolute_percentage_error(test_label, prediction))

    print(mseOos[-1])

    # 트레이닝 테스트 오차 대비  앙상블의 트리 개수 도표 그리기
    plt.plot(nTreeList, mseOos)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('MAPE')
    # plot.ylim([0.0, 1.1*max(mseOob)])
    plt.show()


# split data by each furnace
def split(s):
    df = pd.read_csv(s + '.csv', encoding='euc-kr', index_col=0)
    for i in p_all:
        df_temp = pd.DataFrame(columns=df.columns)
        for j, row in df.iterrows():
            if int(df['가열로번호'].loc[j]) == i:
                df_temp = df_temp.append(row)
                df_temp = df_temp.reset_index(drop=True)
        df_temp.to_csv(s + '_' + str(i) + '.csv', encoding='euc-kr')


# data train-test split, pca, normalizing
def data_manipulate_pca(origin, label, list_feature, k=42):
    list_label = [label]
    print(list_feature)
    x, y = train_test_split(origin, test_size=0.3, random_state=k)
    # x, y = split_tt(origin, test_size=0.3, seed=k)
    # x.to_csv('train_1117.csv', encoding='euc-kr')
    # y.to_csv('test_1117.csv', encoding='euc-kr')
    x_feature = x[list_feature]
    x_label = x[list_label]
    y_feature = y[list_feature]
    y_label = y[list_label]
    pca = PCA(n_components=len(list_feature))
    pca.fit(x_feature)
    x_feature = pca.transform(x_feature)
    y_feature = pca.transform(y_feature)
    std_scale_x = preprocessing.StandardScaler().fit(x_feature)
    x_feature = std_scale_x.transform(x_feature)
    y_feature = std_scale_x.transform(y_feature)
    # print(x_feature)
    # x_feature = x_feature[:, 0, 2, 3, 6]
    # print(x_feature)
    # y_feature = y_feature[:, 0, 2, 3, 6]
    return x_feature, x_label, y_feature, y_label, x, y


# data train-test split, normalizing
def data_manipulate_normal3(x, y, label, list_feature1, list_feature2=None, seed=42, mode=None):
    list_label = [label]
    # print(list_feature1 + list_feature2)
    # if all features are needed normalizing
    if list_feature2 is None:
        print(list_feature1)
        print(list_label)
        x_feature = x[list_feature1]
        x_label = x[list_label]
        y_feature = y[list_feature1]
        y_label = y[list_label]
        std_scale_x = preprocessing.StandardScaler().fit(x_feature)
        x_feature3 = std_scale_x.transform(x_feature)
        y_feature3 = std_scale_x.transform(y_feature)
        x_feature = pd.DataFrame(columns=[list_feature1])
        y_feature = pd.DataFrame(columns=[list_feature1])
        x_feature = x_feature.append(pd.DataFrame(data=x_feature3, columns=x_feature.columns))
        x_feature = x_feature.reset_index(drop=True)
        y_feature = y_feature.append(pd.DataFrame(data=y_feature3, columns=y_feature.columns))
        y_feature = y_feature.reset_index(drop=True)
    else:
        print(list_feature1 + list_feature2)
        print(list_label)
        x_feature2 = {}
        y_feature2 = {}
        for i in list_feature2:
            x_feature2[i] = []
            y_feature2[i] = []
        x_feature1 = x[list_feature1]
        for i in list_feature2:
            temp = x[i]
            temp = temp.reset_index(drop=True)
            x_feature2[i] = temp
        x_label = x[list_label]
        y_feature1 = y[list_feature1]
        for i in list_feature2:
            temp = y[i]
            temp = temp.reset_index(drop=True)
            y_feature2[i] = temp
        y_label = y[list_label]
        std_scale_x = preprocessing.StandardScaler().fit(x_feature1)
        x_feature1 = std_scale_x.transform(x_feature1)
        x_feature3 = []
        # print(x_feature1)
        for i in range(len(x_feature1)):
            temp = []
            for k in range(len(x_feature1[0])):
                temp.append(x_feature1[i][k])
            for t in list_feature2:
                temp.append(x_feature2[t].loc[i])
            x_feature3.append(temp)
        y_feature1 = std_scale_x.transform(y_feature1)
        y_feature3 = []
        for i in range(len(y_feature1)):
            temp = []
            for k in range(len(y_feature1[0])):
                temp.append(y_feature1[i][k])
            for t in list_feature2:
                temp.append(y_feature2[t].loc[i])
            y_feature3.append(temp)
        x_feature = pd.DataFrame(columns=[list_feature1 + list_feature2])
        y_feature = pd.DataFrame(columns=[list_feature1 + list_feature2])
        x_feature = x_feature.append(pd.DataFrame(data=x_feature3, columns=x_feature.columns))
        x_feature = x_feature.reset_index(drop=True)
        y_feature = y_feature.append(pd.DataFrame(data=y_feature3, columns=y_feature.columns))
        y_feature = y_feature.reset_index(drop=True)
    return x_feature, x_label, y_feature, y_label


def Train_Test_split(origin, seed=42, mode=None):
    # leave one out on/off
    # Don't use leave one out
    if mode is None:
        x, y = train_test_split(origin, test_size=0.3, random_state=seed)
    # using leave one out
    else:
        temp_x = pd.DataFrame()
        temp_y = pd.DataFrame()
        for k, row in origin.iterrows():
            if k == mode:
                temp_y = temp_y.append(row)
            else:
                temp_x = temp_x.append(row)
                temp_x = temp_x.reset_index(drop=True)
        x = temp_x
        y = temp_y
    return x, y


def add_prediction_to_normalized_data(test_pred, train_pred, f_list, x, y, train_path, test_path):
    for j in f_list:
        label = j[0]
        x.loc[:, label + '_pred'] = ''
        x.loc[:, label + '_mape'] = ''
        x = x.reset_index(drop=True)
        for k in range(len(x.index)):
            # x[j[2] + 'knn_pred'].loc[k] = knn_train_pred[k][0]
            e = x[label].loc[k]
            ee = train_pred[label][k][0]
            x[label + '_pred'].loc[k] = ee
            x[label + '_mape'].loc[k] = abs(e - ee) / e * 100
        y.loc[:, label + '_pred'] = ''
        y.loc[:, label + '_mape'] = ''
        y = y.reset_index(drop=True)
        for k in range(len(y.index)):
            # y[j[2] + 'knn_pred'].loc[k] = knn_test_pred[k][0]
            e = y[label].loc[k]
            ee = test_pred[label][k][0]
            y[label + '_pred'].loc[k] = ee
            y[label + '_mape'].loc[k] = abs(e - ee) / e * 100
    x.to_csv(train_path, encoding='euc-kr')
    y.to_csv(test_path, encoding='euc-kr')
