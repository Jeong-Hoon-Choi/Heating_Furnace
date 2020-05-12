from preprocessing_pck.data_input import *
from data_for_learning.data_handling_for_using_model import *
import change_point.find_change_point_2 as fc
import press_matching_pck.press_matching as pm
from learning.learning_mod import *
from constant.constant_data_make import *
from bases import plotting
import matplotlib.pyplot as plt


# ------------------------------ make data ---------------------------------
def work_start(view):
    for num in work_:
        data = []
        change_point = []
        for t in os.listdir(base_path + 'input/' + str(num) + '/'):
            path = base_path + 'input/' + str(num) + '/' + t
            print(path)
            get_data_excel(data, path, num)
        h = HF()
        start_real, end_real = fc.data_manipulates(data, num, time_path)
        time_dict, phase_list_dict = fc.find_all(data, change_point, num, start_real, end_real)
        # print("find change_point done heating " + str(num))
        plotting(data, change_point, time_dict['fixed_start_time_list'], time_dict['fixed_end_time_list'], num,
                 time_dict['heat_ended_time_list'], time_dict['real_start_time_list'], time_dict['real_end_time_list'])
        make_database(data, num, h, phase_list_dict)
        h.sett(df_mat, base_path + 'HF_OUT/test_2019_a_')
        print('DB_Done')
    if view:
        plt.show()


# 프레스기 매칭
def work_press():
    h_arr = []
    p = pd.read_csv(base_path + 'data/' + work_space + 'press_par.csv', encoding='euc-kr', index_col=0)
    for i in work_:
        h = HF()
        h.df.read_csv(base_path + 'HF_OUT/test_2019_a_' + str(i) + '.csv', encoding='euc-kr', index_col=0)
        h.set_next_h()
        h.change_list()
        h_arr.append(h)
    pm.matching_press_general(h_arr, p)
    for i in h_arr:
        i.out(base_path + 'HF_OUT/press_2019_1a_')


# 구간 정리
def work_set():
    hh = HF()
    df_t = pd.read_csv(base_path + 'data/start_end_re1.csv', encoding='euc-kr')
    # for num in error_arr_2019:
    for num in work_:
        h = HF()
        h.df.read_csv(base_path + 'HF_OUT/press_2019_1a_' + str(num) + '.csv', encoding='euc-kr', index_col=0)
        # eliminate_error_loop(h, num[1])
        h.match_time(df_t)
        h.fill()
        h.week()
        hh.df = pd.concat([hh.df, h.df])
        hh.df = hh.df.reset_index(drop=True)
    eliminate_drop(hh)
    hh.out(base_path + 'HF_OUT/last_2019_ffa')
    print('phase 2')
    hhh = ['heat', 'hold', 'open', 'reheat']
    for i in hhh:
        print(i)
        h2 = HF()
        for j, row in hh.df.iterrows():
            if hh.df['Type'].loc[j] == i:
                h2.df = h2.df.append(row)
            else:
                pass
        h2.df = h2.df.reset_index(drop=True)
        gum_2(h2)
        h2.df.to_csv(base_path + 'HF_OUT/last_2019_' + str(work_[0]) + '_' + i + '.csv', encoding='euc-kr')
    hh2 = HF()
    hh2.df.read_csv(base_path + 'HF_OUT/last_2019_ffa' + str(work_[0]) + '.csv', encoding='euc-kr', index_col=0)
    handle_first_hold(hh2, work_, base_path + 'HF_OUT/last_2019_' + str(work_[0]) + '_first_hold.csv')
    hh2.out('./HF_OUT_3/last_2019_' + str(work_[0]) + '_drop_first_hold')


# 가열구간 모델용 데이터 만들기
def make_heat():
    s_list, ss_list = sensitive('./data1_201909~201912/p15/sensitive.csv')
    s_list_1, sn_list = sensitive2(base_path + 'data/SS.csv', base_path + 'data/S_N.csv', './data1_201909~201912/p15/sensitive.csv')
    print(len(s_list), len(s_list_1), len(sn_list))
    s_list = s_list + s_list_1
    print(len(s_list))
    df_mat_heat = pd.read_csv(base_path + 'data/heat_steel_par.csv', encoding='euc-kr')
    HT_heat = HF()
    HT_heat.df.read_csv(base_path + 'HF_OUT/last_2019_' + str(work_[0]) + '_heat.csv', encoding='euc-kr', index_col=0)
    HT_heat.df = HT_heat.df.reset_index(drop=True)
    HT_heat.change_list2()
    model_heat_kang_ver_heat(HT_heat, df_mat, df_mat_heat, s_list, ss_list, sn_list, base_path + '/model5/model_' + str(work_[0]) + '.csv')


# 호기 분리
def detach_furnace():
    df_t = pd.read_csv(base_path + 'model5/model_' + str(work_[0]) + '.csv', encoding='euc-kr', index_col=0)
    for num in work_:
        df_t2 = pd.DataFrame(columns=df_t.columns)
        for i, row in df_t.iterrows():
            if int(df_t.loc[i, '가열로번호']) == num:
                df_t2 = df_t2.append(row)
        df_t2 = df_t2.reset_index(drop=True)
        df_t2.to_csv(base_path + 'analysis/furnace_num/' + str(num) + '.csv', encoding='euc-kr')
        print('num :', len(df_t2.index))


# 호기별 특성 csv 저장/그래프
def furnace_characteristic():
    arr_all = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '13': [], '17': [], '18': [], '19': [], '20': []}
    num = 1
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig = plt.figure()
    label = '장입최대중량'
    plt.title(label, position=(0.5, 1.0+0.05), fontsize=15)
    for i in p_all:
        arr_t = []
        count_h = 0
        count_s = 0
        count_t_1 = 0
        count_t_2 = 0
        count_o = 0
        count_A = 0
        count_m = 0
        count_A_1 = 0
        count_A_2 = 0
        df = pd.read_csv(base_path + 'analysis/hogi5/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
        for i2, row in df.iterrows():
            flag_heat = 0
            flag_sense = 0
            flag_open = 0
            flag_m = 0
            arr_t.append(int(df.loc[i2, label]))
            if int(df.loc[i2, '열괴장입소재개수']) > 0:
                count_h += 1
                flag_heat = 1
            if int(df.loc[i2, '문열림횟수']) > 0:
                count_o += 1
                flag_open = 1
            if int(df.loc[i2, '민감소재장입개수']) > 0:
                count_s += 1
                flag_sense = 1
            if int(df.loc[i2, '민감여부']) == 1:
                count_m += 1
                flag_m = 1
            if flag_heat == 0 and flag_sense == 1 and flag_open == 0 and flag_m == 0 and int(df.loc[i2, '시간(총)']) <= 3600:
                count_t_1 += 1
            if flag_heat == 0 and flag_sense == 0 and flag_open == 0 and flag_m == 0 and int(df.loc[i2, '시간(총)']) <= 3600:
                count_t_2 += 1
            if flag_heat == 1 or flag_open == 1 or flag_m == 1:
                count_A += 1
            if flag_heat == 0 and flag_open == 0 and flag_m == 0 and flag_sense == 1:
                count_A_1 += 1
            if flag_heat == 0 and flag_open == 0 and flag_m == 0 and flag_sense == 0:
                count_A_2 += 1
        arr_all[str(i)].append(len(arr_t))
        arr_all[str(i)].append(count_h)
        arr_all[str(i)].append(count_o)
        arr_all[str(i)].append(count_m)
        arr_all[str(i)].append(count_A)
        arr_all[str(i)].append(count_s)
        arr_all[str(i)].append(count_A_1)
        arr_all[str(i)].append(count_t_1)
        arr_all[str(i)].append(count_A_2)
        arr_all[str(i)].append(count_t_2)
        arr_all[str(i)].append(np.mean(arr_t))
        arr_all[str(i)].append(np.var(arr_t))
        arr_all[str(i)].append(np.std(arr_t))
        ax = fig.add_subplot(1, 11, num)
        ax.axhline(y=np.mean(arr_t), color='red', linewidth=3.0)
        plt.ylim([0, 180000])
        # plt.ylim([0, 300000])
        ax.plot(arr_t, '.')
        ax.set_title(i)
        num += 1
    df = pd.DataFrame.from_dict(arr_all)
    df = df.reset_index(drop=True)
    df.to_csv(base_path + 'analysis/analysis_0421.csv', encoding='euc-kr')
    # plt.show()


# 호기 또는 클러스터별 시간-에너지 그래프
def plot_time_energy():
    plt.rcParams["font.family"] = "Malgun Gothic"
    for i in p_bum:
        arr_gas = []
        arr_time = []
        df = pd.read_csv(base_path + 'analysis/hogi/' + str(i[0]) + '_filtered.csv', encoding='euc-kr', index_col=0)
        for i2, row in df.iterrows():
            arr_gas.append(float(df.loc[i2, '에너지']))
            arr_time.append(int(df.loc[i2, i[1]]))
        plt.figure()
        plt.xlim([0, 250000])
        plt.ylim([0, 8000])
        plt.plot(arr_time, arr_gas, '.')
        plt.title(str(i[0]) + ' 시간 - 에너지', position=(0.5, 1.0+0.05), fontsize=15)
    plt.show()


# clustering
def furnace_clustering():
    df = pd.read_csv(base_path + 'model5/model_1.csv', encoding='euc-kr', index_col=0)
    for condition in clustering_condition_constant:
        for i in p_bum:
            arr_t = []
            # df = pd.read_csv(base_path + 'analysis/hogi/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            df_t = pd.DataFrame(columns=df.columns)
            for i2, row in df.iterrows():
                heat_flag = 0
                sense_flag = 0
                time_flag = 0
                flag_m = 0
                if int(df.loc[i2, '가열로번호']) in i:
                    if int(df.loc[i2, '열괴장입소재개수']) > 0 or int(df.loc[i2, '문열림횟수']) > 0:
                        heat_flag = 1
                    if int(df.loc[i2, '민감소재장입개수']) > 0:
                        sense_flag = 1
                    if int(df.loc[i2, '민감여부']) == 1:
                        flag_m = 1
                    if int(df.loc[i2, '시간(총)']) <= 3600:
                        time_flag = 1
                    if clustering_condition(condition, heat_flag, flag_m, time_flag, sense_flag):
                        df_t = df_t.append(row)
            df_t = df_t.reset_index(drop=True)
            # df_t.to_csv(base_path + 'analysis/hogi/' + str(i) + '_filtered.csv', encoding='euc-kr')
            print(len(df_t.index))
            df_t.to_csv(base_path + 'analysis/for_learning5/' + condition + '/' + str(i) + '.csv', encoding='euc-kr')


# ------------------------------ learning ---------------------------------
def HF_heating_module():
    epoch = 2000
    seed_start = 10
    seed_end = 10
    for i2 in path_1:
        print(i2)
        for i in [p_bum[4]]:
            df_origin = pd.read_csv('./heating_furnace/heat/data0330/data5/' + str(i2[0]) + '/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            print(i2, i, '개수', len(df_origin.index))
            df_new = pd.DataFrame()
            for seed1 in range(seed_start, seed_end):
                test_pred = {'시간(총)': '', '시간(0제외)': '', '에너지': '', '시간(총)_mape': '', '시간(0제외)_mape': '', '에너지_mape': ''}
                train_pred = {'시간(총)': '', '시간(0제외)': '', '에너지': '',  '시간(총)_mape': '', '시간(0제외)_mape': '', '에너지_mape': ''}
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list_0325:
                    for j in j2:
                        out = []
                        out2 = []
                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], j[2], seed1)
                            # data_manipulate_no_split(df_origin, j[0], j[1])
                            # data_manipulate_pca(origin2, j[0], j[1], seed1)
                        count = 0
                        print(train_feature)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature, train_label, test_feature, test_label)
                        out2.append(mean_absolute_percentage_error(test_label, knn_test_pred))

                        # MLP
                        for hidden, unit in [[5, 5]]:
                            print('seed : ', seed1, 'epoch : ', epoch, 'unit : ', unit, 'hidden : ', hidden)
                            s1, mlp_test_pred, mlp_train_pred, model = MLP(train_feature, train_label, test_feature, test_label, epoch=epoch, unit=unit, hidden=hidden)
                            train_pred[j[0]] = mlp_train_pred
                            test_pred[j[0]] = mlp_test_pred
                            out.append(s1)
                            df_new.loc[seed1 - seed_start, j[3] + '_MLP_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = out[len(out) - 1]

                        df_new.loc[seed1 - seed_start, j[3] + '_KNN_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = out2[len(out2) - 1]

                        # df_new.loc[seed1 - 10, j[3] + '_MLP_' + j[0]] = out[len(out) - 1]
                        # df_new.loc[seed1 - 10, j[3] + '_KNN_' + j[0]] = out2[len(out2) - 1]


                        # 랜덤 포레스트
                        # random_forest(train_feature, train_label, test_feature, test_label, j[0])

                        # 모든 data를 train set으로 했을 때의 prediction
                        '''
                        df_origin.loc[:, j[0] + '_pred'] = ''
                        df_origin.loc[:, j[0] + '_mape'] = ''
                        df_origin = df_origin.reset_index(drop=True)
                        for k in range(len(df_origin.index)):
                            # x[j[2] + 'knn_pred'].loc[k] = knn_train_pred[k][0]
                            e = df_origin[j[0]].loc[k]
                            ee = train_pred[j[0]][k][0]
                            df_origin[j[0] + '_pred'].loc[k] = ee
                            df_origin[j[0] + '_mape'].loc[k] = abs(e - ee) / e * 100
                            '''

                        # learning with loaded model(FFN)
                        '''
                        arrrr = []
                        epochh = []
                        for k in range(100, epoch+1, 100):
                            model_F = FFN(train_feature.shape[1], train_feature, train_label, test_feature, test_label,
                                          load_path='./check/model checkpoint epoch2_' + str(seed) + '_' + str(k) + '.h5')
                            print(k)
                            re = model_F.load()
                            epochh.append(k)
                            arrrr.append(re)
                        plt.plot(epochh, arrrr)
                        plt.title(str(seed))
                        '''
                    # add prediction to normalized data (FFN)
                    '''
                    train_path = './heating_furnace/heat/data0330/check/' + str(i2) + '/' + str(i) + '/train/' + j[3] + '/' + str(seed1) + '.csv'
                    test_path = './heating_furnace/heat/data0330/check/' + str(i2) + '/' + str(i) + '/test/' + j[3] + '/' + str(seed1) + '.csv'
                    add_prediction_to_normalized_data(test_pred, train_pred, j2, x, y, train_path, test_path)
                '''

            # save_result
            # df_new = df_new.transpose()
            for i0 in df_new.columns:
                print(i0)
                arr_avg = []
                for i01, ro2 in df_new.iterrows():
                    if not pd.isna(df_new.loc[i01, i0]):
                        arr_avg.append(float(df_new.loc[i01, i0]))
                print(arr_avg)
                df_new.loc[seed_end - seed_start - 1, i0] = np.average(arr_avg)
            df_new.to_csv('./heating_furnace/heat/data0330/result0420_1/' + str(i2[0]) + '/result_' + str(i) + '1.csv', encoding='euc-kr')
