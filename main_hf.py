from module import *
import pandas as pd
import os
import random
from scipy.spatial import distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_heating_flat(num, df, data, change_point, phase_list_dict):
    heat = phase_list_dict['heat']
    dt = len(df.index)
    for i in range(len(heat)):
        start = None
        end = None
        tent_gas = []
        tent_tem = []
        for j in range(heat[i][0], heat[i][1] + 1):
            # test = change_point[hold[i][0]:hold[i][1]+1]
            if heat[i][0] == heat[i][1]:
                continue

            if change_point[j] != None:
                if start == None:
                    start = j
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])
                else:
                    end = j
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])

                    # Only for detecting flat / holding period
                    time_diff = data[end]['TIME'] - data[start]['TIME']
                    time_diff = time_diff.total_seconds()/60
                    if abs(data[end]['TEMPERATURE'] - data[start]['TEMPERATURE']) < 50 and time_diff > 30:
                        field = [num, data[start]['TIME'], data[start]['TEMPERATURE'], data[end]['TIME'],
                                 data[end]['TEMPERATURE'], np.mean(tent_tem), np.sum(tent_gas)]
                        df.loc[dt] = field
                        dt += 1

                    # For getting all data
                    # field = [num, data[start]['TIME'], data[start]['TEMPERATURE'], data[end]['TIME'],
                    #          data[end]['TEMPERATURE'], np.mean(tent_tem), np.sum(tent_gas)]
                    # df.loc[dt] = field
                    # dt += 1

                    start = j
                    end = None
                    tent_gas = []
                    tent_tem = []
            else:
                if start != None:
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])


def plot_holding_temperatures(num, df, data, time_dict):
    dt = len(df.index)
    for i in range(len(time_dict['heat_ended_time_list'])):
        tent_gas = []
        tent_tem = []
        start = None
        end = None
        for j in range(len(data)):
            if data[j]['TIME'] == time_dict['heat_ended_time_list'][i]:
                start = j
                break
        for j in range(len(data)):
            if data[j]['TIME'] == time_dict['fixed_end_time_list'][i]:
                end = j
                break
        for j in range(start + 1, end + 1):
            tent_gas.append(data[j]['GAS'])
        for k in range(start, end + 1):
            tent_tem.append(data[k]['TEMPERATURE'])
        # print(len(hold), i)
        field = [num, data[start]['TIME'], data[start]['TEMPERATURE'], data[end]['TIME'], data[end]['TEMPERATURE'], np.mean(tent_tem), np.sum(tent_gas)]
        df.loc[dt] = field
        dt += 1


def get_data():
    # Heating phase data
    df = pd.DataFrame(columns=['가열로 번호', '시작시간', '시작온도', '종료시간', '종료온도', '평균온도', '가스사용량'])
    df.reset_index(drop=True)

    # Holding phase data
    # df2 = pd.DataFrame(columns=['가열로 번호', '시작시간', '시작온도', '종료시간', '종료온도', '평균온도', '가스사용량'])
    # df2.reset_index(drop=True)
    for num in work_:
        data = []
        change_point = []
        for t in os.listdir(base_path + 'input/' + str(num) + '/'):
            path = base_path + 'input/' + str(num) + '/' + t
            print(path)
            get_data_excel(data, path, num)
        start_real, end_real = fc.data_manipulates(data, num, time_path)
        time_dict, phase_list_dict = fc.find_all(data, change_point, num, start_real, end_real)
        # Heating phase data
        plot_heating_flat(num, df, data, change_point, phase_list_dict)

        # Holding phase data
        # plot_holding_temperatures(num, df2, data, time_dict)
    # Heating phase data
    df.to_csv(base_path + "HF_OUT/hf_heating.csv", mode='w', encoding='euc-kr')

    # Holding phase data
    # df2.to_csv(base_path + "HF_OUT/hf_holding_summary.csv", mode='w', encoding='euc-kr')


def summarize_heating_data():
    datas = pd.read_csv(base_path + 'HF_OUT/hf_heating.csv', encoding='euc-kr')

    bins = []
    labels = []
    for x in range(0, 1400, 10):
        y = x + 10
        bins.append(x)
        labels.append(str(x) + " - " + str(y))
    bins.append(np.inf)

    datas = datas.groupby(pd.cut(datas['시작온도'], bins=bins, labels=labels)).size().reset_index(name='count')
    datas = datas[datas['count'] != 0]

    datas.to_csv(base_path + "HF_OUT/hf_heating_summary.csv", mode='w', encoding='euc-kr')


def check_data():
    for j in work_:
        h = HF()
        h.df = pd.read_csv(base_path + 'HF_OUT/hf_2020_' + str(j) + '.csv', encoding='euc-kr', index_col=0)
        h2 = HF()
        h2.df['STATUS'] = ""
        for i in range(len(h.df.index)):
            if h.df['Type'].loc[i] == 'heat':
                h2.df = h2.df.append(h.df.loc[i])
                if len(h2.df['소재 list'].loc[i]) == 2:
                    h2.df['STATUS'].loc[i] = 'NO MATERIAL'
                elif h2.df['drop_flag'].loc[i] == 1:
                    h2.df['STATUS'].loc[i] = 'GAS OFF/TEMP OFF'
                # elif h2.df['가열중 문열림 횟수'].loc[i] > 0:
                #     h2.df['STATUS'].loc[i] = 'DOOR OPENED WHILE HEATING'
                else:
                    h2.df['STATUS'].loc[i] = 'OK'
        h2.df = h2.df.reset_index(drop=True)
        h2.df.to_csv(base_path + 'HF_OUT/check_2020_' + str(j) + '.csv', encoding='euc-kr')


def check_real_start_time():
    df_t = pd.read_csv(base_path + 'data/start_end_re1.csv', encoding='euc-kr', index_col=0)
    df_t['DETECTED'] = "NOT FOUND"
    df_t['STATUS'] = ""
    for j in work_:
        h = HF()
        h.df = pd.read_csv(base_path + 'HF_OUT/check_2020_' + str(j) + '.csv', encoding='euc-kr', index_col=0)
        hf_name = '가열로' + str(j) + '호기'
        for i, row in df_t.iterrows():
            nama = df_t['가열로명'].loc[i]
            if nama == hf_name:
                real_start = str(dt.datetime.strptime(df_t['가열시작일시'].loc[i], "%Y-%m-%d %H:%M"))
                for k, bar in h.df.iterrows():
                    mulai = h.df['실제 시작시간'].loc[k]
                    if mulai == real_start:
                        df_t['DETECTED'].loc[i] = "FOUND"
                        if len(h.df['소재 list'].loc[k]) == 2:
                            df_t['STATUS'].loc[i] = "NO MATERIAL"
                        elif h.df['drop_flag'].loc[k] == 1:
                            df_t['STATUS'].loc[i] = "GAS OFF/TEMP OFF"
                        # elif h.df['가열중 문열림 횟수'].loc[k] > 0:
                        #     df_t['STATUS'].loc[i] = "DOOR OPENED WHILE HEATING"
                        else:
                            df_t['STATUS'].loc[i] = "OK"
                        break
    df_t.to_csv(base_path + 'HF_OUT/check_2020_start_end_re1.csv', encoding='euc-kr')


def plot_heating_data(view):
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
        # plotting(data, change_point, time_dict['fixed_start_time_list'], time_dict['fixed_end_time_list'], num,
        #          time_dict['heat_ended_time_list'], time_dict['real_start_time_list'], time_dict['real_end_time_list'])
        # plt.show()
        plotting_weekly(data, change_point, time_dict['fixed_start_time_list'], time_dict['fixed_end_time_list'], num,
                 time_dict['heat_ended_time_list'], time_dict['real_start_time_list'], time_dict['real_end_time_list'], view)
        # make_database(data, num, h, phase_list_dict)
        make_database2(data, num, h, change_point, phase_list_dict)
        h.sett(df_mat, base_path + 'HF_OUT/hf_2020_')
        print(str(num) + ' DB_Done')


# 프레스기 매칭
def work_press2():
    h_arr = []
    p = pd.read_csv(base_path + 'data/' + work_space + 'press_par.csv', encoding='euc-kr', index_col=0)
    for i in work_:
        h = HF()
        h.df = pd.read_csv(base_path + 'HF_OUT/hf_2020_' + str(i) + '.csv', encoding='euc-kr', index_col=0)
        h.set_next_h_2()
        h.change_list()
        h_arr.append(h)
    pm.matching_press_general(h_arr, p)
    for i in h_arr:
        i.out(base_path + 'HF_OUT/press_2020_')


# 구간 정리
def work_set2(curve_type=0):
    hh = HF()
    df_t = pd.read_csv(base_path + 'data/start_end_re1.csv', encoding='euc-kr')
    for num in work_:
        h = HF()
        # data without press matching
        h.df = pd.read_csv(base_path + 'HF_OUT/hf_2020_' + str(num) + '.csv', encoding='euc-kr', index_col=0)
        # data with press matching
        # h.df = pd.read_csv(base_path + 'HF_OUT/press_2020_' + str(num) + '.csv', encoding='euc-kr', index_col=0)
        h.match_time(df_t)
        print(str(num), '- end time matching')
        h.fill()
        print(str(num), '- end fill items')
        h.week()
        hh.df = pd.concat([hh.df, h.df])
        hh.df = hh.df.reset_index(drop=True)
    eliminate_drop(hh)  # eliminate error drop_flag loop
    eliminate_no_material_list(hh)  # eliminate no material list Loop
    hh.out(base_path + 'HF_OUT/last_2020_ffa')
    print('phase 2')
    # for heat
    print('heat')
    j = 0
    h2 = HF()
    row_data = len(hh.df.index)
    while j < row_data:
        if hh.df['Type'].loc[j] == 'heat':
            if curve_type == 0:
                h2.df = h2.df.append(hh.df.loc[j])
                j += 1
            elif curve_type == 10:
                count = 1
                k = j + 1
                while hh.df['Type'].loc[k] == 'heat' and hh.df['cycle'].loc[k] == hh.df['cycle'].loc[j]:
                    count += 1
                    k += 1
                if count % 2 == 0 or count > 5:
                    for l in range(j, k):
                        h2.df = h2.df.append(hh.df.loc[l])
                j = k
            else:
                count = 1
                k = j + 1
                while hh.df['Type'].loc[k] == 'heat' and hh.df['cycle'].loc[k] == hh.df['cycle'].loc[j]:
                    count += 1
                    k += 1
                if count == curve_type:
                    for l in range(j, k):
                        h2.df = h2.df.append(hh.df.loc[l])
                j = k
        else:
            j += 1
    h2.df = h2.df.reset_index(drop=True)
    gum_22(h2)  # checking that current cycle's material is same with previous cycle's material
    h2.df.to_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_heat.csv', encoding='euc-kr')
    # for other beside heat
    hhh = ['hold', 'open', 'reheat']
    # hhh = ['hold']
    hf_heat = HF()
    hf_heat.df = pd.read_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_heat.csv',
                             encoding='euc-kr', index_col=0)
    for i in hhh:
        print(i)
        h2 = HF()
        for j, row in hh.df.iterrows():
            if hh.df['Type'].loc[j] == i:
                current_cycle = hh.df['cycle'].loc[j]
                current_hf = hh.df['가열로 번호'].loc[j]
                flag_heat = True
                filtered_hf_heat = hf_heat.df[hf_heat.df['cycle'] == current_cycle]
                filtered_hf_heat = filtered_hf_heat[filtered_hf_heat['가열로 번호'] == current_hf]
                if not filtered_hf_heat.empty:
                    flag_heat = False
                if flag_heat:
                    continue
                else:
                    h2.df = h2.df.append(row)
            else:
                pass
        h2.df = h2.df.reset_index(drop=True)
        gum_22(h2)  # checking that current cycle's material is same with previous cycle's material
        h2.df.to_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_' + i + '.csv', encoding='euc-kr')
    hh2 = HF()
    hh2.df = pd.read_csv(base_path + 'HF_OUT/last_2020_ffa' + str(work_[0]) + '.csv', encoding='euc-kr', index_col=0)
    handle_first_hold(hh2, work_, base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_first_hold.csv', hf_heat)
    hh2.out(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_drop_first_hold')


def make_heat_or_hold(model):
    s_list, ss_list = sensitive_()
    df_mat_heat = pd.read_csv(base_path + 'data/heat_steel_par.csv', encoding='euc-kr')
    if model == 'energy-increasing' or model == 'time':
        HT_heat = HF()
        HT_heat.df = pd.read_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_heat_' +
                                 str(model) + '.csv', encoding='euc-kr', index_col=0)
        HT_heat.df = HT_heat.df.reset_index(drop=True)
        HT_heat.change_list2()

        model_heat_kang_ver_heat2(HT_heat, df_mat, df_mat_heat, s_list, ss_list, base_path +
                                 '/model/model_' + str(work_[0]) + '_' + str(model) + '.csv')
    elif model == 'energy-holding':
        HT_heat = HF()
        HT_heat.df = pd.read_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_heat_' +
                                 str(model) + '.csv', encoding='euc-kr', index_col=0)
        HT_heat.df = HT_heat.df.reset_index(drop=True)
        HT_heat.change_list2()

        HT_hold = HF()
        # HT_hold.df = pd.read_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_first_hold.csv',
        #                          encoding='euc-kr', index_col=0)
        HT_hold.df = pd.read_csv(base_path + 'HF_OUT/last_2020_' + str(work_[0]) + '_hold.csv',
                                 encoding='euc-kr', index_col=0)
        HT_hold.df = HT_hold.df.reset_index(drop=True)
        HT_hold.change_list2()

        model_hold_kang_ver_hold(HT_heat, HT_hold, df_mat, df_mat_heat, s_list, ss_list, base_path +
                                 '/model/model_' + str(work_[0]) + '_' + str(model) + '.csv')
    else:
        print('wrong model')
        exit(0)


def HF_learning(model, save_model=False):
    feature_list = ''
    if model == 'energy-increasing':
        print('energy-increasing')
        # feature_list = feature_list_0325_3_1
        # feature_list = feature_list_0325_time_only
        feature_list = feature_list_0325_3_3
    elif model == 'energy-holding':
        print('energy-holding')
        # feature_list = feature_list_0325_3_2
        # feature_list = feature_list_0325_time_only
        feature_list = feature_list_0325_3_3
    elif model == 'time':
        print('time')
        feature_list = feature_list_0325_3_4
    elif model == 'start-temperature':
        print('start-temperature')
        feature_list = feature_list_0325_3_5
    else:
        print('wrong model')
        exit(0)

    epoch = 20000
    seed_start = random.randint(1, 10000)
    # seed_start = 1
    seed_end = seed_start + 1

    for i2 in path_1:
        df_complete = pd.DataFrame()
        # for i in [p_bum[4]]:
        for i in p_bum:
            df_origin = pd.read_csv(base_path + 'analysis/for_learning_' + str(model) + '/' + str(i2[0]) +
                                    '/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            print(i2, i, '개수', len(df_origin.index))
            df_new = pd.DataFrame()
            for seed1 in range(seed_start, seed_end):
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list:  # Feature list
                    for j in j2:
                        out = []
                        out2 = []
                        out3 = []
                        out4 = []

                        if model == 'energy-increasing':
                            if i == [1]:
                                j[1] = ['종료온도', '시간(총)']
                                j[3] = '종료온도_시간(총)'
                            elif i == [18] or i == [19] or i == [20]:
                                j[1] = ['시작온도', '시간(총)']
                                j[3] = '시작온도_시간(총)'
                            else:
                                j[1] = ['시간(총)']
                                j[3] = '시간(총)'
                        if model == 'energy-holding':
                            if i == [4]:
                                j[1] = ['장입최대중량', '시간(총)']
                                j[3] = '장입최대중량_시간(총)'
                            elif i == [5]:
                                j[1] = ['시작온도', '시간(총)']
                                j[3] = '시작온도_시간(총)'
                            elif i == [13]:
                                j[1] = ['종료온도', '시간(총)']
                                j[3] = '종료온도_시간(총)'
                            else:
                                j[1] = ['시간(총)']
                                j[3] = '시간(총)'
                        if model == 'time':
                            if i == [1]:
                                j[1] = ['장입중량총합', '시작온도', '종료온도']
                                j[3] = '장입중량총합_시작온도_종료온도'
                            elif i == [2] or i == [18]:
                                j[1] = ['장입소재개수', '시작온도']
                                j[3] = '장입소재개수_시작온도'
                            elif i == [3] or i == [20]:
                                j[1] = ['장입소재개수', '시작온도', '종료온도']
                                j[3] = '장입소재개수_시작온도_종료온도'
                            elif i == [4]:
                                j[1] = ['장입중량총합', '시작온도']
                                j[3] = '장장입중량총합_시작온도'
                            elif i == [5]:
                                j[1] = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '종료온도']
                                j[3] = '장입중량총합_장입최대중량_장입소재개수_시작온도_종료온도'
                            elif i == [6] or i == [19]:
                                j[1] = ['시작온도', '종료온도']
                                j[3] = '시작온도_종료온도'
                            elif i == [13] or i == [17]:
                                j[1] = ['장입중량총합', '장입소재개수', '시작온도']
                                j[3] = '장입중량총합_장입소재개수_시작온도'

                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], j[2], seed1)
                        # data_manipulate_no_split(df_origin, j[0], j[1])
                        # data_manipulate_pca(origin2, j[0], j[1], seed1)
                        # print(train_feature)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)

                        layer = []
                        # layer.append([5, 5])
                        if model == 'energy-increasing':
                            if i == [2]:
                                layer.append([2, 2])
                            elif i == [4]:
                                layer.append([6, 6])
                            elif i == [5]:
                                layer.append([5, 3])
                            elif i == [13]:
                                layer.append([1, 3])
                            elif i == [18]:
                                layer.append([9, 9])
                            elif i == [19]:
                                layer.append([10, 10])
                            elif i == [20]:
                                layer.append([5, 5])
                            else:
                                layer.append([7, 7])
                        if model == 'energy-holding':
                            if i == [3] or i == [18]:
                                layer.append([6, 6])
                            elif i == [4] or i == [20]:
                                layer.append([4, 4])
                            elif i == [17]:
                                layer.append([3, 5])
                            elif i == [2] or i == [13] or i == [17]:
                                layer.append([5, 5])
                            else:
                                layer.append([3, 3])
                        if model == 'time':
                            layer.append([5, 5])
                        if model == 'start-temperature':
                            if i == [1] or i == [5]:
                                layer.append([1, 3])
                            elif i == [3] or i == [13]:
                                layer.append([3, 1])
                            elif i == [4] or i == [17] or i == [20]:
                                layer.append([3, 3])
                            elif i == [2]:
                                layer.append([3, 5])
                            else:
                                layer.append([5, 5])

                        save_loc = None
                        if save_model:
                            save_loc = str(model) + '_' + str(i[0])

                        # MLP
                        # for hidden, unit in [[5, 5]]:
                        for hidden, unit in layer:
                            print('seed : ', seed1, 'epoch : ', epoch, 'unit : ', unit, 'hidden : ', hidden)
                            s1, mlp_test_pred, mlp_train_pred, data_model = MLP(train_feature, train_label, test_feature,
                                                                           test_label, epoch=epoch, unit=unit,
                                                                           hidden=hidden, save=save_loc)
                            out.append(s1)
                            df_new.loc[seed1 - seed_start, j[3] + '_MLP_' + str(hidden) + '_' + str(unit) + '_' + j[0]] \
                                = out[len(out) - 1]
                            df_complete.loc[seed1 - seed_start, str(i) + '_' + j[3] + '_MLP_' + str(hidden) + '_' + str(unit) + '_' + j[0]] \
                                = out[len(out) - 1]

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature, train_label, test_feature, test_label)
                        out2.append(mean_absolute_percentage_error(test_label, knn_test_pred))
                        df_new.loc[seed1 - seed_start, j[3] + '_KNN_' + j[0]] = out2[len(out2) - 1]
                        df_complete.loc[seed1 - seed_start, str(i) + '_' + j[3] + '_KNN_' + j[0]] = out2[len(out2) - 1]

                        # Linear Regression
                        # lr_test_pred, lr_train_pred, lrparam = linear_reg(train_feature, train_label, test_feature, test_label)
                        # out4.append(mean_absolute_percentage_error(test_label, lr_test_pred))
                        # df_new.loc[seed1 - seed_start, j[3] + '_LinReg_' + j[0]] = out4[len(out4) - 1]
                        # df_complete.loc[seed1 - seed_start, str(i) + '_' + j[3] + '_LinReg_' + j[0]] = out4[len(out4) - 1]

                        # Decision Tree
                        # decision_tree_test_pred, decision_tree_train_pred, dt1 = decision_tree_reg(train_feature, train_label, test_feature, test_label)
                        # out3.append(mean_absolute_percentage_error(test_label, decision_tree_test_pred))
                        # df_new.loc[seed1 - seed_start, j[3] + '_DT_' + j[0]] = out3[len(out3) - 1]
                        # df_complete.loc[seed1 - seed_start, str(i) + '_' + j[3] + '_DT_' + j[0]] = out3[len(out3) - 1]

            for i0 in df_new.columns:
                print(i0)
                arr_avg = []
                for i01, ro2 in df_new.iterrows():
                    if not pd.isna(df_new.loc[i01, i0]):
                        arr_avg.append(float(df_new.loc[i01, i0]))
                print(arr_avg)
                df_new.loc[seed_end - seed_start, i0] = np.average(arr_avg)
            df_new = df_new.rename(index={seed_end - seed_start: 'average'})
            df_new.to_csv(base_path + 'model_result/model_result_' + str(epoch) + '_' + str(model) + '/' +
                          str(i2[0]) + '/result_' + str(i) + '1.csv', encoding='euc-kr')

        for i0 in df_complete.columns:
            # print(i0)
            arr_avg = []
            for i01, ro2 in df_complete.iterrows():
                if not pd.isna(df_complete.loc[i01, i0]):
                    arr_avg.append(float(df_complete.loc[i01, i0]))
            # print(arr_avg)
            df_complete.loc[seed_end - seed_start, i0] = np.average(arr_avg)
        df_complete = df_complete.rename(index={seed_end - seed_start: 'average'})
        df_complete.to_csv(base_path + 'model_result/model_result_' + str(epoch) + '_' + str(model) + '/' +
                      str(i2[0]) + '/result_complete1.csv', encoding='euc-kr')


def HF_learning_result_check(model):
    feature_list = ''
    if model == 'energy-increasing':
        print('energy-increasing')
        feature_list = feature_list_0325_3_3
    elif model == 'energy-holding':
        print('energy-holding')
        feature_list = feature_list_0325_3_3
    elif model == 'time':
        print('time')
        feature_list = feature_list_0325_3_4
    else:
        print('wrong model')
        exit(0)

    epoch = 20000
    seed_start = 10
    seed_end = 20

    for i2 in path_1:
        # for i in [p_bum[4]]:
        for i in p_bum:
            df_origin = pd.read_csv(base_path + 'analysis/for_learning_' + str(model) + '/' + str(i2[0]) +
                                    '/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            print(i2, i, '개수', len(df_origin.index))
            df_result = pd.DataFrame()
            for seed1 in range(seed_start, seed_end):
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list:  # Feature list
                    for j in j2:
                        out = []
                        out2 = []
                        out3 = []
                        y_feature = y[j[1]]
                        y_feature = y_feature.reset_index(drop=True)
                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], j[2], seed1)
                        # data_manipulate_no_split(df_origin, j[0], j[1])
                        # data_manipulate_pca(origin2, j[0], j[1], seed1)
                        print(train_feature)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)

                        # MLP
                        for hidden, unit in [[5, 5]]:
                            print('seed : ', seed1, 'epoch : ', epoch, 'unit : ', unit, 'hidden : ', hidden)
                            s1, mlp_test_pred, mlp_train_pred, data_model = MLP(train_feature, train_label, test_feature,
                                                                           test_label, epoch=epoch, unit=unit,
                                                                           hidden=hidden)
                            out.append(s1)

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature, train_label, test_feature, test_label)
                        out2.append(mean_absolute_percentage_error(test_label, knn_test_pred))

                        print('path : ', i2, ' p_bum : ', i)
                        print('seed : ', seed1, ' feature_list : ', j2)

                        # df_result.append(y_feature)
                        df_result = df_result.append(
                            pd.DataFrame(data=y_feature, columns=y_feature.columns), ignore_index=True)
                        # df_result.reset_index(drop=True)
                        z = len(df_result.index) - len(test_label.index)
                        for k in range(len(test_label.index)):
                            df_result.loc[z, 'REAL'] = test_label['에너지'].loc[k]
                            df_result.loc[z, 'MLP'] = mlp_test_pred[k]
                            df_result.loc[z, 'error_MLP'] = abs(test_label['에너지'].loc[k] - mlp_test_pred[k])
                            df_result.loc[z, 'KNN'] = knn_test_pred[k]
                            df_result.loc[z, 'error_KNN'] = abs(test_label['에너지'].loc[k] - knn_test_pred[k])
                            start_temp = df_result.loc[z, '시작온도']
                            if start_temp > 1000:
                                df_result.loc[z, 'Type'] = 'Holding Phase'
                            else:
                                df_result.loc[z, 'Type'] = 'Holding Period'
                            z += 1
            df_result.to_csv(base_path + 'model_result/check_result_' + str(epoch) + '_' + str(model) +'/' +
                          str(i2[0]) + '/result_' + str(i) + '1.csv', encoding='euc-kr')


def wrapper_feature_selection(model):
    feature_list = ''
    if model == 'energy-increasing':
        print('energy-increasing')
        feature_list = feature_list_0325_3_3
    elif model == 'energy-holding':
        print('energy-holding')
        feature_list = feature_list_0325_3_3
    elif model == 'time':
        print('time')
        feature_list = feature_list_0325_3_4
    elif model == 'start-temperature':
        print('start-temperature')
        feature_list = feature_list_0325_3_5
    else:
        print('wrong model')
        exit(0)

    epoch = 5000
    seed_start = random.randint(1, 10000)
    seed_end = seed_start + 5

    for i2 in path_1:
        df_complete = pd.DataFrame()
        for i in p_bum:
            df_origin = pd.read_csv(base_path + 'analysis/for_learning_' + str(model) + '/' + str(i2[0]) +
                                    '/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            print(i2, i, '개수', len(df_origin.index))
            df_new = pd.DataFrame()
            for seed1 in range(seed_start, seed_end):
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list:  # Feature list
                    for j in j2:
                        # feature = df_origin[j[1]]
                        # label = df_origin[j[0]]
                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], None, seed1)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)
                        label = train_label[j[0]]

                        # Sequential Forward Selection(sfs) - MLP
                        my_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
                        if model == 'time' or model == 'start-temperature':
                            sfs = SFS(MLPRegressor(hidden_layer_sizes=(5,), max_iter=epoch), k_features=(1, 5),
                                      forward=True, floating=False, scoring=my_scorer, cv=3)
                            sbs = SFS(MLPRegressor(hidden_layer_sizes=(5,), max_iter=epoch), k_features=(1, 5),
                                      forward=False, floating=False, scoring=my_scorer, cv=3)
                        else:
                            sfs = SFS(MLPRegressor(hidden_layer_sizes=(5, ), max_iter=epoch), k_features=(1, 6),
                                      forward=True, floating=False, scoring=my_scorer, cv=3)
                            sbs = SFS(MLPRegressor(hidden_layer_sizes=(5, ), max_iter=epoch), k_features=(1, 6),
                                      forward=False, floating=False, scoring=my_scorer, cv=3)
                        sfs.fit(train_feature, label)
                        sbs.fit(train_feature, label)
                        feature_result_mlp_sfs = sfs.k_feature_names_
                        feature_result_mlp_sbs = sbs.k_feature_names_

                        train_feature_sfs = train_feature[list(feature_result_mlp_sfs)]
                        test_feature_sfs = test_feature[list(feature_result_mlp_sfs)]
                        train_feature_sbs = train_feature[list(feature_result_mlp_sbs)]
                        test_feature_sbs = test_feature[list(feature_result_mlp_sbs)]

                        # MLP
                        for hidden, unit in [[5, 5]]:
                            s1, mlp_test_pred, mlp_train_pred, data_model = \
                                MLP(train_feature_sfs, train_label, test_feature_sfs, test_label,
                                    epoch=epoch, unit=unit, hidden=hidden)
                            df_new.loc[seed1 - seed_start, 'MLP_forward'] = str(list(feature_result_mlp_sfs))
                            df_new.loc[seed1 - seed_start, 'f_' + j[3] + '_MLP_' + str(hidden) + '_' + str(unit) +
                                       '_' + j[0]] = s1
                            df_complete.loc[seed1 - seed_start, str(i) + '_MLP_forward'] = str(list(feature_result_mlp_sfs))
                            df_complete.loc[seed1 - seed_start, str(i) + '_f_' + j[3] + '_MLP_' + str(hidden) + '_' +
                                            str(unit) + '_' + j[0]] = s1

                            s1, mlp_test_pred, mlp_train_pred, data_model = \
                                MLP(train_feature_sbs, train_label, test_feature_sbs, test_label,
                                    epoch=epoch, unit=unit, hidden=hidden)
                            df_new.loc[seed1 - seed_start, 'MLP_backward'] = str(list(feature_result_mlp_sbs))
                            df_new.loc[seed1 - seed_start, 'b_' + j[3] + '_MLP_' + str(hidden) + '_' + str(unit) +
                                       '_' + j[0]] = s1
                            df_complete.loc[seed1 - seed_start, str(i) + '_MLP_backward'] = str(list(feature_result_mlp_sbs))
                            df_complete.loc[seed1 - seed_start, str(i) + '_b_' + j[3] + '_MLP_' + str(hidden) + '_' +
                                            str(unit) + '_' + j[0]] = s1

                        # Sequential Forward Selection(sfs) - KNN
                        my_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
                        sfs = SFS(KNeighborsRegressor(weights='distance'), k_features=(1, 5),
                                  forward=True, floating=False, scoring=my_scorer, cv=5)
                        sbs = SFS(KNeighborsRegressor(weights='distance'), k_features=(1, 5),
                                  forward=False, floating=False, scoring=my_scorer, cv=5)
                        sfs.fit(train_feature, label)
                        sbs.fit(train_feature, label)
                        feature_result_knn_sfs = sfs.k_feature_names_
                        feature_result_knn_sbs = sbs.k_feature_names_

                        train_feature_sfs = train_feature[list(feature_result_knn_sfs)]
                        test_feature_sfs = test_feature[list(feature_result_knn_sfs)]
                        train_feature_sbs = train_feature[list(feature_result_knn_sbs)]
                        test_feature_sbs = test_feature[list(feature_result_knn_sbs)]

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature_sfs, train_label, test_feature_sfs, test_label)
                        result = mean_absolute_percentage_error(test_label, knn_test_pred)
                        df_new.loc[seed1 - seed_start, 'KNN_forward'] = str(list(feature_result_knn_sfs))
                        df_new.loc[seed1 - seed_start, 'f_' + j[3] + '_KNN_' + j[0]] = result
                        df_complete.loc[seed1 - seed_start, str(i) + '_KNN_forward'] = str(list(feature_result_knn_sfs))
                        df_complete.loc[seed1 - seed_start, str(i) + '_f_' + j[3] + '_KNN_' + j[0]] = result

                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature_sbs, train_label, test_feature_sbs, test_label)
                        result = mean_absolute_percentage_error(test_label, knn_test_pred)
                        df_new.loc[seed1 - seed_start, 'KNN_backward'] = str(list(feature_result_knn_sbs))
                        df_new.loc[seed1 - seed_start, 'b_' + j[3] + '_KNN_' + j[0]] = result
                        df_complete.loc[seed1 - seed_start, str(i) + '_KNN_backward'] = str(list(feature_result_knn_sbs))
                        df_complete.loc[seed1 - seed_start, str(i) + '_b_' + j[3] + '_KNN_' + j[0]] = result

                        # Sequential Forward Selection(sfs) - Linear Regression
                        my_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
                        sfs = SFS(LinearRegression(), k_features=(1, 5),
                                  forward=True, floating=False, scoring=my_scorer, cv=5)
                        sbs = SFS(LinearRegression(), k_features=(1, 5),
                                  forward=False, floating=False, scoring=my_scorer, cv=5)
                        sfs.fit(train_feature, label)
                        sbs.fit(train_feature, label)
                        feature_result_lr_sfs = sfs.k_feature_names_
                        feature_result_lr_sbs = sbs.k_feature_names_

                        train_feature_sfs = train_feature[list(feature_result_lr_sfs)]
                        test_feature_sfs = test_feature[list(feature_result_lr_sfs)]
                        train_feature_sbs = train_feature[list(feature_result_lr_sbs)]
                        test_feature_sbs = test_feature[list(feature_result_lr_sbs)]

                        # Linear Regression
                        lr_test_pred, lr_train_pred, lrparam = linear_reg(train_feature_sfs, train_label, test_feature_sfs, test_label)
                        result = mean_absolute_percentage_error(test_label, lr_test_pred)
                        df_new.loc[seed1 - seed_start, 'LinReg_forward'] = str(list(feature_result_lr_sfs))
                        df_new.loc[seed1 - seed_start, 'f_' + j[3] + '_LinReg_' + j[0]] = result
                        df_complete.loc[seed1 - seed_start, str(i) + '_LinReg_forward'] = str(list(feature_result_lr_sfs))
                        df_complete.loc[seed1 - seed_start, str(i) + '_f_' + j[3] + '_LinReg_' + j[0]] = result

                        lr_test_pred, lr_train_pred, lrparam = linear_reg(train_feature_sbs, train_label, test_feature_sbs, test_label)
                        result = mean_absolute_percentage_error(test_label, lr_test_pred)
                        df_new.loc[seed1 - seed_start, 'LinReg_backward'] = str(list(feature_result_lr_sbs))
                        df_new.loc[seed1 - seed_start, 'b_' + j[3] + '_LinReg_' + j[0]] = result
                        df_complete.loc[seed1 - seed_start, str(i) + '_LinReg_backward'] = str(list(feature_result_lr_sbs))
                        df_complete.loc[seed1 - seed_start, str(i) + '_b_' + j[3] + '_LinReg_' + j[0]] = result

            for i0 in df_new.columns:
                print(i0)
                arr_avg = []
                for i01, ro2 in df_new.iterrows():
                    if not pd.isna(df_new.loc[i01, i0]) and not isinstance(df_new.loc[i01, i0], str):
                        arr_avg.append(float(df_new.loc[i01, i0]))
                print(arr_avg)
                df_new.loc[seed_end - seed_start, i0] = np.average(arr_avg)
            df_new = df_new.rename(index={seed_end - seed_start: 'average'})
            df_new.to_csv(base_path + 'model_result/model_result_' + str(epoch) + '_' + str(model) + '/' +
                          str(i2[0]) + '/result_' + str(i) + '1.csv', encoding='euc-kr')
        for i0 in df_complete.columns:
            # print(i0)
            arr_avg = []
            for i01, ro2 in df_complete.iterrows():
                if not pd.isna(df_complete.loc[i01, i0]) and not isinstance(df_complete.loc[i01, i0], str):
                    arr_avg.append(float(df_complete.loc[i01, i0]))
            # print(arr_avg)
            df_complete.loc[seed_end - seed_start, i0] = np.average(arr_avg)
        df_complete = df_complete.rename(index={seed_end - seed_start: 'average'})
        df_complete.to_csv(base_path + 'model_result/model_result_' + str(epoch) + '_' + str(model) + '/' +
                      str(i2[0]) + '/result_complete1.csv', encoding='euc-kr')


def make_rule(model):
    for i2 in path_1:
        df_complete = pd.DataFrame()
        # for i in [p_bum[4]]:
        for i in p_bum:
            df_origin = pd.read_csv(base_path + 'analysis/for_learning_' + str(model) + '/' + str(i2[0]) +
                                    '/' + str(i) + '.csv', encoding='euc-kr', index_col=0)
            df_origin['velocity'] = ''
            df_origin['constant'] = ''
            df_origin['prediction'] = ''
            df_origin['error'] = ''
            for a, row in df_origin.iterrows():
                start = df_origin['시작온도'].loc[a]
                end = df_origin['종료온도'].loc[a]
                time = df_origin['시간(총)'].loc[a]
                df_origin['velocity'].loc[a] = (end - start) / time

            # sort by 시작온도
            df_origin = df_origin.sort_values(['시작온도', '종료온도'], ascending=[True, True])

            df_result = pd.DataFrame(columns=['시작온도', '종료온도', 'constant', 'count'])
            flag_start = None
            flag_end = None
            temp_start = []
            temp_end = []
            temp_velocity = []
            tolerance = 50
            for j, row in df_origin.iterrows():
                if flag_start == None:
                    flag_start = df_origin['시작온도'].loc[j]
                    flag_end = df_origin['종료온도'].loc[j]
                    temp_start.append(df_origin['시작온도'].loc[j])
                    temp_end.append(df_origin['종료온도'].loc[j])
                    temp_velocity.append(df_origin['velocity'].loc[j])
                else:
                    start = df_origin['시작온도'].loc[j]
                    end = df_origin['종료온도'].loc[j]
                    if abs(start - flag_start) <= tolerance and abs(end - flag_end) <= tolerance:
                        temp_start.append(df_origin['시작온도'].loc[j])
                        temp_end.append(df_origin['종료온도'].loc[j])
                        temp_velocity.append(df_origin['velocity'].loc[j])

                        flag_start = np.mean(temp_start)
                        flag_end = np.mean(temp_end)
                    else:
                        df_result = df_result.append(pd.DataFrame(data=np.array([[np.mean(temp_start), np.mean(temp_end), np.mean(temp_velocity), len(temp_velocity)]]), columns=['시작온도', '종료온도', 'constant', 'count']))

                        temp_start = []
                        temp_end = []
                        temp_velocity = []
                        flag_start = df_origin['시작온도'].loc[j]
                        flag_end = df_origin['종료온도'].loc[j]
                        temp_start.append(df_origin['시작온도'].loc[j])
                        temp_end.append(df_origin['종료온도'].loc[j])
                        temp_velocity.append(df_origin['velocity'].loc[j])
            df_result = df_result.append(pd.DataFrame(data=np.array([[np.mean(temp_start), np.mean(temp_end), np.mean(temp_velocity), len(temp_velocity)]]), columns=['시작온도', '종료온도', 'constant', 'count']))
            df_result = df_result.reset_index(drop=True)
            df_result.to_csv(base_path + 'model_result/model_result_' + str(model) + '/' + str(i2[0]) + '/model_' + str(model) + '_' + str(i[0]) + '.csv', encoding='euc-kr')

            # data check
            print('original data :', len(df_origin))
            count = 0
            for l, row in df_result.iterrows():
                count += df_result['count'].loc[l]
            print('result data :', int(count))

            df_origin['mape'] = ''
            for k, row in df_origin.iterrows():
                start = df_origin['시작온도'].loc[k]
                end = df_origin['종료온도'].loc[k]
                diff = abs(end - start)
                constant = None
                flag_diff = 1000
                for n, bar in df_result.iterrows():
                    res_start = df_result['시작온도'].loc[n]
                    res_end = df_result['종료온도'].loc[n]
                    euc_one = (start, end)
                    euc_two = (res_start, res_end)
                    cur_diff = distance.euclidean(euc_one, euc_two)
                    if cur_diff < flag_diff:
                        flag_diff = cur_diff
                        constant = df_result['constant'].loc[n]
                true_time = df_origin['시간(총)'].loc[k]
                prediction = diff / constant
                df_origin['constant'].loc[k] = constant
                df_origin['prediction'].loc[k] = prediction
                df_origin['error'].loc[k] = abs(true_time - prediction)
                df_origin['mape'].loc[k] = abs(true_time - prediction) / true_time * 100
            df_origin = df_origin.reset_index(drop=True)
            df_origin = df_origin[['시작온도', '종료온도', '시간(총)', 'velocity', 'constant', 'prediction', 'error', 'mape']]
            for i0 in df_origin.columns:
                if i0 == 'mape':
                    arr_avg = []
                    for i01, ro2 in df_origin.iterrows():
                        arr_avg.append(float(df_origin.loc[i01, i0]))
                    # print(arr_avg)
                    df_origin.loc[len(df_origin), i0] = np.average(arr_avg)
            df_origin.to_csv(base_path + 'model_result/model_result_' + str(model) + '/' + str(i2[0]) + '/model_result_' + str(i[0]) + '.csv', encoding='euc-kr')
            print('end -', i, '\n')


if __name__ == '__main__':
    # get_data()
    # summarize_heating_data(

    # plot_heating_data(view=True)
    # check_data()
    # check_real_start_time()

    # work_press2()     # optional

    # 0 = all, 1 = heating curve type 1, 3 = heating curve type 2, 5 = heating curve type 3, 10 = strange heating curve
    # work_set2(curve_type=0)

    model = 'start-temperature'   # energy-increasing, energy-holding, time, start-temperature
    # model = 'time'

    # make_heat_or_hold(model=model)
    # furnace_clustering2(model=model)
    # HF_learning(model=model, save_model=True)

    make_rule('time')

    # wrapper_feature_selection(model=model)

    # HF_learning_result_check(model=model)
