from module import *
import pandas as pd


def plot_holding_temperatures(df, data, change_point, phase_list_dict):
    hold = phase_list_dict['hold']
    dt = len(df.index)
    for i in range(len(hold)):
        drop_flag = 0
        tent_gas = []
        tent_tem = []
        for j in range(hold[i][0] + 1, hold[i][1] + 1):
            tent_gas.append(data[j]['GAS'])
        for k in range(hold[i][0], hold[i][1] + 1):
            tent_tem.append(data[k]['TEMPERATURE'])
        start = data[hold[i][0]]['TEMPERATURE']
        end = data[hold[i][1]]['TEMPERATURE']
        if data[hold[i][0]]['GAS_OFF'] == 1 or data[hold[i][1]]['GAS_OFF'] == 1 or data[hold[i][0]]['TEMP_OFF'] == 1 or data[hold[i][1]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(hold), i)
        field = [data[hold[i][0]]['TIME'], start, data[hold[i][1]]['TIME'], end, np.mean(tent_tem), np.sum(tent_gas)]
        df.loc[dt] = field
        dt += 1


def plot_heating_flat(df, data, change_point, phase_list_dict):
    heat = phase_list_dict['heat']
    dt = 0
    for i in range(len(heat)):
        start_temp = None
        end_temp = None
        start_time = None
        end_time = None
        tent_gas = []
        tent_tem = []
        for j in range(heat[i][0], heat[i][1] + 1):
            # test = change_point[hold[i][0]:hold[i][1]+1]
            if change_point[j] != None:
                if start_temp == None:
                    start_temp = data[j]['TEMPERATURE']
                    start_time = data[j]['TIME']
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])
                else:
                    end_temp = data[j]['TEMPERATURE']
                    end_time = data[j]['TIME']
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])
                    gas = np.sum(tent_gas)
                    temp = np.mean(tent_tem)
                    time = (end_time - start_time).total_seconds() // 60

                    # Only for detecting flat / holding phase
                    # if abs(end_temp - start_temp) < 10:
                    #     field = [start_temp, end_temp, time, gas]
                    #     df.loc[dt] = field
                    #     dt += 1

                    field = [start_temp, end_temp, time, gas]
                    df.loc[dt] = field
                    dt += 1

                    start_temp = data[j]['TEMPERATURE']
                    end_temp = None
                    tent_gas = []
                    tent_tem = []
                    # tent_gas.append(data[j]['GAS'])
                    # tent_tem.append(data[j]['TEMPERATURE'])
            else:
                if start_temp != None:
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])


def divide_data():
    # Heating data
    # df = pd.DataFrame(columns=['시작온도', '종료온도', '총시간', '가스사용량'])
    # df.reset_index(drop=True)

    # Holding data
    df2 = pd.DataFrame(columns=['시작시간', '시작온도', '종료시간', '종료온도', '가스사용량', '평균온도'])
    df2.reset_index(drop=True)
    for num in work_:
        data = []
        change_point = []
        for t in os.listdir(base_path + 'input/' + str(num) + '/'):
            path = base_path + 'input/' + str(num) + '/' + t
            print(path)
            get_data_excel(data, path, num)
        start_real, end_real = fc.data_manipulates(data, num, time_path)
        time_dict, phase_list_dict = fc.find_all(data, change_point, num, start_real, end_real)
        # Heating data
        # plot_heating_flat(df, data, change_point, phase_list_dict)

        # Holding data
        plot_holding_temperatures(df2, data, change_point, phase_list_dict)
    # Heating data
    # df.to_csv(base_path + "HF_OUT/hf_heating.csv", mode='w', encoding='euc-kr')

    # Holding data
    df2.to_csv(base_path + "HF_OUT/hf_holding_summary.csv", mode='w', encoding='euc-kr')


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
        # print("find change_point done heating " + str(num))
        plotting(data, change_point, time_dict['fixed_start_time_list'], time_dict['fixed_end_time_list'], num,
                 time_dict['heat_ended_time_list'], time_dict['real_start_time_list'], time_dict['real_end_time_list'])
        # make_database(data, num, h, phase_list_dict)
        make_database2(data, num, h, change_point, phase_list_dict)
        h.sett(df_mat, base_path + 'HF_OUT/hf_2019_')
        print('DB_Done')
    if view:
        plt.show()


# 프레스기 매칭
def work_press2():
    h_arr = []
    p = pd.read_csv(base_path + 'data/' + work_space + 'press_par.csv', encoding='euc-kr', index_col=0)
    for i in work_:
        h = HF()
        h.df = pd.read_csv(base_path + 'HF_OUT/hf_2019_' + str(i) + '.csv', encoding='euc-kr', index_col=0)
        h.set_next_h()
        h.change_list()
        h_arr.append(h)
    pm.matching_press_general(h_arr, p)
    for i in h_arr:
        i.out(base_path + 'HF_OUT/press_2019_')


# 구간 정리
def work_set2():
    hh = HF()
    df_t = pd.read_csv(base_path + 'data/start_end_re1.csv', encoding='euc-kr')
    # for num in error_arr_2019:
    for num in work_:
        h = HF()
        h.df = pd.read_csv(base_path + 'HF_OUT/press_2019_' + str(num) + '.csv', encoding='euc-kr', index_col=0)
        # eliminate_error_loop(h, num[1])
        h.match_time(df_t)
        print('end time matching')
        h.fill()
        print('end fill items')
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
    hh2.df = pd.read_csv(base_path + 'HF_OUT/last_2019_ffa' + str(work_[0]) + '.csv', encoding='euc-kr', index_col=0)
    handle_first_hold(hh2, work_, base_path + 'HF_OUT/last_2019_' + str(work_[0]) + '_first_hold.csv')
    hh2.out(base_path + 'HF_OUT/last_2019_' + str(work_[0]) + '_drop_first_hold')


def HF_heating_learning():
    epoch = 3000
    seed_start = 10
    seed_end = 20

    for i2 in path_1:
        print(i2)
        # for i in [p_bum[4]]:
        for i in p_bum:
            df_origin = pd.read_csv(base_path + 'analysis/for_learning/' + str(i2[0]) + '/' + str(i) + '.csv',
                                    encoding='euc-kr', index_col=0)
            print(i2, i, '개수', len(df_origin.index))
            df_new = pd.DataFrame()
            for seed1 in range(seed_start, seed_end):
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list_0325_2:
                    for j in j2:
                        out = []
                        out2 = []
                        out3 = []
                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], j[2], seed1)
                        # data_manipulate_no_split(df_origin, j[0], j[1])
                        # data_manipulate_pca(origin2, j[0], j[1], seed1)
                        print(train_feature)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature, train_label, test_feature, test_label)
                        out2.append(mean_absolute_percentage_error(test_label, knn_test_pred))

                        # MLP
                        for hidden, unit in [[5, 5]]:
                            print('seed : ', seed1, 'epoch : ', epoch, 'unit : ', unit, 'hidden : ', hidden)
                            s1, mlp_test_pred, mlp_train_pred, model = MLP(train_feature, train_label, test_feature,
                                                                           test_label, epoch=epoch, unit=unit,
                                                                           hidden=hidden)
                            out.append(s1)
                            df_new.loc[
                                seed1 - seed_start, j[3] + '_MLP_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = out[
                                len(out) - 1]

                        df_new.loc[seed1 - seed_start, j[3] + '_KNN_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = \
                        out2[len(out2) - 1]

                        # Decision Tree
                        decision_tree_test_pred, decision_tree_train_pred, dt1 = decision_tree_reg(train_feature, train_label, test_feature, test_label)
                        out3.append(mean_absolute_percentage_error(test_label, decision_tree_test_pred))

                        df_new.loc[seed1 - seed_start, j[3] + '_DTREE_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = \
                        out3[len(out3) - 1]

            for i0 in df_new.columns:
                print(i0)
                arr_avg = []
                for i01, ro2 in df_new.iterrows():
                    if not pd.isna(df_new.loc[i01, i0]):
                        arr_avg.append(float(df_new.loc[i01, i0]))
                print(arr_avg)
                df_new.loc[seed_end - seed_start, i0] = np.average(arr_avg)
            df_new = df_new.rename(index={seed_end - seed_start: 'average'})
            df_new.to_csv(base_path + 'model_result/' + str(i2[0]) + '/result_' + str(i) + '1.csv', encoding='euc-kr')


if __name__ == '__main__':
    divide_data()
    # summarize_heating_data()

    # plot_heating_data(view=False)
    # work_press2()
    # work_set2()
    # make_heat()
    # furnace_clustering()
    # HF_heating_learning()

