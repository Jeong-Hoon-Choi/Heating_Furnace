from module import *
import pandas as pd
import glob

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
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])
            else:
                if start_temp != None:
                    tent_gas.append(data[j]['GAS'])
                    tent_tem.append(data[j]['TEMPERATURE'])

def plot_heating_data():
    for num in work_:
        df = pd.DataFrame(columns=['시작온도', '종료온도', '총시간', '가스사용량'])
        df.reset_index(drop=True)
        data = []
        change_point = []
        for t in os.listdir(base_path + 'input/' + str(num) + '/'):
            path = base_path + 'input/' + str(num) + '/' + t
            print(path)
            get_data_excel(data, path, num)
        start_real, end_real = fc.data_manipulates(data, num, time_path)
        time_dict, phase_list_dict = fc.find_all(data, change_point, num, start_real, end_real)
        # plotting(data, change_point, time_dict['fixed_start_time_list'], time_dict['fixed_end_time_list'], num,
        #          time_dict['heat_ended_time_list'], time_dict['real_start_time_list'], time_dict['real_end_time_list'])
        # plt.show()
        plot_heating_flat(df, data, change_point, phase_list_dict)
        df.to_csv(base_path + "HF_OUT/hf_2019_" + str(num) + ".csv", mode='w', encoding='euc-kr')

def summarize_heating_data():
    all_datas = glob.glob(os.path.join(base_path + "HF_OUT/", "hf_2019_*"))
    li = []
    for filename in all_datas:
        df = pd.read_csv(filename, index_col=0, header=0, encoding="euc_kr")
        df.columns = [c.strip().lower().replace(' ', '') for c in df.columns]
        li.append(df)

    datas = pd.concat(li, axis=0, ignore_index=True)

    bins = []
    labels = []
    for x in range(0, 1400, 10):
        y = x + 10
        bins.append(x)
        labels.append(str(x) + " - " + str(y))
    bins.append(np.inf)

    datas = datas.groupby(pd.cut(datas['시작온도'], bins=bins, labels=labels)).size().reset_index(name='count')
    datas = datas[datas['count'] != 0]

    datas.to_csv(base_path + "HF_OUT/hf_summary.csv", mode='w', encoding='euc-kr')\

def HF_heating_learning():
    epoch = 2000
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
                test_pred = {'시간(총)': '', '시간(0제외)': '', '에너지': '', '시간(총)_mape': '', '시간(0제외)_mape': '',
                             '에너지_mape': ''}
                train_pred = {'시간(총)': '', '시간(0제외)': '', '에너지': '', '시간(총)_mape': '', '시간(0제외)_mape': '',
                              '에너지_mape': ''}
                x, y = Train_Test_split(df_origin, seed1)
                for j2 in feature_list_0325_2:
                    for j in j2:
                        out = []
                        out2 = []
                        train_feature, train_label, test_feature, test_label = \
                            data_manipulate_normal3(x, y, j[0], j[1], j[2], seed1)
                        # data_manipulate_no_split(df_origin, j[0], j[1])
                        # data_manipulate_pca(origin2, j[0], j[1], seed1)
                        print(train_feature)
                        train_label = train_label.reset_index(drop=True)
                        test_label = test_label.reset_index(drop=True)

                        # KNN
                        knn_test_pred, knn_train_pred, k1 = KNN_reg(train_feature, train_label, test_feature,
                                                                    test_label)
                        result = mean_absolute_percentage_error(train_label, knn_train_pred)
                        result2 = mean_absolute_percentage_error(test_label, knn_test_pred)
                        out2.append(mean_absolute_percentage_error(test_label, knn_test_pred))

                        # MLP
                        for hidden, unit in [[5, 5]]:
                            print('seed : ', seed1, 'epoch : ', epoch, 'unit : ', unit, 'hidden : ', hidden)
                            s1, mlp_test_pred, mlp_train_pred, model = MLP(train_feature, train_label, test_feature,
                                                                           test_label, epoch=epoch, unit=unit,
                                                                           hidden=hidden)
                            train_pred[j[0]] = mlp_train_pred
                            test_pred[j[0]] = mlp_test_pred
                            out.append(s1)
                            df_new.loc[
                                seed1 - seed_start, j[3] + '_MLP_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = out[
                                len(out) - 1]

                        df_new.loc[seed1 - seed_start, j[3] + '_KNN_' + str(hidden) + '_' + str(unit) + '_' + j[0]] = \
                        out2[len(out2) - 1]

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
    # plot_heating_data()
    # summarize_heating_data()

    # work_start(view=False)
    # work_press()
    # work_set()
    # make_heat()
    # furnace_clustering()
    HF_heating_learning()

