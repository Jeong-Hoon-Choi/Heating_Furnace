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

                    if abs(end_temp - start_temp) < 10:
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

    datas.to_csv(base_path + "HF_OUT/hf_summary.csv", mode='w', encoding='euc-kr')

if __name__ == '__main__':
    plot_heating_data()
    # summarize_heating_data()

