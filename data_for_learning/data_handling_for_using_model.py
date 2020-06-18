from bases import *


# make furnace's data
def make_database(data, num, h, phase_list_dict):
    op_en = phase_list_dict['open']
    heat = phase_list_dict['heat']
    hold = phase_list_dict['hold']
    # df = pd.DataFrame(columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type'])
    # heat
    for i in range(len(heat)):
        drop_flag = 0
        tent_gas = []
        tent_gas2 = []
        if heat[i][7] is None:
            for j in range(heat[i][0] + 1, heat[i][1] + 1):
                tent_gas.append(data[j]['GAS'])
            for j in range(heat[i][0] + 1, heat[i][1] + 1):
                tent_gas2.append(data[j]['GAS'])
        else:
            for j in range(heat[i][7] + 1, heat[i][1] + 1):
                tent_gas.append(data[j]['GAS'])
            for j in range(heat[i][0] + 1, heat[i][1] + 1):
                tent_gas2.append(data[j]['GAS'])
        # print(len(heat), i)
        if data[heat[i][0]]['GAS_OFF'] == 1 or data[heat[i][1]]['GAS_OFF'] == 1 or\
                data[heat[i][0]]['TEMP_OFF'] == 1 or data[heat[i][1]]['TEMP_OFF'] == 1 or heat[i][8] == 0 or\
                heat[i][8] is None:
            drop_flag = 1
        h.df = h.df.append(
            pd.DataFrame(data=np.array([[num, data[heat[i][0]]['TIME'], data[heat[i][1]]['TIME'], np.sum(tent_gas),
                                         'heat', data[heat[i][2]]['TIME'], data[heat[i][0]]['TEMPERATURE'], data[heat[i][1]]['TEMPERATURE'], heat[i][3],
                                         data[heat[i][2]]['TEMPERATURE'], np.sum(tent_gas2), heat[i][4], heat[i][5],
                                         heat[i][6], heat[i][8], data[heat[i][1]]['TEMPERATURE'], i, drop_flag]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량(마지막 구간)',
                                  'Type', '실제 시작시간', '시작온도', '종료온도', '뺄시간',
                                  '원래시작점온도', '가스사용량', '가열중 문열림 횟수', '가열중 마지막 문닫힌 시간',
                                  '최종 가열시작 온도', '이전 종료시간', '가열완료 온도', 'cycle', 'drop_flag']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # hold
    for i in range(len(hold)):
        drop_flag = 0
        tent_gas = []
        tent_tem = []
        for j in range(hold[i][0] + 1, hold[i][1] + 1):
            tent_gas.append(data[j]['GAS'])
        for k in range(hold[i][0], hold[i][1] + 1):
            tent_tem.append(data[k]['TEMPERATURE'])
        if data[hold[i][0]]['GAS_OFF'] == 1 or data[hold[i][1]]['GAS_OFF'] == 1 or data[hold[i][0]]['TEMP_OFF'] == 1 or data[hold[i][1]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(hold), i)
        h.df = h.df.append(
            pd.DataFrame(data=np.array(
                [[num, data[hold[i][0]]['TIME'], data[hold[i][1]]['TIME'], np.sum(tent_gas), 'hold', np.mean(tent_tem), drop_flag]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '평균온도', 'drop_flag']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # open
    for i in range(len(op_en)):
        drop_flag = 0
        tent_gas = []
        for j in range(op_en[i][0] + 1, op_en[i][1] + 1):
            tent_gas.append(data[j]['GAS'])
        if data[op_en[i][0]]['GAS_OFF'] == 1 or data[op_en[i][1]]['GAS_OFF'] == 1 or data[op_en[i][0]]['TEMP_OFF'] == 1 or data[op_en[i][1]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(op_en), i)
        h.df = h.df.append(
            pd.DataFrame(data=np.array([[num, data[op_en[i][0]]['TIME'], data[op_en[i][1]]['TIME'],
                                         np.sum(tent_gas), 'open', data[op_en[i][0]]['TEMPERATURE'], data[op_en[i][1]]['TEMPERATURE'], drop_flag]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '시작온도', '종료온도', 'drop_flag']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # close
    for i in range(len(op_en)):
        drop_flag = 0
        tent_gas = []
        time_out = 0
        for j in range(op_en[i][1] + 1, op_en[i][2] + 1):
            tent_gas.append(data[j]['GAS'])
            if int(data[j]['GAS']) == 0 and data[j]['GAS_OFF'] == 0:
                time_out += 1
        if data[op_en[i][1]]['GAS_OFF'] == 1 or data[op_en[i][2]]['GAS_OFF'] == 1 or data[op_en[i][1]]['TEMP_OFF'] == 1 or data[op_en[i][2]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(op_en), i)
        h.df = h.df.append(
            pd.DataFrame(data=np.array(
                [[num, data[op_en[i][1]]['TIME'], data[op_en[i][2]]['TIME'], np.sum(tent_gas), 'reheat', data[op_en[i][1]]['TEMPERATURE'], drop_flag, time_out]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '시작온도', 'drop_flag', '뺄시간']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # display(df)
    # df.to_csv("test_201901_2.csv", mode='w', encoding='euc-kr')


# make furnace's data - Steven
def make_database2(data, num, h, change_point, phase_list_dict):
    op_en = phase_list_dict['open']
    heat = phase_list_dict['heat']
    hold = phase_list_dict['hold']
    # df = pd.DataFrame(columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type'])
    # heat
    for i in range(len(heat)):
        first_section = None
        start = None
        end = None
        drop_flag = 0
        tent_gas = []
        tent_gas2 = []

        if heat[i][7] is not None:
            first_section = heat[i][7]

        count = change_point[heat[i][0]:heat[i][1] + 1]
        if len([x for x in count if x is not None]) == 2 and heat[i][0] != heat[i][1] and data[heat[i][1]]['TEMPERATURE'] - data[heat[i][0]]['TEMPERATURE'] >= 50:
            drop_flag = 0
            tent_gas = []
            tent_gas2 = []
            if heat[i][7] is None:
                for j in range(heat[i][0] + 1, heat[i][1] + 1):
                    tent_gas.append(data[j]['GAS'])
                for j in range(heat[i][0] + 1, heat[i][1] + 1):
                    tent_gas2.append(data[j]['GAS'])
            else:
                for j in range(heat[i][7] + 1, heat[i][1] + 1):
                    tent_gas.append(data[j]['GAS'])
                for j in range(heat[i][0] + 1, heat[i][1] + 1):
                    tent_gas2.append(data[j]['GAS'])
            # print(len(heat), i)
            if data[heat[i][0]]['GAS_OFF'] == 1 or data[heat[i][1]]['GAS_OFF'] == 1 or \
                    data[heat[i][0]]['TEMP_OFF'] == 1 or data[heat[i][1]]['TEMP_OFF'] == 1 or heat[i][8] == 0 or \
                    heat[i][8] is None:
                drop_flag = 1
            h.df = h.df.append(
                pd.DataFrame(
                    data=np.array([[num, data[heat[i][0]]['TIME'], data[heat[i][1]]['TIME'], np.sum(tent_gas),
                                    'heat', data[heat[i][2]]['TIME'], data[heat[i][0]]['TEMPERATURE'],
                                    data[heat[i][1]]['TEMPERATURE'], heat[i][3],
                                    data[heat[i][2]]['TEMPERATURE'], np.sum(tent_gas2), heat[i][4], heat[i][5],
                                    heat[i][6], heat[i][8], data[heat[i][1]]['TEMPERATURE'], i, drop_flag]]),
                    columns=['가열로 번호', '시작시간', '종료시간', '가스사용량(마지막 구간)',
                             'Type', '실제 시작시간', '시작온도', '종료온도', '뺄시간',
                             '원래시작점온도', '가스사용량', '가열중 문열림 횟수', '가열중 마지막 문닫힌 시간',
                             '최종 가열시작 온도', '이전 종료시간', '가열완료 온도', 'cycle', 'drop_flag']), sort=True)
            h.df = h.df.reset_index(drop=True)
            continue

        # if heat[i][0] == heat[i][1]:
        #     tent_gas.append(data[j]['GAS'])
        #     tent_gas2.append(data[j]['GAS'])
        #     if data[heat[i][0]]['GAS_OFF'] == 1 or data[heat[i][1]]['GAS_OFF'] == 1 or \
        #             data[heat[i][0]]['TEMP_OFF'] == 1 or data[heat[i][1]]['TEMP_OFF'] == 1 or heat[i][8] == 0 or \
        #             heat[i][8] is None:
        #         drop_flag = 1
        #     h.df = h.df.append(
        #         pd.DataFrame(
        #             data=np.array([[num, data[heat[i][0]]['TIME'], data[heat[i][1]]['TIME'], np.sum(tent_gas),
        #                             'heat', data[heat[i][2]]['TIME'], data[heat[i][0]]['TEMPERATURE'],
        #                             data[heat[i][1]]['TEMPERATURE'], heat[i][3],
        #                             data[heat[i][2]]['TEMPERATURE'], np.sum(tent_gas2), heat[i][4], heat[i][5],
        #                             heat[i][6], heat[i][8], data[heat[i][1]]['TEMPERATURE'], i, drop_flag]]),
        #             columns=['가열로 번호', '시작시간', '종료시간', '가스사용량(마지막 구간)',
        #                      'Type', '실제 시작시간', '시작온도', '종료온도', '뺄시간',
        #                      '원래시작점온도', '가스사용량', '가열중 문열림 횟수', '가열중 마지막 문닫힌 시간',
        #                      '최종 가열시작 온도', '이전 종료시간', '가열완료 온도', 'cycle', 'drop_flag']), sort=True)
        #     h.df = h.df.reset_index(drop=True)
        #     break

        for j in range(heat[i][0], heat[i][1] + 1):
            if change_point[j] is not None:
                if start is None:
                    start = j
                    tent_gas.append(data[j]['GAS'])
                else:
                    if j < start + 30:
                        # start = j
                        continue

                    end = j
                    tent_gas.append(data[j]['GAS'])

                    if first_section is not None:
                        for k in range(first_section, end):
                            tent_gas2.append(data[k]['GAS'])
                        first_section = None
                    else:
                        tent_gas2 = tent_gas

                    if data[start]['GAS_OFF'] == 1 or data[end]['GAS_OFF'] == 1 or data[start]['TEMP_OFF'] == 1 or \
                            data[end]['TEMP_OFF'] == 1 or heat[i][8] == 0 or heat[i][8] is None:
                        drop_flag = 1
                    if data[end]['TEMPERATURE'] - data[start]['TEMPERATURE'] >= 50:
                        h.df = h.df.append(
                            pd.DataFrame(
                                data=np.array([[num, data[start]['TIME'], data[end]['TIME'], np.sum(tent_gas),
                                                'heat', data[heat[i][2]]['TIME'], data[start]['TEMPERATURE'],
                                                data[end]['TEMPERATURE'], heat[i][3],
                                                data[heat[i][2]]['TEMPERATURE'], np.sum(tent_gas2), heat[i][4], heat[i][5],
                                                heat[i][6], heat[i][8], data[end]['TEMPERATURE'], i, drop_flag]]),
                                columns=['가열로 번호', '시작시간', '종료시간', '가스사용량(마지막 구간)',
                                         'Type', '실제 시작시간', '시작온도', '종료온도', '뺄시간',
                                         '원래시작점온도', '가스사용량', '가열중 문열림 횟수', '가열중 마지막 문닫힌 시간',
                                         '최종 가열시작 온도', '이전 종료시간', '가열완료 온도', 'cycle', 'drop_flag']), sort=True)
                        h.df = h.df.reset_index(drop=True)

                    start = j
                    end = None
                    tent_gas = []
                    tent_gas2 = []
            else:
                if start is not None:
                    tent_gas.append(data[j]['GAS'])

        # Steven - heat[i][7] = heating_parameter_dict['heat_start_index']
        # if heat[i][7] is None:
        #     for j in range(heat[i][0] + 1, heat[i][1] + 1):
        #         tent_gas.append(data[j]['GAS'])
        #     for j in range(heat[i][0] + 1, heat[i][1] + 1):
        #         tent_gas2.append(data[j]['GAS'])
        # else:
        #     for j in range(heat[i][7] + 1, heat[i][1] + 1):
        #         tent_gas.append(data[j]['GAS'])
        #     for j in range(heat[i][0] + 1, heat[i][1] + 1):
        #         tent_gas2.append(data[j]['GAS'])
        # # print(len(heat), i)
        # if data[heat[i][0]]['GAS_OFF'] == 1 or data[heat[i][1]]['GAS_OFF'] == 1 or\
        #         data[heat[i][0]]['TEMP_OFF'] == 1 or data[heat[i][0]]['TEMP_OFF'] == 1 or heat[i][8] == 0 or\
        #         heat[i][8] is None:
        #     drop_flag = 1
        # h.df = h.df.append(
        #     pd.DataFrame(data=np.array([[num, data[heat[i][0]]['TIME'], data[heat[i][1]]['TIME'], np.sum(tent_gas),
        #                                  'heat', data[heat[i][2]]['TIME'], data[heat[i][0]]['TEMPERATURE'], data[heat[i][1]]['TEMPERATURE'], heat[i][3],
        #                                  data[heat[i][2]]['TEMPERATURE'], np.sum(tent_gas2), heat[i][4], heat[i][5],
        #                                  heat[i][6], heat[i][8], data[heat[i][1]]['TEMPERATURE'], i, drop_flag]]),
        #                  columns=['가열로 번호', '시작시간', '종료시간', '가스사용량(마지막 구간)',
        #                           'Type', '실제 시작시간', '시작온도', '종료온도', '뺄시간',
        #                           '원래시작점온도', '가스사용량', '가열중 문열림 횟수', '가열중 마지막 문닫힌 시간',
        #                           '최종 가열시작 온도', '이전 종료시간', '가열완료 온도', 'cycle', 'drop_flag']), sort=True)
        # h.df = h.df.reset_index(drop=True)
    # hold
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
        h.df = h.df.append(
            pd.DataFrame(data=np.array(
                [[num, data[hold[i][0]]['TIME'], data[hold[i][1]]['TIME'], np.sum(tent_gas), 'hold', np.mean(tent_tem), drop_flag, start, end]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '평균온도', 'drop_flag', '시작온도', '종료온도']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # open
    for i in range(len(op_en)):
        drop_flag = 0
        tent_gas = []
        for j in range(op_en[i][0] + 1, op_en[i][1] + 1):
            tent_gas.append(data[j]['GAS'])
        if data[op_en[i][0]]['GAS_OFF'] == 1 or data[op_en[i][1]]['GAS_OFF'] == 1 or data[op_en[i][0]]['TEMP_OFF'] == 1 or data[op_en[i][1]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(op_en), i)
        h.df = h.df.append(
            pd.DataFrame(data=np.array([[num, data[op_en[i][0]]['TIME'], data[op_en[i][1]]['TIME'],
                                         np.sum(tent_gas), 'open', data[op_en[i][0]]['TEMPERATURE'], data[op_en[i][1]]['TEMPERATURE'], drop_flag]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '시작온도', '종료온도', 'drop_flag']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # close
    for i in range(len(op_en)):
        drop_flag = 0
        tent_gas = []
        time_out = 0
        for j in range(op_en[i][1] + 1, op_en[i][2] + 1):
            tent_gas.append(data[j]['GAS'])
            if int(data[j]['GAS']) == 0 and data[j]['GAS_OFF'] == 0:
                time_out += 1
        if data[op_en[i][1]]['GAS_OFF'] == 1 or data[op_en[i][2]]['GAS_OFF'] == 1 or data[op_en[i][1]]['TEMP_OFF'] == 1 or data[op_en[i][2]]['TEMP_OFF'] == 1:
            drop_flag = 1
        # print(len(op_en), i)
        h.df = h.df.append(
            pd.DataFrame(data=np.array(
                [[num, data[op_en[i][1]]['TIME'], data[op_en[i][2]]['TIME'], np.sum(tent_gas), 'reheat', data[op_en[i][1]]['TEMPERATURE'], drop_flag, time_out]]),
                         columns=['가열로 번호', '시작시간', '종료시간', '가스사용량', 'Type', '시작온도', 'drop_flag', '뺄시간']), sort=True)
        h.df = h.df.reset_index(drop=True)
    # display(df)
    # df.to_csv("test_201901_2.csv", mode='w', encoding='euc-kr')


# checking that current cycle's material is same with previous cycle's material
def gum_2(h1):
    h1.change_list()
    del_arr = []
    for i, row in h1.df.iterrows():
        count = 0
        flag = 0
        for j in h1.df['소재 list'].loc[i]:
            if i > 0 and j in h1.df['소재 list'].loc[i-1]:
                flag = 1
                # del_arr.append(i)
                # print(i)
                count += 1
                # break
        if flag == 1:
            del_arr.append(i)
        h1.df.loc[i, '겹침수량'] = str(count)
        if i == 0 or h1.df.loc[i, 'cycle'] == '0':
            h1.df.loc[i, '직전 사이클 소재 수량'] = 0
        elif i > 0:
            h1.df.loc[i, '직전 사이클 소재 수량'] = len(h1.df['소재 list'].loc[i-1])
    for i, row in h1.df.iterrows():
        if i in del_arr:
            h1.df.loc[i, '겹침여부'] = '겹침'
            # h1.df.loc[i, '겹침수량'] = str()
        else:
            h1.df.loc[i, '겹침여부'] = '안겹침'
            # h1.df.loc[i, '겹침수량'] = '겹침'
    h1.df = h1.df.reset_index(drop=True)


def gum_22(h1):
    h1.set_next_h_2()
    h1.change_list()
    del_arr = []
    for i, row in h1.df.iterrows():
        count = 0
        flag = 0
        for j in h1.df['소재 list'].loc[i]:
            if i > 0 and j in h1.df['소재 list'].loc[i-1]:
                flag = 1
                # del_arr.append(i)
                # print(i)
                count += 1
                # break
        if flag == 1:
            del_arr.append(i)
        h1.df.loc[i, '겹침수량'] = str(count)
        if i == 0 or h1.df.loc[i, 'cycle'] == '0':
            h1.df.loc[i, '직전 사이클 소재 수량'] = 0
        elif i > 0:
            h1.df.loc[i, '직전 사이클 소재 수량'] = len(h1.df['소재 list'].loc[i-1])
    for i, row in h1.df.iterrows():
        if i in del_arr:
            h1.df.loc[i, '겹침여부'] = '겹침'
            # h1.df.loc[i, '겹침수량'] = str()
        else:
            h1.df.loc[i, '겹침여부'] = '안겹침'
            # h1.df.loc[i, '겹침수량'] = '겹침'
    h1.df = h1.df.reset_index(drop=True)


# extract/leaving first holding
def handle_first_hold(h1, work, s):
    del_arr = []
    df_temp = pd.DataFrame(columns=h1.df.columns)
    for j in work:
        cycle_num = None
        for i, row in h1.df.iterrows():
            if int(h1.df['가열로 번호'].loc[i]) == j and h1.df['Type'].loc[i] == 'hold' and cycle_num is None:
                cycle_num = int(h1.df['cycle'].loc[i])
            if int(h1.df['가열로 번호'].loc[i]) == j and int(h1.df['cycle'].loc[i]) != cycle_num and h1.df['Type'].loc[i] == 'heat':
                cycle_num = int(h1.df['cycle'].loc[i])
            if int(h1.df['가열로 번호'].loc[i]) == j and int(h1.df['cycle'].loc[i]) == cycle_num:
                if h1.df['Type'].loc[i] == 'heat' and h1.df['Type'].loc[i+1] == 'heat':
                    cycle_num += 1
                if h1.df['Type'].loc[i] == 'hold':
                    # print('work')
                    end = dt.datetime.strptime(h1.df['종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
                    start = dt.datetime.strptime(h1.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
                    dis = (end - start).total_seconds() / 60
                    # print(dis)
                    if dis > 30:
                        df_temp = df_temp.append(row)
                        df_temp = df_temp.reset_index(drop=True)
                        del_arr.append(i)
                        cycle_num += 1
                    else:
                        pass
    for i in del_arr:
        h1.df = h1.df.drop([i])
        # print(i, h1.df['Type'].loc[i])
    df_temp.to_csv(s, encoding='euc-kr')
    h1.df = h1.df.reset_index(drop=True)


# eliminate error loop
def eliminate_drop(h):
    for i, row in h.df.iterrows():
        if h.df.loc[i, 'drop_flag'] == 1:
            h.df = h.df.drop([i])
    h.df = h.df.reset_index(drop=True)


def determine_weekend(h):
    datee = ['월', '화', '수', '목', '금', '토', '일']
    for i, row in h.df.iterrows():
        if i > 0:
            d1 = dt.datetime.strptime(h.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d2 = dt.datetime.strptime(h.df['시작시간'].loc[i-1], "%Y-%m-%d %H:%M:%S")
            last_day = datee[d2.weekday()]
            to_day = datee[d1.weekday()]
            if to_day == '토' or to_day == '일':
                h.df.loc[i, '주말여부'] = '주말'
            elif to_day == '월' and (last_day != '토' and last_day != '일'):
                h.df.loc[i, '주말여부'] = '주말'
            else:
                h.df.loc[i, '주말여부'] = '평일'
        else:
            h.df.loc[i, '주말여부'] = '평일'


# make data for holding model
def model_hold(HT, df_mat, s):
    temp_dict = {'에너지': [], '시간': [], '장입소재개수': [], '장입중량총합': [], '사이클': [],
                 '장입최대중량': [], '평균온도': [], '가열로 번호': [], '시작시간': [], '종료시간': []}
    for i in range(len(HT.df.index)):
        flag = 0
        if HT.df['Type'].loc[i] == 'hold':
            d1 = dt.datetime.strptime(HT.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d2 = dt.datetime.strptime(HT.df['종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            temp_dict['시작시간'].append(d1)
            temp_dict['종료시간'].append(d2)
            temp_dict['사이클'].append(HT.df['cycle'].loc[i])
            t_mean = HT.df['평균온도'].loc[i]
            t = d2 - d1
            t = t.total_seconds()
            nn = HT.df['가열로 번호'].loc[i]
            if HT.df['소재 list'].loc[i][0] == '':
                num = 0
            else:
                num = len(HT.df['소재 list'].loc[i])
            list_m = []
            for k in HT.df['소재 list'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m.append(int(df_mat['투입중량'].loc[j]))
                        if int(df_mat['투입중량'].loc[j]) == 0:
                            flag = 1
                        break
            temp_dict['에너지'].append(HT.df['가스사용량'].loc[i])
            temp_dict['시간'].append(t)
            temp_dict['장입소재개수'].append(num)
            if flag == 1:
                temp_dict['장입중량총합'].append(0)
            else:
                temp_dict['장입중량총합'].append(np.sum(list_m))
            if len(list_m) == 0:
                temp_dict['장입최대중량'].append(0)
            elif len(list_m) > 0:
                temp_dict['장입최대중량'].append(np.max(list_m))
            temp_dict['평균온도'].append(t_mean)
            temp_dict['가열로 번호'].append(nn)
        print(i)
    df2 = pd.DataFrame.from_dict(temp_dict)
    df2.to_csv(s, encoding='euc-kr')


# make data for opening model
def model_open(HT, df_mat):
    df = pd.DataFrame(columns=['에너지', '시간', '나간소재개수', '나간중량총합', '나간최대중량',
                               '들어온소재개수', '들어온중량총합', '들어온최대중량',
                               '장입소재개수', '장입중량총합', '장입최대중량', '시작온도', '종료온도', '가열로 번호'])
    for i in range(len(HT.df.index)):
        flag = 0
        temp = []
        if HT.df['Type'].loc[i] == 'open':
            d1 = dt.datetime.strptime(HT.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d2 = dt.datetime.strptime(HT.df['종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            t_start = HT.df['시작온도'].loc[i]
            t_end = HT.df['종료온도'].loc[i]
            t = d2 - d1
            t = t.total_seconds()
            nn = HT.df['가열로 번호'].loc[i]
            if HT.df['소재 list'].loc[i][0] == '':
                num_original = 0
            else:
                num_original = len(HT.df['소재 list'].loc[i])
            if HT.df['out'].loc[i][0] == '':
                num_out = 0
            else:
                num_out = len(HT.df['out'].loc[i])
            if HT.df['in'].loc[i][0] == '':
                num_in = 0
            else:
                num_in = len(HT.df['in'].loc[i])
            list_m = []
            list_m_1 = []
            list_m_2 = []
            for k in HT.df['소재 list'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m.append(int(df_mat['투입중량'].loc[j]))
                        if int(df_mat['투입중량'].loc[j]) == 0:
                            flag = 1
                        break
            for k_1 in HT.df['out'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k_1.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m_1.append(int(df_mat['투입중량'].loc[j]))
                        break
            for k_2 in HT.df['in'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k_2.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m_2.append(int(df_mat['투입중량'].loc[j]))
                        break
            temp.append(HT.df['가스사용량'].loc[i])
            temp.append(t)
            temp.append(num_out)
            temp.append(np.sum(list_m_1))
            if len(list_m_1) == 0:
                temp.append(0)
            elif len(list_m_1) > 0:
                temp.append(np.max(list_m_1))
            temp.append(num_in)
            temp.append(np.sum(list_m_2))
            if len(list_m_2) == 0:
                temp.append(0)
            elif len(list_m_2) > 0:
                temp.append(np.max(list_m_2))
            temp.append(num_original)
            if flag == 1:
                temp.append(0)
            else:
                temp.append(np.sum(list_m))
            if len(list_m) == 0:
                temp.append(0)
            elif len(list_m) > 0:
                temp.append(np.max(list_m))
            temp.append(t_start)
            temp.append(t_end)
            temp.append(nn)
            df = df.append(pd.DataFrame(data=np.array([[temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6],
                                                        temp[7], temp[8], temp[9], temp[10], temp[11], temp[12], temp[13]]]),
                                        columns=df.columns))
            df = df.reset_index(drop=True)
        print(i)
    df.to_csv('./model/model_open.csv', encoding='euc-kr')


# make data for reheating model
def model_reheat(HT, df_mat, s):
    temp_dict = {'에너지': [], '시간': [], '시간(0제외)': [], '추가소재유무': [],
                 '들어온소재개수': [], '들어온중량총합': [], '들어온최대중량': [],
                 '장입소재개수': [], '장입중량총합': [], '장입최대중량': [],
                 '시작온도': [], '가열로 번호': [], '사이클': [], '시작시간': [], '종료시간': []}
    for i in range(len(HT.df.index)):
        flag = 0
        flag_in = 0
        if HT.df['Type'].loc[i] == 'reheat':
            d1 = dt.datetime.strptime(HT.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d2 = dt.datetime.strptime(HT.df['종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d0 = int(HT.df['뺄시간'].loc[i]) * 60
            temp_dict['시작시간'].append(d1)
            temp_dict['종료시간'].append(d2)
            temp_dict['사이클'].append(HT.df['cycle'].loc[i])
            t_start = HT.df['시작온도'].loc[i]
            t = d2 - d1
            temp_dict['에너지'].append(HT.df['가스사용량'].loc[i])
            temp_dict['시간'].append(t.total_seconds())
            temp_dict['시간(0제외)'].append(t.total_seconds() - d0)
            nn = HT.df['가열로 번호'].loc[i]
            if HT.df['소재 list'].loc[i][0] == '':
                num_original = 0
            else:
                num_original = len(HT.df['소재 list'].loc[i])
            if HT.df['in'].loc[i][0] == '':
                num_in = 0
            else:
                num_in = len(HT.df['in'].loc[i])
            if num_in > 0:
                flag_in = 1
            list_m = []
            list_m_1 = []
            for k in HT.df['소재 list'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m.append(int(df_mat['투입중량'].loc[j]))
                        if int(df_mat['투입중량'].loc[j]) == 0:
                            flag = 1
                        break
            for k_1 in HT.df['in'].loc[i]:
                for j in range(len(df_mat.index)):
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k_1.split('_')[0] == df_mat['수주번호'].loc[j]:
                        list_m_1.append(int(df_mat['투입중량'].loc[j]))
                        if int(df_mat['투입중량'].loc[j]) == 0:
                            flag = 1
                        break
            temp_dict['추가소재유무'].append(flag_in)
            temp_dict['들어온소재개수'].append(num_in)
            temp_dict['들어온중량총합'].append(np.sum(list_m_1))
            if len(list_m_1) == 0:
                temp_dict['들어온최대중량'].append(0)
            elif len(list_m_1) > 0:
                temp_dict['들어온최대중량'].append(np.max(list_m_1))
            temp_dict['장입소재개수'].append(num_original)
            if flag == 1:
                temp_dict['장입중량총합'].append(0)
            else:
                temp_dict['장입중량총합'].append(np.sum(list_m))
            if len(list_m) == 0:
                temp_dict['장입최대중량'].append(0)
            elif len(list_m) > 0:
                temp_dict['장입최대중량'].append(np.max(list_m))
            temp_dict['시작온도'].append(t_start)
            temp_dict['가열로 번호'].append(nn)
        print(i)
    df2 = pd.DataFrame.from_dict(temp_dict)
    df2.to_csv(s, encoding='euc-kr')


# make data for heating model
def model_heat_kang_ver_heat(HT, df_mat, df_mat_heat, s_list, ss_list, s):
    to_day = ['월', '화', '수', '목', '금', '토', '일']
    determine_weekend(HT)
    HT.df = HT.df.sort_values(by=['작업일자'], axis=0)
    HT.df = HT.df.reset_index(drop=True)
    print(HT.df)
    # list_mn = [0]*len(sn_list)
    temp_dict = {'에너지': [], '시간(총)': [], '시간(0제외)': [], '시작온도': [], '종료온도': [],
                 'A_num': [], 'A_sum': [], 'A_max': [],
                 'C_num': [], 'C_sum': [], 'C_max': [],
                 'S_num': [], 'S_sum': [], 'S_max': [],
                 'H_A_num': [], 'H_A_sum': [], 'H_A_max': [],
                 'H_C_num': [], 'H_C_sum': [], 'H_C_max': [],
                 'H_S_num': [], 'H_S_sum': [], 'H_S_max': [],
                 '장입소재개수': [], '장입중량총합': [], '장입최대중량': [],
                 '열괴장입소재개수': [], '열괴장입중량총합': [], '열괴장입최대중량': [],
                 '민감소재장입개수': [], '민감소재중량총합': [], '민감소재최대중량': [],
                 '비민감소재장입개수': [], '비민감소재중량총합': [], '비민감소재최대중량': [],
                 '쉰시간': [], '문열림횟수': [], 'drop': [], '가열로번호': [],
                 '작업일자': [], '주/야간': [], '가열시작시간': [], '요일': [], '겹침여부': [], '민감비고': [],
                 '직전 사이클 소재 수량': [], '겹치는 소재 수량': [], '강종종류': [], '주말여부': [], '에러발생': []}
    print(len(HT.df.index))
    for i in range(len(HT.df.index)):
        flag = 0
        flag_S = 0
        flag_E = 0
        if HT.df['Type'].loc[i] == 'heat':
            dd1 = dt.datetime.strptime(HT.df['작업일자'].loc[i], '%Y-%m-%d')
            d1 = dt.datetime.strptime(HT.df['시작시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d2 = dt.datetime.strptime(HT.df['종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d3 = dt.datetime.strptime(HT.df['이전 종료시간'].loc[i], "%Y-%m-%d %H:%M:%S")
            d0 = int(HT.df['뺄시간'].loc[i]) * 60
            t_start = HT.df['시작온도'].loc[i]
            t_end = HT.df['종료온도'].loc[i]
            temp_dict['겹치는 소재 수량'].append(HT.df['겹침수량'].loc[i])
            temp_dict['직전 사이클 소재 수량'].append(HT.df['직전 사이클 소재 수량'].loc[i])
            temp_dict['작업일자'].append(HT.df['작업일자'].loc[i])
            temp_dict['주/야간'].append(HT.df['주/야간'].loc[i])
            temp_dict['가열시작시간'].append(HT.df['시작시간'].loc[i])
            temp_dict['겹침여부'].append(HT.df['겹침여부'].loc[i])
            temp_dict['시작온도'].append(t_start)
            temp_dict['종료온도'].append(t_end)
            temp_dict['주말여부'].append(HT.df['주말여부'].loc[i])
            temp_dict['요일'].append(to_day[d1.weekday()])
            t = d2 - d1
            temp_dict['시간(총)'].append(t.total_seconds())
            t = t.total_seconds() - d0
            temp_dict['시간(0제외)'].append(t)
            t2 = d1 - d3
            t2 = t2.total_seconds()
            temp_dict['쉰시간'].append(t2)
            nn = HT.df['가열로 번호'].loc[i]
            temp_dict['가열로번호'].append(nn)
            temp_dict['문열림횟수'].append(int(HT.df['가열중 문열림 횟수'].loc[i]))
            list_A = []
            list_C = []
            list_S = []
            list_HA = []
            list_HC = []
            list_HS = []
            list_M = []
            list_SM = []
            list_SNM = []
            list_HM = []
            list_kang = ''
            print(HT.df['작업일자'].loc[i], '가열로 번호 : ', nn)
            print(HT.df['소재 list'].loc[i])
            for k in HT.df['소재 list'].loc[i]:
                # print(k.split('_')[0])
                temp_len_list = len(list_M)
                for j in range(len(df_mat.index)):
                    flag_H = 0
                    if HT.df['주/야간'].loc[i] == df_mat['주/야간'].loc[j] and \
                            HT.df['작업일자'].loc[i] == df_mat['작업일자'].loc[j] and k.split('_')[0] == df_mat['수주번호'].loc[j]:
                        for t in range(len(df_mat_heat)):
                            dd0 = dt.datetime.strptime(df_mat_heat['1'].loc[t], '%Y-%m-%d %H:%M')
                            dd0_1 = dt.datetime(year=2019, month=dd0.month, day=dd0.day)
                            dd1_1 = dt.datetime(year=2019, month=dd1.month, day=dd1.day)
                            d_gap = (dd1_1 - dd0_1).total_seconds()
                            if k.split('_')[0] == df_mat_heat['0'].loc[t] and 0 <= d_gap <= 172800:
                                print('in')
                                print(k.split('_')[0])
                                list_HM.append(int(df_mat['투입중량'].loc[j]))
                                if type(df_mat['강종'].loc[j]) == float:
                                    flag = 2
                                elif df_mat['강종'].loc[j] == 'ALLOY':
                                    list_HA.append(df_mat['투입중량'].loc[j])
                                    flag_H = 1
                                    break
                                elif df_mat['강종'].loc[j] == 'CARBON':
                                    list_HC.append(df_mat['투입중량'].loc[j])
                                    flag_H = 1
                                    break
                                elif df_mat['강종'].loc[j] == 'SUS' or 'SUS 304' or 'SUS 321':
                                    list_HS.append(df_mat['투입중량'].loc[j])
                                    flag_H = 1
                                    break
                        list_M.append(int(df_mat['투입중량'].loc[j]))
                        if df_mat['사내재질'].loc[j] in ss_list:
                            flag_S = 1
                        if pd.isna(df_mat['사내재질'].loc[j]):
                            flag_S = 2
                            # list_mn[sn_list.index(df_mat['사내재질'].loc[j])] += 1
                        if df_mat['사내재질'].loc[j] in s_list:
                            list_SM.append(int(df_mat['투입중량'].loc[j]))
                        else:
                            list_SNM.append(int(df_mat['투입중량'].loc[j]))
                        if flag_H == 0:
                            # list_M.append(int(df_mat['투입중량'].loc[j]))
                            if type(df_mat['강종'].loc[j]) == float:
                                flag = 2
                            elif df_mat['강종'].loc[j] == 'ALLOY':
                                list_A.append(df_mat['투입중량'].loc[j])
                            elif df_mat['강종'].loc[j] == 'CARBON':
                                list_C.append(df_mat['투입중량'].loc[j])
                            elif df_mat['강종'].loc[j] == 'SUS' or 'SUS 304' or 'SUS 321':
                                list_S.append(df_mat['투입중량'].loc[j])
                        if int(df_mat['투입중량'].loc[j]) == 0:
                            if flag == 2:
                                flag = 3
                            else:
                                flag = 1
                        break
                if temp_len_list == len(list_M):
                    flag_E = 1
            temp_dict['에너지'].append(HT.df['가스사용량'].loc[i])
            if HT.df['가스사용량'].loc[i] == 0:
                flag_E = 1
            if len(list_A) != 0 or len(list_HA) != 0:
                list_kang += 'A'
            if len(list_C) != 0 or len(list_HC) != 0:
                list_kang += 'C'
            if len(list_S) != 0 or len(list_HS) != 0:
                list_kang += 'S'
            temp_dict['강종종류'].append(list_kang)
            # print(HT.df['가스사용량(총)'].loc[i])
            temp_dict['A_num'].append(len(list_A))
            temp_dict['A_sum'].append(np.sum(list_A))
            if len(list_A) == 0:
                temp_dict['A_max'].append(0)
            else:
                temp_dict['A_max'].append(np.max(list_A))
            temp_dict['C_num'].append(len(list_C))
            temp_dict['C_sum'].append(np.sum(list_C))
            if len(list_C) == 0:
                temp_dict['C_max'].append(0)
            else:
                temp_dict['C_max'].append(np.max(list_C))
            temp_dict['S_num'].append(len(list_S))
            temp_dict['S_sum'].append(np.sum(list_S))
            if len(list_S) == 0:
                temp_dict['S_max'].append(0)
            else:
                temp_dict['S_max'].append(np.max(list_S))
            temp_dict['H_A_num'].append(len(list_HA))
            temp_dict['H_A_sum'].append(np.sum(list_HA))
            if len(list_HA) == 0:
                temp_dict['H_A_max'].append(0)
            else:
                temp_dict['H_A_max'].append(np.max(list_HA))
            temp_dict['H_C_num'].append(len(list_HC))
            temp_dict['H_C_sum'].append(np.sum(list_HC))
            if len(list_HC) == 0:
                temp_dict['H_C_max'].append(0)
            else:
                temp_dict['H_C_max'].append(np.max(list_HC))
            temp_dict['H_S_num'].append(len(list_HS))
            temp_dict['H_S_sum'].append(np.sum(list_HS))
            if len(list_HS) == 0:
                temp_dict['H_S_max'].append(0)
            else:
                temp_dict['H_S_max'].append(np.max(list_HS))
            temp_dict['장입소재개수'].append(len(list_M))
            temp_dict['장입중량총합'].append(np.sum(list_M))
            if len(list_M) == 0:
                temp_dict['장입최대중량'].append(0)
            else:
                temp_dict['장입최대중량'].append(np.max(list_M))
            temp_dict['열괴장입소재개수'].append(len(list_HM))
            temp_dict['열괴장입중량총합'].append(np.sum(list_HM))
            if len(list_HM) == 0:
                temp_dict['열괴장입최대중량'].append(0)
            else:
                temp_dict['열괴장입최대중량'].append(np.max(list_HM))
            temp_dict['민감소재장입개수'].append(len(list_SM))
            temp_dict['민감소재중량총합'].append(np.sum(list_SM))
            if len(list_SM) == 0:
                temp_dict['민감소재최대중량'].append(0)
            else:
                temp_dict['민감소재최대중량'].append(np.max(list_SM))
            temp_dict['비민감소재장입개수'].append(len(list_SNM))
            temp_dict['비민감소재중량총합'].append(np.sum(list_SNM))
            if len(list_SNM) == 0:
                temp_dict['비민감소재최대중량'].append(0)
            else:
                temp_dict['비민감소재최대중량'].append(np.max(list_SNM))
            if flag == 0:
                temp_dict['drop'].append('누락없음')
            elif flag == 1:
                temp_dict['drop'].append('중량누락')
            elif flag == 2:
                temp_dict['drop'].append('강종누락')
            elif flag == 3:
                temp_dict['drop'].append('전체누락')
            if flag_S == 1:
                temp_dict['민감비고'].append('초민감')
            elif flag_S == 2:
                temp_dict['민감비고'].append('민감누락')
            else:
                temp_dict['민감비고'].append('이상없음')
            if flag_E == 1:
                temp_dict['에러발생'].append(1)
            else:
                temp_dict['에러발생'].append(0)
        print('index :', i)
    df2 = pd.DataFrame.from_dict(temp_dict)
    df2.to_csv(s, encoding='euc-kr')


# clustering conditions
def clustering_condition(s, heat_flag, flag_m, time_flag, sense_flag):
    if s == '민감만/1h제외':
        if heat_flag == 0 and flag_m == 0 and time_flag == 0 and sense_flag == 1:
            return True
        else:
            return False
    elif s == '민감만/1h포함':
        if heat_flag == 0 and flag_m == 0 and sense_flag == 1:
            return True
        else:
            return False
    elif s == '민감제외/1h제외':
        if heat_flag == 0 and flag_m == 0 and time_flag == 0 and sense_flag == 0:
            return True
        else:
            return False
    elif s == '민감제외/1h포함':
        if heat_flag == 0 and flag_m == 0 and sense_flag == 0:
            return True
        else:
            return False
    elif s == '전부/1h제외':
        if heat_flag == 0 and flag_m == 0 and time_flag == 0:
            return True
        else:
            return False
    elif s == '전부/1h포함':
        if heat_flag == 0 and flag_m == 0:
            return True
        else:
            return False
    else:
        raise NameError("Wrong input")
