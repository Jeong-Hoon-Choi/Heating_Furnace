from bases import *
from constant.constant_data_make import *


# 소재 -시간 매칭
def add_time_to_material(s1, s2, s3):
    df_material_temp = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    # "./material_new/material_time_re2.csv"
    df_material_time = pd.read_csv(s2, encoding='euc-kr', index_col=0)
    df_match = pd.DataFrame()
    match_dict = {}
    for i in range(len(df_material_temp.index)):
        if pd.isna(df_material_temp['장입가열로'].loc[i]):
            pass
        else:
            work_date = df_material_temp['작업일자'].loc[i]
            d_n = df_material_temp['주/야간'].loc[i]
            hogi = df_material_temp['장입가열로'].loc[i]
            if work_date + d_n + hogi not in match_dict:
                match_dict[work_date + d_n + hogi] = list()
            match_dict[work_date + d_n + hogi].append(i)
            # print("work")
    print(len(match_dict))
    for j, row2 in df_material_time.iterrows():
        dict_ = {}
        work_date = df_material_time['작업일자'].loc[j]
        d_n = df_material_time['주/야간'].loc[j]
        hogi = df_material_time['가열로명'].loc[j]
        if work_date + d_n + hogi in match_dict:
            for index_ in match_dict[work_date + d_n + hogi]:
                for col_name in df_material_temp.columns:
                    dict_[col_name] = [df_material_temp.loc[index_, col_name]]
                dict_['시작시간'] = [df_material_time.loc[j, '가열시작일시']]
                df_temp = pd.DataFrame.from_dict(dict_)
                df_match = pd.concat([df_match, df_temp])
                df_match = df_match.reset_index(drop=True)
    print("done")
    df_match.to_csv(s3, mode='w', encoding='euc-kr')


# 소재/가열로 파싱
def parsing_order_number(s):
    s_2 = s.split('/')[3]
    s_1 = s_2.split('.')[0]
    df_1 = pd.read_csv(s)
    df = pre_parsing(df_1)
    df2 = pd.DataFrame(columns=df.columns)
    e = len(df.index)
    for k, row in df.iterrows():
        ss = df['수주번호'].loc[k]
        s5 = []
        if type(ss) != float:
            if len(ss.split('\n')) > 1:
                s5.append(ss.split('\n')[0])
            elif len(ss) > 16 and ss[16] == '#':
                print(ss, ss[0:16])
                s5.append(ss[0:16])
            elif len(ss) > 16 and ss[16:18] == ' #':
                print(ss, ss[0:16])
                s5.append(ss[0:16])
            elif len(ss) > 16:
                flag = 0
                s1 = ss.split('-')
                try:
                    int(s1[0])
                except:
                    flag = 2
                try:
                    int(s1[1])
                except:
                    flag = 1
                if flag == 0:
                    s2 = s1[2].split(',')
                    s3 = s1[2].split('~')
                    s4 = []
                    s5 = []
                    if len(s2) >= 2 and len(s3) == 2:
                        s3 = s2[len(s2) - 1].strip().split('~')
                        for i in range(len(s2) - 1):
                            s5.append(s1[0] + '-' + s1[1] + '-' + s2[i].strip())
                    else:
                        pass
                    if len(s3) == 1:
                        if s3[0] == s1[2] and len(s2) < 2:
                            s5.append(ss.split()[0])
                        else:
                            for i in range(len(s2)):
                                if len(s2[i].strip()) == 1:
                                    s4.append('00' + s2[i])
                                elif len(s2[i].strip()) == 2:
                                    s4.append('0' + s2[i])
                                else:
                                    s4.append(s2[i])
                            for i in range(len(s4)):
                                s5.append(s1[0] + '-' + s1[1] + '-' + s4[i].strip())
                    else:
                        start = int(s3[0].strip())
                        end = int(s3[1].strip())
                        for j in range(start, end + 1):
                            if len(str(j)) == 1:
                                s4.append('00' + str(j))
                            elif len(str(j)) == 2:
                                s4.append('0' + str(j))
                            else:
                                s4.append(str(j))
                        for i in range(len(s4)):
                            s5.append(s1[0] + '-' + s1[1] + '-' + s4[i].strip())
                    print(ss, s5)
                elif flag == 1:
                    s5.clear()
                    s5.append(ss.split()[0])
                    print(ss, s5)
                elif flag == 2:
                    pass
        if len(s5) > 0:
            for z in range(len(s5)):
                df2 = df2.append(row)
                df2['수주번호'].iloc[len(df2.index)-1] = s5[z]
            df = df.drop([k])
    print(e, len(df.index))
    df = df.append(df2)
    df = df.reset_index(drop=[True])
    print(df)
    print(df2)
    df = df.sort_values(["작업일자"], ascending=[True])
    df = df.reset_index(drop=[True])
    df.to_csv(base_path + 'raw/' + s_1 + '_par.csv', encoding='euc-kr')


# 프레스기 파싱
def pre_parsing(df_press_3):
    for i in range(len(df_press_3.index)):
        if type(df_press_3['수주번호'].loc[int(i)]) != float and len(df_press_3['수주번호'].loc[int(i)]) > 16:
            if df_press_3['수주번호'].loc[int(i)][0] == 'A':
                df_press_3['수주번호'].loc[int(i)] =\
                    df_press_3['수주번호'].loc[int(i)][2:]
            elif df_press_3['수주번호'].loc[int(i)][12] == 'F':
                df_press_3['수주번호'].loc[int(i)] =\
                    df_press_3['수주번호'].loc[int(i)][0:12] + df_press_3['수주번호'].loc[int(i)][13:]
    return df_press_3


# 시간 체크
def wrong_st_ed(s, work=p_all2):
    s_2 = s.split('/')[3]
    s_1 = s_2.split('.')[0]
    df = pd.read_csv(s)
    df = df.fillna(0)
    df_temp = pd.DataFrame()
    del_arr = []
    count0 = 0
    for num in work:
        del_arr2 = []
        df_new = pd.DataFrame()
        count = 0
        for i, row in df.iterrows():
            if df['가열로명'].loc[i] == '가열로' + str(num) + '호기':
                df_new = df_new.append(row)
                df_new = df_new.reset_index(drop=True)
        print(len(df_new))
        count0 += len(df_new)
        if len(df_new.index) != 0:
            df_new = df_new.sort_values(['가열시작일시'], ascending=[True])
            df_new = df_new.reset_index(drop=True)
            for j, row in df_new.iterrows():
                if j > 0:
                    start = dt.datetime.strptime(df_new['가열시작일시'].loc[j], "%Y-%m-%d %H:%M")
                    end2 = dt.datetime.strptime(df_new['가열종료일시'].loc[j], "%Y-%m-%d %H:%M")
                    end = dt.datetime.strptime(df_new['가열종료일시'].loc[j - 1], "%Y-%m-%d %H:%M")
                    if (start - end).total_seconds() < 0:
                        # print(int(df_new['순번'].loc[j]), int(df_new['순번'].loc[j-1]))
                        print(start, end, start - end, j)
                        del_arr.append(int(df_new['순번'].loc[j - 1]))
                        del_arr.append(int(df_new['순번'].loc[j]))
                        del_arr2.append(j - 1)
                        del_arr2.append(j)
                        count += 2
                    if (end2 - start).total_seconds() < 0:
                        del_arr.append(int(df_new['순번'].loc[j]))
                        del_arr2.append(j)
                        count += 0.01
            print('가열로 ' + str(num) + '호기 : ', count)
            print(del_arr2)
            del_arr2 = set(del_arr2)
            for t in del_arr2:
                df_new = df_new.drop([t])
            df_temp = pd.concat([df_temp, df_new])
            df_temp = df_temp.reset_index(drop=True)
    df_temp.to_csv(base_path + 'raw/' + s_1 + '_re1.csv', encoding='euc-kr')


# 민감소재 매칭
def match_S(s_list, df):
    # df = pd.read_csv(s, encoding='euc-kr')
    count = 0
    list1 = []
    list0 = []
    for i, row in df.iterrows():
        if type(df.loc[i, '사내재질']) != float:
            list0.append(df.loc[i, '사내재질'])
        # print(str(df.loc[i, '사내재질']).upper(), str(df.loc[i, '사내재질']).upper()==)
        if str(df.loc[i, '사내재질']).upper() in s_list:
            # print(i, df.loc[i, '사내재질'])
            list1.append(df.loc[i, '사내재질'])
            count += 1
    list2 = set(list1)
    list3 = set(list0)
    '''
    df2 = pd.DataFrame(data=list3, columns=['사내재질'])
    print(count)
    print(df.loc[0, '프래스명'])
    print(df2)
    df2.to_csv(str(df.loc[0, '프래스명']) + '_사내재질_list.csv', encoding='euc-kr')
    '''
    return list3


# first you need to change format .xlsx to .csv

# file B -> B'
# parsing order number in material file
file_path_material = base_path + 'raw/material.csv'
parsing_order_number(file_path_material)

# file C -> C'
# Check the wrong data.
file_path_time = base_path + 'raw/start_end.csv'
wrong_st_ed(file_path_time, work=p_all2)

# file D -> D'
# parsing order number in press data
file_path_press = base_path + 'raw/press.csv'
parsing_order_number(file_path_press)

# file M
# needed B', C'
# matching B', C'
file_path_materials = base_path + 'raw/material_par.csv'
file_path_times = base_path + 'raw/start_end_re1.csv'
file_path_output = base_path + 'raw/material_par3.csv'
add_time_to_material(file_path_materials, file_path_times, file_path_output)