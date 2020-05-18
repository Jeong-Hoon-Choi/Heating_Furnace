from bases import *


def drop_drop(s, s2):
    df = pd.read_csv(s)
    for i, row in df.iterrows():
        # print(df['SO_DT'].loc[i][0:2], df['SO_DT'].loc[i])
        if df['SO_DT'].loc[i][0:2] != '19':
            df = df.drop([i])
    df = df.reset_index(drop=True)
    df.to_csv(s2, encoding='euc=kr')


def match_volume(s, s2, s3):
    df = pd.read_csv(s, encoding='euc-kr')
    df2 = pd.read_csv(s2, encoding='euc-kr')
    df['LEN1'] = 0
    df['LEN2'] = 0
    df['LEN3'] = 0
    df['SHAPE'] = ''
    for i, row in df.iterrows():
        ss = str(df['수주번호'].loc[i])
        for j, row2 in df2.iterrows():
            if len(str(df2['SO_SEQ'].loc[j])) == 1:
                ss2 = str(df2['SO_NO'].loc[j]) + '-' + '00' + str(df2['SO_SEQ'].loc[j])
            elif len(str(df2['SO_SEQ'].loc[j])) == 2:
                ss2 = str(df2['SO_NO'].loc[j]) + '-' + '0' + str(df2['SO_SEQ'].loc[j])
            else:
                ss2 = str(df2['SO_NO'].loc[j]) + '-' + str(df2['SO_SEQ'].loc[j])
            print(ss, ss2)
            if ss == ss2:
                df['LEN1'].loc[i] = df2['LEN_1'].loc[j]
                df['LEN2'].loc[i] = df2['LEN_2'].loc[j]
                df['LEN3'].loc[i] = df2['LEN_3'].loc[j]
                df['SHAPE'].loc[i] = df2['SHAPE_GU_NM'].loc[j]
                break
        print(i)
    df.to_csv(s3, encoding='euc-kr')


def compare_time(s1, s2, s3):
    df1 = pd.read_csv(s1, index_col=0)
    df2 = pd.read_csv(s2, index_col=0)
    df3 = pd.DataFrame()
    for i, row in df1.iterrows():
        t1 = dt.datetime.strptime(df1['가열시작일시'].loc[i], "%Y-%m-%d %H:%M")
        flag = 0
        for j, row2 in df2.iterrows():
            t2 = dt.datetime.strptime(df2['가열시작일시'].loc[j], "%Y-%m-%d %H:%M")
            if t1 == t2:
                flag = 1
                break
        if flag == 0:
            df3 = df3.append(row)
    df3 = df3.reset_index(drop=True)
    df3.to_csv(s3, encoding='euc-kr')


def compare_mat(s1, s2, s3):
    df1 = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc-kr', index_col=0)
    df3 = pd.DataFrame(columns=df1.columns)
    for i, row in df1.iterrows():
        flag = 0
        a0 = df1['수주번호'].loc[i]
        b0 = df1['작업일자'].loc[i]
        c0 = df1['주/야간'].loc[i]
        for j, row2 in df2.iterrows():
            a1 = df2['수주번호'].loc[j]
            b1 = df2['작업일자'].loc[j]
            c1 = df2['주/야간'].loc[j]
            if a0 == a1 and b0 == b1 and c0 == c1:
                flag = 1
                break
        if flag == 0:
            df3 = df3.append(row)
        print(i)
    df3 = df3.reset_index()
    df3.to_csv(s3, encoding='euc-kr')


def data_proc(s1, s2, s3, s4):
    df1 = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc-kr')
    df3 = pd.read_csv(s3, encoding='euc-kr', index_col=0)
    df4 = pd.DataFrame()
    temp = []
    for i, row in df1.iterrows():
        if int(df1['가열로번호'].loc[i]) == 5:
            if df1['drop'].loc[i] != '누락없음' or int(df1['장입소재개수'].loc[i]) == 0:
                temp.append(i)
            for j, row2 in df2.iterrows():
                if round(df1['에너지'].loc[i], 1) == df2['에너지'].loc[j] and df1['시간'].loc[i] == df2['시간'].loc[j] and \
                        df1['A_sum'].loc[i] == df2['A_sum'].loc[j] and df1['C_sum'].loc[i] == df2['C_sum'].loc[j]:
                    temp.append(i)
    print(temp, len(temp))
    for t, row in df3.iterrows():
        if t not in temp:
            df4 = df4.append(row)
            df4 = df4.reset_index(drop=True)
    df4.to_csv(s4, encoding='euc=kr')


def kang_diference(s1, s2, s3):
    df1 = pd.read_csv(s1, encoding='euc=kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc=kr', index_col=0)
    df3 = pd.DataFrame()
    for i, row in df1.iterrows():
        c1 = ''
        if type(df1['강종'].loc[i]) != float:
            if df1['강종'].loc[i] == 'ALLOY':
                c1 = 'A'
            elif df1['강종'].loc[i] == 'C-STEEL':
                c1 = 'C'
            elif df1['강종'].loc[i] == 'SUS' or df1['강종'].loc[i] == 'STAINLESS' or df1['강종'].loc[i] == 'DUPLEX':
                c1 = 'S'
        for j, row2 in df2.iterrows():
            c2 = ''
            if type(df2['강종'].loc[j]) != float:
                if df2['강종'].loc[j] == 'ALLOY':
                    c2 = 'A'
                elif df2['강종'].loc[j] == 'CARBON':
                    c2 = 'C'
                elif df2['강종'].loc[j] == 'SUS' or df2['강종'].loc[j] == 'SUS 304' or df2['강종'].loc[j] == 'SUS 321':
                    c2 = 'S'
            if df1['수주번호'].loc[i] == df2['수주번호'].loc[j] and df1['작업일자'].loc[i] == df2['작업일자'].loc[j] \
                    and df1['주/야간'].loc[i] == df2['주/야간'].loc[j]:
                if c1 == c2:
                    break
                else:
                    df3 = df3.append(row)
                    df3 = df3.reset_index(drop=True)
                    break
        print(i)
    df3.to_csv(s3, encoding='euc-kr')


def export_kang(s1, s2, s3, s4):
    df1 = pd.read_csv(s1, encoding='euc=kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc=kr', index_col=0)
    kang_arr1 = []
    for i, row in df1.iterrows():
        c1 = ''
        if type(df1['강종'].loc[i]) != float:
            if df1['강종'].loc[i] == 'ALLOY':
                c1 = 'A'
            elif df1['강종'].loc[i] == 'C-STEEL':
                c1 = 'C'
            elif df1['강종'].loc[i] == 'SUS' or df1['강종'].loc[i] == 'STAINLESS' or df1['강종'].loc[i] == 'DUPLEX':
                c1 = 'S'
            temp = [df1['사내재질'].loc[i], c1]
            if temp in kang_arr1:
                pass
            else:
                kang_arr1.append(temp)
    kang_arr2 = []
    for i, row in df2.iterrows():
        c2 = ''
        if type(df2['강종'].loc[i]) != float:
            if df2['강종'].loc[i] == 'ALLOY':
                c2 = 'A'
            elif df2['강종'].loc[i] == 'CARBON':
                c2 = 'C'
            elif df2['강종'].loc[i] == 'SUS' or df2['강종'].loc[i] == 'SUS 304' or df2['강종'].loc[i] == 'SUS 321':
                c2 = 'S'
            temp = [df2['사내재질'].loc[i], c2]
            if temp in kang_arr2:
                pass
            else:
                kang_arr2.append(temp)
    kang_arr1.sort()
    kang_arr2.sort()
    df3 = pd.DataFrame(data=kang_arr1, columns=['소재', '강종'])
    df4 = pd.DataFrame(data=kang_arr2, columns=['소재', '강종'])
    df3.to_csv(s3, encoding='euc=kr')
    df4.to_csv(s4, encoding='euc=kr')


def diff_kang(s1, s2, s3):
    df1 = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc-kr', index_col=0)
    df3 = pd.DataFrame(columns=['사내재질', '강종-태웅표', '강종-erp'])
    for i, row in df1.iterrows():
        s1 = df1['소재'].loc[i]
        k1 = df1['강종'].loc[i]
        for j, row2 in df2.iterrows():
            s2 = df2['소재'].loc[j]
            k2 = df2['강종'].loc[j]
            if s1 == s2 and k1 != k2:
                df3.loc[len(df3.index)] = ''
                df3['사내재질'].loc[len(df3.index) - 1] = s1
                df3['강종-태웅표'].loc[len(df3.index) - 1] = k1
                df3['강종-erp'].loc[len(df3.index) - 1] = k2
                break
    df3 = df3.reset_index(drop=True)
    df3.to_csv(s3, encoding='euc-kr')


def export_kang2(s2, s4):
    df2 = pd.read_csv(s2, encoding='euc=kr', index_col=0)
    kang_arr2 = []
    for i, row in df2.iterrows():
        if type(df2['강종'].loc[i]) == float:
            temp = [df2['사내재질'].loc[i]]
            if temp in kang_arr2:
                pass
            else:
                kang_arr2.append(temp)
    kang_arr2.sort()
    df4 = pd.DataFrame(data=kang_arr2, columns=['사내재질'])
    df4.to_csv(s4, encoding='euc=kr')


def excel_test(s, s2, df):
    ex1 = load_workbook(s, data_only=True)['생산계획']
    index = 9
    # print(ex1.max_row, type(ex1.max_row))
    for i in range(index, ex1.max_row, 2):
        # print(i, len((ex1['AR' + str(i)].value).split('\n')), (ex1['AR' + str(i)].value).split('\n'))
        # print(i, ex1['AR' + str(i)].value)
        second_index = 2
        if ex1['AR' + str(i)].value is None or ex1['AR' + str(i)].value == '-' or type(ex1['AR' + str(i)].value) != str:
            pass
        else:
            # print()
            # print(i)
            # print(ex1['AR' + str(i)].value)
            # print('in')
            arr1 = (ex1['AR' + str(i)].value).split('\n')
            arr2 = []
            flag = 0
            while flag == 0:
                second_index += 2
                if ex1['AR' + str(i + second_index)].value is not None:
                    flag = 1
                if (i + second_index) == ex1.max_row:
                    flag = 1
                # print('second index : ', second_index)
                # print(ex1['AR' + str(i + second_index)].value)
            # print('second index : ', second_index)
            if len(arr1) > 1:
                # print(arr1)
                for t in range(len(arr1)):
                    if len(arr1[t].split(' ')) > 1:
                        arr1[t] = arr1[t].split(' ')[0]
                for i2 in range(1, second_index, 2):
                    if ex1['AS' + str(i + i2)].value is not None:
                        excel_test2(ex1, i, i2, arr2)
            elif len((ex1['AR' + str(i)].value).split('\n')) == 1:
                # print(arr1)
                if arr1[0] == '냉괴이송':
                    pass
                else:
                    for i2 in range(1, second_index, 2):
                        if ex1['AS' + str(i + i2)].value is not None:
                            excel_test2(ex1, i, i2, arr2)
            if len(arr2) != 0:
                # pass
                # print(i)
                # print(arr1)
                # print(arr2)
                for t in arr2:
                    # print(t)
                    df = df.append([[t[0], t[1], t[2], s2]])
                    df = df.reset_index(drop=True)
    return df


def excel_test2(ex1, i, i2, arr2):
    index_front = 0
    index_back = len(ex1['AS' + str(i + i2)].value)
    index_front += ex1['AS' + str(i + i2)].value.count('(')
    index_back -= ex1['AS' + str(i + i2)].value.count(')')
    if len(ex1['AS' + str(i + i2)].value[index_front:index_back].split(',')) > 1:
        # print('i값 : ' + str(i) + ' / i2값 : ' + str(i2) + ' / i + i2 :' + str(i + i2))
        # print('원본 : ' + ex1['AS' + str(i + i2)].value[index_front:index_back])
        for t in ex1['AS' + str(i + i2)].value[index_front:index_back].split(', '):
            if len(t.split(',\n')) > 1:
                for k in t.split(',\n'):
                    # print(k)
                    arr2.append([k.strip(), ex1['AP' + str(i + i2 - 1)].value,
                                 [ex1['AT' + str(i)].value, ex1['AT' + str(i + i2 - 1)].value]])
            elif len(t.split('\n')) > 1:
                for k in t.split('\n'):
                    if len(k) < 1:
                        pass
                    else:
                        # print(k)
                        arr2.append([k.strip(), ex1['AP' + str(i + i2 - 1)].value,
                                     [ex1['AT' + str(i)].value, ex1['AT' + str(i + i2 - 1)].value]])
            elif len(t.split(' \n')) > 1:
                for k in t.split(' \n'):
                    if len(k) < 1:
                        pass
                    else:
                        # print(k)
                        arr2.append([k.strip(), ex1['AP' + str(i + i2 - 1)].value,
                                     [ex1['AT' + str(i)].value, ex1['AT' + str(i + i2 - 1)].value]])
            else:
                # print(t)
                arr2.append([t.strip(), ex1['AP' + str(i + i2 - 1)].value,
                             [ex1['AT' + str(i)].value, ex1['AT' + str(i + i2 - 1)].value]])
    else:
        # print('i값 : ' + str(i) + ' / i2값 : ' + str(i2) + ' / i + i2 :' + str(i + i2))
        # print('원본 : ' + ex1['AS' + str(i + i2)].value[index_front:index_back])
        # print(ex1['AS' + str(i + i2)].value[index_front:index_back])
        arr2.append([ex1['AS' + str(i + i2)].value[index_front:index_back], ex1['AP' + str(i + i2 - 1)].value,
                     [ex1['AT' + str(i)].value, ex1['AT' + str(i + i2 - 1)].value]])


def matching_heat(s1, s2, s3):
    df1 = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc-kr', index_col=0)
    del_arr = []
    df_new = pd.DataFrame(columns=df2.columns)
    for i, row in df1.iterrows():
        ss1 = df1['수주번호'].loc[i]
        d1 = dt.datetime.strptime(df1['작업일자'].loc[i], '%Y-%m-%d')
        for i2, row2 in df2.iterrows():
            d2 = dt.datetime.strptime(df2['1'].loc[i2], '%Y-%m-%d %H:%M')
            dd0_1 = dt.datetime(year=2019, month=d1.month, day=d1.day)
            dd1_1 = dt.datetime(year=2019, month=d2.month, day=d2.day)
            d_gap = (dd0_1 - dd1_1).total_seconds()
            ss2 = df2['0'].loc[i2]
            if ss1 == ss2 and 0 <= d_gap <= 172800:
                # (d2.month < d1.month or (d2.month == d1.month and d2.day < d1.day)):
                df_new = df_new.append(row2)
                df_new = df_new.reset_index(drop=True)
                break
        print(i)
    df_new.to_csv(s3, encoding='euc-kr')


def matching_heat2(s1, s2, s3, s4):
    df1 = pd.read_csv(s1, encoding='euc-kr', index_col=0)
    df2 = pd.read_csv(s2, encoding='euc-kr', index_col=0)
    del_arr = []
    df_new = pd.DataFrame(columns=df2.columns)
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    # print(df1)
    # print(df2)
    for i2, row2 in df2.iterrows():
        d2 = dt.datetime.strptime(df2['1'].loc[i2], '%Y-%m-%d %H:%M')
        ss2 = df2['0'].loc[i2]
        flag_heat = 0
        for i, row in df1.iterrows():
            ss1 = df1['수주번호'].loc[i]
            d1 = dt.datetime.strptime(df1['작업일자'].loc[i], '%Y-%m-%d')
            dd0_1 = dt.datetime(year=2019, month=d1.month, day=d1.day)
            dd1_1 = dt.datetime(year=2019, month=d2.month, day=d2.day)
            d_gap = (dd0_1 - dd1_1).total_seconds()
            if ss1 == ss2 and 0 <= d_gap <= 172800:
                df_new = df_new.append(row2)
                df_new = df_new.reset_index(drop=True)
                df_new.loc[len(df_new.index) - 1, 'p'] = df1.loc[i, '프래스명']
                flag_heat = 1
                # print(df_new)
                break
        if flag_heat == 1:
            del_arr.append(i2)
        print(i2)
    for k in del_arr:
        df2 = df2.drop([k])
    df2 = df2.reset_index(drop=True)
    df_new.to_csv(s3, encoding='euc-kr')
    df2.to_csv(s4, encoding='euc-kr')
