from bases import *


# 프레스기 매칭2
def matching_ph_part2(h1, s, s_1, d, flag):
    print(h1.df['가열로 번호'].loc[0], h1.index_h, h1.index_h_next, h1.index)
    print((h1.df['소재 list'].loc[h1.index_h]))
    if h1.index == len(h1.df.index)-1:
        h1.index = h1.index_h
        return 2
    if (s in h1.df['소재 list'].loc[h1.index_h] or s + s_1 in h1.df['소재 list'].loc[h1.index_h]) and \
            d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index_h], "%Y-%m-%d %H:%M:%S") > dt.timedelta(minutes=0):
        if s + s_1 in h1.df['소재 list'].loc[h1.index_h]:
            print("re_in")
            print(h1.index, h1.df['소재 list'].loc[h1.index], h1.temp_list)
            if h1.df['Type'].loc[h1.index] != 'open':
                h1.index += 1
                return 0
            elif h1.df['Type'].loc[h1.index] == 'open' and flag == 0:
                if type(h1.df['소재 list'].loc[h1.index]) != float:
                    h1.temp_list = h1.df['소재 list'].loc[h1.index][:]
                elif type(h1.df['소재 list'].loc[h1.index]) == float:
                    pass
                if d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S") > dt.timedelta(
                        minutes=0):
                    if h1.index >= h1.index_h_next:
                        h1.index = h1.index_h
                        return 2
                    else:
                        if h1.df['Type'].loc[h1.temp_2] == 'open':
                            h1.temp_1 = h1.temp_2
                            h1.temp_2 = h1.index
                        elif h1.df['Type'].loc[h1.temp_2] == 'heat':
                            h1.temp_1 = h1.index
                            h1.temp_2 = h1.index
                        if h1.index < len(h1.df.index) - 1:
                            h1.index += 1
                            return 0
                        else:
                            h1.index = h1.index_h
                            return 2
                elif d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S") <= dt.timedelta(
                        minutes=0):
                    if h1.temp_2 == h1.index_h:
                        h1.index = h1.index_h
                        return 2
                    h1.index = h1.temp_1
                    h1.temp_time = d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S")
                    # print("Done")
                    return 4
            elif h1.df['Type'].loc[h1.index] == 'open' and flag == 1:
                if s in h1.df['out'].loc[h1.index] or s + s_1 in h1.df['out'].loc[h1.index]:
                    print("error!", h1.temp_2, h1.temp_1, h1.index, h1.df['가열로 번호'].loc[h1.index])
                    h1.df['in'].loc[h1.index] = ['wrong']
                    h1.temp_time = dt.timedelta()
                    return 3
                if (h1.df['in'].loc[h1.index]) == '[]':
                    h1.df['in'].loc[h1.index] = [s]
                elif (h1.df['in'].loc[h1.index]) != '[]':
                    h1.df['in'].loc[h1.index].append(s + s_1)
                if (h1.df['out'].loc[h1.temp_2]) == '[]':
                    h1.df['out'].loc[h1.temp_2] = [s + s_1]
                elif (h1.df['out'].loc[h1.temp_2]) != '[]':
                    h1.df['out'].loc[h1.temp_2].append(s + s_1)
                if type(h1.df['소재 list'].loc[h1.index]) == float:
                    h1.df['소재 list'].loc[h1.index] = h1.temp_list[:]
                    (h1.df['소재 list'].loc[h1.index]).append(s + s_1)
                elif type(h1.df['소재 list'].loc[h1.index]) != float:
                    (h1.df['소재 list'].loc[h1.index]).append(s + s_1)
                if type(h1.df['소재 list'].loc[h1.temp_2]) == float:
                    h1.df['소재 list'].loc[h1.temp_2] = h1.temp_list[:]
                h1.temp_1 = h1.index_h
                h1.temp_2 = h1.index_h
                h1.temp_time = dt.timedelta()
                return 3
        else:
            print("first_in")
            print(h1.index, h1.df['소재 list'].loc[h1.index], flag, h1.temp_list, type(h1.temp_list))
            if h1.index != h1.index_h and h1.df['Type'].loc[h1.index] == 'heat':
                if h1.df['Type'].loc[h1.index - 1] == 'hold' and \
                    (dt.datetime.strptime(h1.df['종료시간'].loc[h1.index - 1], "%Y-%m-%d %H:%M:%S") - d).total_seconds() < 0:
                    h1.index = h1.index_h
                    return 2
                else:
                    h1.index += 1
                    return 0
            elif h1.df['Type'].loc[h1.index] != 'open':
                h1.index += 1
                return 0
            elif h1.df['Type'].loc[h1.index] == 'open' and flag == 0:
                if type(h1.df['소재 list'].loc[h1.index]) != float:
                    h1.temp_list = h1.df['소재 list'].loc[h1.index][:]
                elif type(h1.df['소재 list'].loc[h1.index]) == float:
                    pass
                if d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S") > dt.timedelta(minutes=0):
                    if h1.index >= h1.index_h_next:
                        h1.index = h1.index_h
                        return 2
                    else:
                        h1.temp_2 = h1.index
                        if h1.index < len(h1.df.index)-1:
                            h1.index += 1
                            return 0
                        else:
                            h1.index = h1.index_h
                            return 2
                elif d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S") <= dt.timedelta(minutes=0):
                    if h1.temp_2 == h1.index_h:
                        h1.index = h1.index_h
                        return 2
                    h1.index = h1.temp_2
                    # print("Done", h1.index)
                    h1.temp_time = d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index], "%Y-%m-%d %H:%M:%S")
                    return 4
            elif h1.df['Type'].loc[h1.index] == 'open' and flag == 1:
                # h1.temp_out.clear()
                if (h1.df['out'].loc[h1.index]) == '[]':
                    # h1.temp_out.append(s)
                    # print(s, 'out', h1.index)
                    h1.df['out'].loc[h1.index] = [s]
                    # h1.temp_out.clear()
                elif (h1.df['out'].loc[h1.index]) != '[]':
                    # print(h1.df['out'].loc[h1.index], type(h1.df['out'].loc[h1.index]))
                    h1.df['out'].loc[h1.index].append(s)
                h1.df['소재 list'].loc[h1.index] = h1.temp_list[:]
                (h1.df['소재 list'].loc[h1.index]).remove(h1.df['소재 list'].loc[h1.index][h1.df['소재 list'].loc[h1.index].index(s)])
                (h1.df['소재 list'].loc[h1.index_h]).remove(h1.df['소재 list'].loc[h1.index_h][h1.df['소재 list'].loc[h1.index_h].index(s)])
                (h1.df['소재 list'].loc[h1.index_h]).append(s + s_1)
                h1.temp_2 = h1.index_h
                h1.temp_time = dt.timedelta()
                return 3
    elif not (s in h1.df['소재 list'].loc[h1.index_h] or s + s_1 in h1.df['소재 list'].loc[h1.index_h]) and \
            d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index_h_next], "%Y-%m-%d %H:%M:%S") <= dt.timedelta(minutes=0):
        print("nothing")
        return 2
    if (s in h1.df['소재 list'].loc[h1.index_h] or s + s_1 in h1.df['소재 list'].loc[h1.index_h]) and \
            d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index_h], "%Y-%m-%d %H:%M:%S") <= dt.timedelta(minutes=0):
        print("wrong timing")
        return 2
    elif not (s in h1.df['소재 list'].loc[h1.index_h] or s + s_1 in h1.df['소재 list'].loc[h1.index_h]) and \
            d - dt.datetime.strptime(h1.df['시작시간'].loc[h1.index_h_next], "%Y-%m-%d %H:%M:%S") > dt.timedelta(minutes=0):
        print("find_h")
        h1.search_next_h()
        if h1.flag_end == 0:
            return 0
        elif h1.flag_end == 1:
            return 2


# 프레스기 매칭 general
def matching_press_general(heating_furnace_list, p):
    par = {}
    par_flag = {}
    count = 0
    for t in heating_furnace_list:
        t.df['in'] = '[]'
        t.df['out'] = '[]'
    for i in range(len(p.index)):
        d = dt.datetime.strptime(p['시작'].loc[i], "%Y-%m-%d %H:%M:%S")
        s = p['수주번호'].loc[i]
        s_1 = p['시리얼번호'].loc[i]
        for k in range(1, len(heating_furnace_list) + 1):
            par['c'+str(k)] = 0
            par_flag['c' + str(k) + '_flag'] = 0
        if s_1[0] == '_':
            pass
        else:
            s_1 = '_' + s_1
        while 1:
            print()
            print(i, s, s + s_1, par)
            print(par_flag)
            for k in range(1, len(heating_furnace_list) + 1):
                if par['c' + str(k)] != 2 and par['c' + str(k)] != 4:
                    par['c' + str(k)] = matching_ph_part2(heating_furnace_list[k - 1], s, s_1, d, par['c' + str(k)])
                    if par['c' + str(k)] == 2 or par['c' + str(k)] == 4:
                        par_flag['c' + str(k) + '_flag'] = 1
            if condition1(par_flag):
                time_com = []
                h_com = []
                for k in range(1, len(heating_furnace_list) + 1):
                    if par['c' + str(k)] == 4:
                        time_com.append(heating_furnace_list[k - 1].temp_time)
                        h_com.append(k)
                print(par)
                if len(time_com) > 0:
                    cf = np.argmin(time_com)
                    print(cf)
                    for k in range(1, len(heating_furnace_list) + 1):
                        if k in h_com and h_com.index(k) == cf:
                            par['c' + str(k)] = 1
                        elif k in h_com and h_com.index(k) != cf:
                            par['c' + str(k)] = 2
                else:
                    pass
                print(par)
            count += 1
            if condition2(par):
                break
            elif condition3(par):
                break
            else:
                pass
        print(i)
        for k in heating_furnace_list:
            k.index = k.index_h


# if c1_flag == 1 and c2_flag == 1 and c3_flag == 1 and c4_flag == 1 and c5_flag == 1 and c6_flag == 1:
def condition1(par_flag):
    for i in par_flag:
        if par_flag[i] != 1:
            return False
    else:
        return True


# if c1 == 3 or c2 == 3 or c3 == 3 or c4 == 3 or c5 == 3 or c6 == 3:
def condition2(par):
    for i in par:
        if par[i] == 3:
            return True
    else:
        return False


# elif c1 == 2 and c2 == 2 and c3 == 2 and c4 == 2 and c5 == 2 and c6 == 2:
def condition3(par):
    for i in par:
        if par[i] != 2:
            return False
    else:
        return True
