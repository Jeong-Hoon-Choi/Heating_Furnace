from bases import *


find_change_point_dict = {'before_state': None, 'after_state': None, 'door_open_estimate': None,
                          'door_open_save': None, 'door_close_estimate': None, 'door_open': None,
                          'door_close_save': None, 'door_open_candidate': None}
temp_array = [0, 0, None, 0, 0, 0, 0, 0]
start_fix = []
end_fix = []
op_en = []
hold = []
heat = []
flag_ready = 0
start = 0
close = 0
end_real = []
start_real = []
re = []


# global variable 초기화
def initialize_all():
    global op_en
    global hold
    global temp_array
    global flag_ready
    global start
    global close
    global heat
    global start_real
    global end_real
    global start_fix
    global end_fix
    global re
    temp_array = [0, 0, None, 0, 0, 0, 0, 0]
    op_en = []
    hold = []
    heat = []
    flag_ready = 0
    start = 0
    close = 0
    end_real = []
    start_real = []
    start_fix = []
    end_fix = []
    re = []


# Data input,reinforcement, smoothing
def data_manipulates(data, s, num, time):
    global end_real
    global start_real
    initialize_all()
    # get_data(data, s)
    reinforce(data)
    st_end_all(num, start_real, end_real, time)
    print(len(start_real), start_real)
    print(len(end_real), end_real)


# change point detection algorithm
def find_all(data, change_point, num, TIME_MARGIN):
    p = []
    f = []

    heat_start_index = 0
    door_close_index = 0
    real_end_time_list = []
    fixed_end_time_list = []
    real_start_time_list = []
    fixed_start_time_list = []
    heat_ended_time_list = []

    heat_index_list = []
    hold_index_list = []
    open_close_reheat_list = []

    temp_low = None
    temp_s = None
    furnace_status = 'initial'
    flag_start = 0
    flag_s_1 = 0
    flag_s_2 = 0
    count_0_gas = 0
    save_temp = 0
    save_time = 0
    save_end = 0
    cycle_status = 'initial'
    num_of_door_open = 0
    close_while_heat = None
    temp_s_time = None
    global op_en
    global hold
    global temp_array
    global flag_ready
    global start
    global close
    global heat
    global start_real
    global end_real
    global end_fix
    # 시작상태 세팅
    furnace_status, flag_s_1, temp_s = initialize_status(data[0]['TIME'], start_real[0])
    if num == 5 or num == 6:
        thr_end = 1220
    else:
        thr_end = 1140
    for i in range(TIME_MARGIN):
        change_point.append(None)
    for i in range(TIME_MARGIN, len(data) - TIME_MARGIN):
        WINDOW_DATA = data[i - TIME_MARGIN:i + TIME_MARGIN + 1]
        current = len(WINDOW_DATA) // 2
        # if i == 10: print(len(WINDOW_DATA), current)
        change_point.append(None)
        # 가동종료 - 가열완료 사이클
        if furnace_status == 0:
            furnace_status, flag_start, flag_s_1, flag_s_2, count_0_gas, temp_s, temp_low, save_temp, \
            save_time, cycle_status, num_of_door_open, close_while_heat, save_end, temp_s_time = \
                before_work(WINDOW_DATA, i, current, change_point, count_0_gas, temp_low, temp_s, flag_start, flag_s_1,
                            flag_s_2, furnace_status, save_temp, save_time, num, cycle_status,
                            TIME_MARGIN, p, f, num_of_door_open, close_while_heat, save_end, temp_s_time, thr_end)
            if furnace_status == 1:
                differential_arr = make_check_parameter(int(TIME_MARGIN / 2), WINDOW_DATA, current, p, f)
                close = 0

        # 가열완료 - 가동종료 사이클
        elif furnace_status == 1:
            differential_arr = make_check_parameter(int(TIME_MARGIN / 2), WINDOW_DATA, current, p, f)
            if check_end(WINDOW_DATA[current]['TIME']):
                save_end = WINDOW_DATA[current]['TIME']
                if temp_array[5] == 0 and close != 0:
                    hold.append([close, i])
                    close = 0
                if flag_ready == 1:
                    # hold.append([start, current])
                    flag_ready = 0
                if temp_array[5] == 1 and (temp_array[2] is not None or temp_array[3] != 0):
                    if temp_array[2] is not None:
                        save_end = temp_array[2]['now']['TIME']
                    elif temp_array[3] != 0:
                        save_end = temp_array[3]['now']['TIME']
                    close = 0
                end_fix.append(save_end)
                # save_end = dt.datetime
                furnace_status = 0
                temp_array = [0, 0, None, 0, 0, 0, 0, 0]
                change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
                if WINDOW_DATA[current]['TIME'] in start_real:
                    flag_s_1 = 1
                    temp_s = i
            if (differential_arr[0] != 0 or differential_arr[1] != 0) and differential_arr[0] * differential_arr[
                1] <= 0:
                cycle_status = point_detection(WINDOW_DATA, current, change_point, differential_arr[0],
                                               differential_arr[1], num, i, cycle_status)
            elif (differential_arr[2] != 0 or differential_arr[3] != 0) and differential_arr[2] * differential_arr[
                3] <= 0:
                cycle_status = point_detection(WINDOW_DATA, current, change_point, differential_arr[2],
                                               differential_arr[3], num, i, cycle_status)
            elif cycle_status == 'door_close':
                if num == 1:
                    thr = 40
                elif num != 1:
                    thr = 70
                cycle_status = module_door_close(WINDOW_DATA, current, change_point, i, thr, cycle_status)

    for i in range(TIME_MARGIN):
        change_point.append(None)


def initialize_status(time_zero, start_zero):
    if (time_zero - start_zero).total_seconds() >= 0:
        furnace_status = 'wait_for_heating'
        flag_s_1 = 1
        temp_s = 0
    else:
        furnace_status = 'wait_for_heating'
        flag_s_1 = 0
        temp_s = 0
    return furnace_status, flag_s_1, temp_s


def make_check_parameter(TIME_MARGIN, WINDOW_DATA, current, p, f):
    past = []
    future = []
    for j in range(0, TIME_MARGIN):
        past.append(float(WINDOW_DATA[current - j]['TEMPERATURE']))
        future.append((float(WINDOW_DATA[current + j]['TEMPERATURE'])))
    if len(p) < 2:
        p.append(np.mean(past))
    else:
        p[0] = p[1]
        p[1] = np.mean(past)
    if len(f) < 2:
        f.append(np.mean(future))
    else:
        f[0] = f[1]
        f[1] = np.mean(future)
    return get_diff(WINDOW_DATA, current)


def check_end(time):
    global end_real
    for e in end_real:
        if time == e:
            return True


def before_work(WINDOW_DATA, i, current, change_point, count_0_gas, temp_low, temp_s, flag_start,
                flag_s_1, flag_s_2, flag_status, save_temp, save_time, num, cycle_status,
                TIME_MARGIN, p, f, num_of_door_open, close_while_heat, save_end, temp_s_time, thr_end):
    past_data = []
    future_data = []
    global temp_array
    global start
    global flag_ready
    global heat
    global re
    if flag_start == 0:
        for s in start_real:
            if flag_start == 0 and WINDOW_DATA[current]['TIME'] == s:
                flag_s_1 = 1
                temp_s = i
                temp_s_time = WINDOW_DATA[current]['TIME']
                break
        for e in end_real:
            if WINDOW_DATA[current]['TIME'] == e and temp_s is not None and WINDOW_DATA[current]['TIME'] != temp_s_time:
                save_end = WINDOW_DATA[current]['TIME']
                flag_s_1 = 0
                temp_s = None
                break
        # 일단 가스로 만든다. 모니터링은 누락을 고려하여 만들어야 한다.
        if WINDOW_DATA[current + 1]['GAS'] > 0 and WINDOW_DATA[current + 2]['GAS'] > 0 and WINDOW_DATA[current + 3][
            'GAS'] > 0:
            if temp_low is None:
                temp_low = i
                save_temp = WINDOW_DATA[current]['TEMPERATURE']
                save_time = WINDOW_DATA[current]['TIME']
                flag_s_2 = 1
        if flag_s_2 == 1:
            sum_gas = 0
            for j in range(5):
                sum_gas += WINDOW_DATA[current + j]['GAS']
            if sum_gas == 0:
                temp_low = None
                flag_s_2 = 0
                num_of_door_open = 0
                cycle_status = 'initial'
                temp_array = [0, 0, None, 0, 0, 0, dt.datetime, 0]
            else:
                pass
        if flag_s_1 == 1 and flag_s_2 == 1:
            flag_start = 1
            # close_while_heat = None
    if flag_s_2 == 1:
        differencial_arr = make_check_parameter(int(TIME_MARGIN / 2), WINDOW_DATA, current, p, f)
        if (differencial_arr[0] != 0 or differencial_arr[1] != 0) and differencial_arr[0] * differencial_arr[1] <= 0:
            cycle_status, num_of_door_open, close_while_heat = \
                point_detection_while_heating(WINDOW_DATA, current, change_point, differencial_arr[0],
                                              differencial_arr[1], num, i, cycle_status, num_of_door_open,
                                              close_while_heat)
        elif (differencial_arr[2] != 0 or differencial_arr[3] != 0) and differencial_arr[2] * differencial_arr[3] <= 0:
            cycle_status, num_of_door_open, close_while_heat = \
                point_detection_while_heating(WINDOW_DATA, current, change_point, differencial_arr[0],
                                              differencial_arr[1], num, i, cycle_status, num_of_door_open,
                                              close_while_heat)
    if flag_start == 1:
        if WINDOW_DATA[current]['GAS'] == 0 and WINDOW_DATA[current]['GAS_OFF'] == 0:
            count_0_gas += 1
        for e in end_real:
            if WINDOW_DATA[current]['TIME'] == e:
                save_end = WINDOW_DATA[current]['TIME']
                flag_status = 0
                temp_array = [0, 0, None, 0, 0, 0, dt.datetime, 0]
                temp_low = None
                temp_s = None
                temp_s_time = None
                flag_start = 0
                flag_s_1 = 0
                flag_s_2 = 0
                count_0_gas = 0
        if float(WINDOW_DATA[current]['TEMPERATURE']) > thr_end:
            for j in range(10):
                past_data.append(float(WINDOW_DATA[current + 1 + j]['TEMPERATURE']))
                future_data.append(float(WINDOW_DATA[current - 10 + j]['TEMPERATURE']))
            if abs(np.mean(past_data) - np.mean(future_data)) < 1:
                change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
                start = i
                re.append(WINDOW_DATA[current]['TIME'])
                if cycle_status == 'door_close':
                    num_of_door_open += 1
                if temp_s is not None:
                    start_fix.append(save_time)
                    change_point[temp_low] = save_temp
                    if num_of_door_open == 1:
                        if close_while_heat is None:
                            heat.append(
                                [temp_low, start, temp_s, count_0_gas, 0, save_time, save_temp, temp_low, save_end])
                        else:
                            heat.append([temp_low, start, temp_s, count_0_gas, num_of_door_open, close_while_heat[0],
                                         close_while_heat[1], close_while_heat[2], save_end])
                    else:
                        if close_while_heat is None or num_of_door_open == 0:
                            heat.append([temp_low, start, temp_s, count_0_gas, num_of_door_open, 0, 0, None, save_end])
                        else:
                            heat.append([temp_low, start, temp_s, count_0_gas, num_of_door_open, close_while_heat[0],
                                         close_while_heat[1], close_while_heat[2], save_end])
                temp_low = None
                temp_s = None
                temp_s_time = None
                flag_status = 1
                flag_ready = 1
                flag_start = 0
                flag_s_1 = 0
                flag_s_2 = 0
                count_0_gas = 0
                save_temp = 0
                save_time = 0
                cycle_status = 'initial'
                temp_array = [0, 0, None, 0, 0, 0, dt.datetime, 0]
                num_of_door_open = 0
                close_while_heat = None
            past_data.clear()
            future_data.clear()
    return flag_status, flag_start, flag_s_1, flag_s_2, count_0_gas, temp_s, temp_low, \
           save_temp, save_time, cycle_status, num_of_door_open, close_while_heat, save_end, temp_s_time


def point_detection_while_heating(WINDOW_DATA, current, change_point, pp_mean, ff_mean,
                                  num, i, cycle_status, num_of_door_open, close_while_heat):
    global temp_array
    if num == 1:
        thr = 40
    elif num != 1:
        thr = 70

    # 대분류
    cycle_status = categorize_while_heat(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, cycle_status)

    # 대분류를 바탕으로 change point detect

    if cycle_status == 'initial':
        # 초기화
        if temp_array[0] == 0:
            temp_array[0] = temp_array[1]
            cycle_status = 'wait'
            # temp_array['previous_status'] = temp_array['current_status']

    elif cycle_status == 'door_open':
        cycle_status, close_while_heat = module_door_open_while_heat(WINDOW_DATA, current, cycle_status,
                                                                     close_while_heat)

    elif cycle_status == 'door_close':
        cycle_status, num_of_door_open, close_while_heat = \
            module_door_close_while_heat(WINDOW_DATA, current, change_point, i,
                                         thr, cycle_status, num_of_door_open, close_while_heat)

    elif cycle_status == 'wait':
        # 대기상태
        if (temp_array[0] == 9 or temp_array[0] == 1) and temp_array[1] == 8:
            temp_array[0] = temp_array[1]

    return cycle_status, num_of_door_open, close_while_heat


# change_point Algorithm functions:
def point_detection(WINDOW_DATA, current, change_point, pp_mean, ff_mean, num, i, cycle_status):
    global temp_array
    if num == 1:
        thr = 40
    elif num != 1:
        thr = 70

    # 대분류
    cycle_status = categorize(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, cycle_status)

    # 대분류를 바탕으로 change point detect
    if cycle_status == 'initial':
        # 초기화
        if temp_array[0] == 0:
            temp_array[0] = temp_array[1]
            cycle_status = 'wait'

    elif cycle_status == 'door_open':
        cycle_status = module_door_open(WINDOW_DATA, current, cycle_status, i)

    elif cycle_status == 'door_close':
        cycle_status = module_door_close(WINDOW_DATA, current, change_point, i, thr, cycle_status)

    elif cycle_status == 'wait':
        # 대기상태
        if (temp_array[0] == 9 or temp_array[0] == 1) and temp_array[1] == 8:
            temp_array[0] = temp_array[1]

    return cycle_status


def categorize_while_heat(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, cycle_status):
    global temp_array
    global close
    global flag_ready
    global start
    #  case : + > 0
    if pp_mean > 0 and ff_mean == 0:
        temp_array[1] = 1
        # temp_array['current_status'] = 'plus_to_zero'
        if cycle_status == 'door_close' and temp_array[4] != 0:
            if temp_array[7] == 0:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[7]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : + > -
    elif pp_mean > 0 and ff_mean < 0:
        temp_array[1] = 2
        # temp_array['current_status'] = 'plus_to_minus'
        if temp_array[5] == 0:
            for l in range(1, 10):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[0] = temp_array[1]
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    if flag_ready == 1 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                        flag_ready = 0
                    if close != 0 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                        close = 0
                    break
        if temp_array[5] == 1:
            if temp_array[2] is not None and temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
        if cycle_status == 'door_close' and temp_array[4] != 0:
            if temp_array[7] == 0:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[7]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : - > 0
    elif pp_mean < 0 and ff_mean == 0:
        temp_array[1] = 4
        # temp_array['current_status'] = 'minus_to_zero'
        if temp_array[5] == 1:
            if temp_array[4] == 0:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[4]['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : - > +
    elif pp_mean < 0 and ff_mean > 0:
        temp_array[1] = 6
        # temp_array['current_status'] = 'minus_to_plus'
        if temp_array[5] == 1:
            if temp_array[4] == 0:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[4]['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : 0 > -
    elif pp_mean == 0 and ff_mean < 0:
        temp_array[1] = 8
        # temp_array['current_status'] = 'zero_to_minus'
        if temp_array[5] == 0:
            for l in range(1, 10):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[0] = temp_array[1]
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    if flag_ready == 1 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                        flag_ready = 0
                    if close != 0 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                        close = 0
                    break
        if temp_array[5] == 1:
            if temp_array[2] is not None and temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : 0 > +
    elif pp_mean == 0 and ff_mean > 0:
        temp_array[1] = 9
        # temp_array['current_status'] = 'zero_to_plus'

    return cycle_status


def categorize(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, cycle_status):
    global temp_array
    global close
    global flag_ready
    global start
    global hold
    #  case : + > 0
    if pp_mean > 0 and ff_mean == 0:
        temp_array[1] = 1
        # temp_array['current_status'] = 'plus_to_zero'
        if cycle_status == 'door_close' and temp_array[4] != 0:
            if temp_array[7] == 0:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[7]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : + > -
    elif pp_mean > 0 and ff_mean < 0:
        temp_array[1] = 2
        # temp_array['current_status'] = 'plus_to_minus'
        if temp_array[5] == 0:
            for l in range(1, 5):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[0] = temp_array[1]
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    # if flag_ready == 1 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                    #     hold.append([start, i])
                    #     flag_ready = 0
                    # if close != 0 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                    #     hold.append([close, i])
                    #     close = 0
                    break
        if temp_array[5] == 1:
            if temp_array[2] is not None and temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
        if cycle_status == 'door_close' and temp_array[4] != 0:
            if temp_array[7] == 0:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[7]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                temp_array[7] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : - > 0
    elif pp_mean < 0 and ff_mean == 0:
        temp_array[1] = 4
        # temp_array['current_status'] = 'minus_to_zero'
        if temp_array[5] == 1:
            if temp_array[4] == 0:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[4]['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            if temp_array[7] != 0:
                temp_array[7] = 0

    # case : - > +
    elif pp_mean < 0 and ff_mean > 0:
        temp_array[1] = 6
        # temp_array['current_status'] = 'minus_to_plus'
        if temp_array[5] == 1:
            if temp_array[4] == 0:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            elif temp_array[4]['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
            if temp_array[7] != 0:
                temp_array[7] = 0

    # case : 0 > -
    elif pp_mean == 0 and ff_mean < 0:
        temp_array[1] = 8
        # temp_array['current_status'] = 'zero_to_minus'
        if temp_array[5] == 0:
            for l in range(1, 5):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[0] = temp_array[1]
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    # if flag_ready == 1 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                    #     hold.append([start, i])
                    #     flag_ready = 0
                    # if close != 0 and temp_array[2]['now']['TIME'] == WINDOW_DATA[current]['TIME']:
                    #     hold.append([close, i])
                    #     close = 0
                    break
        if temp_array[5] == 1:
            if temp_array[2] is not None and temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : 0 > +
    elif pp_mean == 0 and ff_mean > 0:
        temp_array[1] = 9
        # temp_array['current_status'] = 'zero_to_plus'

    return cycle_status


def module_door_open_while_heat(WINDOW_DATA, current, cycle_status, close_while_heat):
    global temp_array
    # 문열림 갱신
    if temp_array[0] == 2 and temp_array[1] == 8:
        temp_array[0] = temp_array[1]

    # 문닫힘
    elif temp_array[5] == 1 and (temp_array[0] == 8 or temp_array[0] == 2) and (
            temp_array[1] == 6 or temp_array[1] == 4) and temp_array[6] <= WINDOW_DATA[current]['TIME']:
        if temp_array[3] == 0:
            temp_array[3] = {'index': temp_array[2]['index'], 'now': temp_array[2]['now']}
        temp_array[2] = None
        temp_array[0] = temp_array[1]
        cycle_status = 'door_close'
        close_while_heat = [temp_array[4]['now']['TIME'], temp_array[4]['now']['TEMPERATURE'], temp_array[4]['index']]
    return cycle_status, close_while_heat


# 문열림 - 문닫힘 구간
def module_door_open(WINDOW_DATA, current, cycle_status, i):
    global temp_array
    # 문열림 갱신
    if temp_array[0] == 2 and temp_array[1] == 8:
        temp_array[0] = temp_array[1]

    # 문닫힘
    elif temp_array[5] == 1 and (temp_array[0] == 8 or temp_array[0] == 2) and (
            temp_array[1] == 6 or temp_array[1] == 4) and temp_array[6] <= WINDOW_DATA[current]['TIME']:
        if temp_array[3] == 0:
            temp_array[3] = {'index': temp_array[2]['index'], 'now': temp_array[2]['now']}
        if temp_array[4] == 0:
            temp_array[4] = {'index': i, 'now': WINDOW_DATA[current]}
        # print(WINDOW_DATA[current]['TIME'], temp_array)
        temp_array[2] = None
        temp_array[0] = temp_array[1]
        cycle_status = 'door_close'
    return cycle_status


def module_door_close_while_heat(WINDOW_DATA, current, change_point, i, thr, cycle_status, num_of_door_open, close_while_heat):
    global temp_array
    global close
    global op_en
    global end_real
    # 문닫힘 갱신
    if (temp_array[0] == 4 or temp_array[0] == 6) and (temp_array[1] == 4 or temp_array[1] == 6):
        temp_array[0] = temp_array[1]

    # 재가열 완료
    elif temp_array[5] == 1 and (reheat_end(WINDOW_DATA, current, 5) or is_door_open(WINDOW_DATA, current, thr)) and\
            temp_array[7] != 0 and temp_array[4] != 0:
        change_point[temp_array[4]['index']] = temp_array[4]['now']['TEMPERATURE']
        change_point[temp_array[3]['index']] = temp_array[3]['now']['TEMPERATURE']
        change_point[temp_array[7]['index']] = temp_array[7]['now']['TEMPERATURE']
        temp_array[3] = 0
        temp_array[5] = 0
        temp_array[7] = 0
        if temp_array[5] == 0 and temp_array[1] == 2:
            for l in range(1, 5):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(
                        WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    break
        if temp_array[5] == 0:
            close = i
            cycle_status = 'wait'
        close_while_heat = [temp_array[4]['now']['TIME'], temp_array[4]['now']['TEMPERATURE'], temp_array[4]['index']]
        temp_array[4] = 0
        temp_array[0] = temp_array[1]
        num_of_door_open += 1
    return cycle_status, num_of_door_open, close_while_heat


# 문닫힘 - 재가열완료 구간
def module_door_close(WINDOW_DATA, current, change_point, i, thr, cycle_status):
    global temp_array
    global close
    global op_en
    global end_real
    global flag_ready
    # 문닫힘 갱신
    if (temp_array[0] == 4 or temp_array[0] == 6) and (temp_array[1] == 4 or temp_array[1] == 6):
        temp_array[0] = temp_array[1]

    # 재가열 완료
    elif temp_array[5] == 1 and (reheat_end(WINDOW_DATA, current, 5) or is_door_open(WINDOW_DATA, current, thr)) and\
            temp_array[7] != 0 and temp_array[4] != 0:
        change_point[temp_array[4]['index']] = temp_array[4]['now']['TEMPERATURE']
        change_point[temp_array[3]['index']] = temp_array[3]['now']['TEMPERATURE']
        change_point[temp_array[7]['index']] = temp_array[7]['now']['TEMPERATURE']
        op_en.append([temp_array[3]['index'], temp_array[4]['index'], temp_array[7]['index']])
        if flag_ready == 1:
            hold.append([start, temp_array[3]['index']])
            flag_ready = 0
        elif close != 0:
            hold.append([close, temp_array[3]['index']])
            close = 0
        temp_array[3] = 0
        temp_array[5] = 0
        # temp_array[7] = 0
        if temp_array[5] == 0 and temp_array[1] == 2:
            for l in range(1, 5):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(
                        WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    break
        if temp_array[5] == 0:
            close = temp_array[7]['index']
            cycle_status = 'wait'
        temp_array[7] = 0
        temp_array[4] = 0
        temp_array[0] = temp_array[1]

    '''
    elif temp_array[5] == 1 and (temp_array[0] == 6 or temp_array[0] == 4) and (
            temp_array[1] == 1 or temp_array[1] == 2):
        # if WINDOW_DATA[current]['GAS'] > 0 and WINDOW_DATA[current]['GAS'] >= WINDOW_DATA[current + 1]['GAS']:
        if temp_array[4] != 0:
            change_point[temp_array[4]['index']] = temp_array[4]['now']['TEMPERATURE']
        change_point[temp_array[3]['index']] = temp_array[3]['now']['TEMPERATURE']
        change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
        op_en.append([temp_array[3]['index'], temp_array[4]['index'], i])
        temp_array[3] = 0
        temp_array[5] = 0
        if temp_array[5] == 0 and temp_array[1] == 2:
            for l in range(1, 5):
                if WINDOW_DATA[current]['TIME'] in end_real:
                    break
                elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(
                        WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
                    temp_array[5] = 1
                    cycle_status = 'door_open'
                    temp_array[6] = WINDOW_DATA[current + l]['TIME']
                    if temp_array[2] is None or temp_array[2]['now']['TEMPERATURE'] < WINDOW_DATA[current][
                        'TEMPERATURE']:
                        temp_array[2] = {'index': i, 'now': WINDOW_DATA[current]}
                    break
        if temp_array[5] == 0:
            close = i
            cycle_status = 'wait'
        temp_array[4] = 0
        temp_array[0] = temp_array[1]
    '''
    return cycle_status


# 기울기
def calculate_diff(data, t, num1, num2, sign):
    temp = []
    for i in range(0, num2+1):
        for j in range(i+1, num1+1):
            if sign == 'past':
                temp.append((data[t - i]['TEMPERATURE'] - data[t - j]['TEMPERATURE']) / abs(i + j))
            elif sign == 'future':
                temp.append((data[t + j]['TEMPERATURE'] - data[t + i]['TEMPERATURE']) / abs(i - j))
    if abs(np.mean(temp)) < 1:
        return 0
    else:
        return np.mean(temp)


# 기울기 계산
def get_diff(data, t):
    arr = [0, 0, 0, 0]
    arr[0] = calculate_diff(data, t, 2, 0, 'past')
    arr[1] = calculate_diff(data, t, 2, 0, 'future')
    arr[2] = calculate_diff(data, t, 2, 1, 'past')
    arr[3] = calculate_diff(data, t, 2, 1, 'future')
    return arr


def reheat_end(WINDOW_DATA, current, margin):
    p = []
    f = []
    for i in range(0, margin):
        p.append(WINDOW_DATA[current - i]['TEMPERATURE'])
        f.append(WINDOW_DATA[current + i]['TEMPERATURE'])
    if abs(np.mean(p) - np.mean(f)) < 1:
        return True
    else:
        return False


def is_door_open(WINDOW_DATA, current, thr):
    global temp_array
    global end_real
    for l in range(1, 10):
        if WINDOW_DATA[current]['TIME'] in end_real:
            return False
        elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
            return True
    return False
