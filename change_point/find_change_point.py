from change_point.while_heating import *
from change_point.after_heating import *


# change point detection algorithm
def find_all(data, change_point, num, TIME_MARGIN):
    p = []
    f = []
    temp_low = None
    temp_s = None
    flag_status = 0
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
    # temp_array = [0, 0, None, 0, 0, 0, 0, 0]
    # flag_past, flag_now, temp_grey, temp_yellow, temp_low, flag_open, temp_open
    # 시작상태 세팅
    flag_status, flag_s_1, temp_s = initialize_status(data[0]['TIME'], start_real[0], flag_status, flag_s_1, temp_s)
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
        if flag_status == 0:
            flag_status, flag_start, flag_s_1, flag_s_2, count_0_gas, temp_s, temp_low, save_temp, \
            save_time, cycle_status, num_of_door_open, close_while_heat, save_end, temp_s_time = \
                before_work(WINDOW_DATA, i, current, change_point, count_0_gas, temp_low, temp_s, flag_start, flag_s_1,
                            flag_s_2, flag_status, save_temp, save_time, num, cycle_status,
                            TIME_MARGIN, p, f, num_of_door_open, close_while_heat, save_end, temp_s_time, thr_end)
            if flag_status == 1:
                differential_arr = make_check_parameter(int(TIME_MARGIN / 2), WINDOW_DATA, current, p, f)
                close = 0

        # 가열완료 - 가동종료 사이클
        elif flag_status == 1:
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
                flag_status = 0
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
                else:
                    thr = 70
                cycle_status = module_door_close(WINDOW_DATA, current, change_point, i, thr, cycle_status)

    for i in range(TIME_MARGIN):
        change_point.append(None)
