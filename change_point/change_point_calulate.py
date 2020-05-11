from bases import *

# current_state, after_state, door_open_estimate, door_open_estimate2
# door_close_estimate, flag_door_open, min_door_close, door_opening_timing
find_change_point_dict = {'before_state': None, 'after_state': None, 'door_open_estimate': None,
                          'door_open_save': None, 'door_close_estimate': None, 'door_open': None,
                          'door_close_save': None, 'door_open_candidate': None}
temp_array = [0, 0, None, 0, 0, 0, 0, 0]
start_fix_list = []
end_fix_list = []
op_en = []
hold = []
heat = []
flag_ready = 0
wait_first_open = True
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
def data_manipulates(data, num, time):
    global end_real
    global start_real
    initialize_all()
    reinforce(data)
    st_end_all(num, start_real, end_real, time)
    print(len(start_real), start_real)
    print(len(end_real), end_real)


def initialize_status(time_zero, start_zero, flag_status, flag_s_1, temp_s):
    if (time_zero - start_zero).total_seconds() > 0:
        flag_status = 0
        flag_s_1 = 1
        temp_s = 0
    elif (time_zero - start_zero).total_seconds() < 0:
        flag_status = 0
        flag_s_1 = 0
    return flag_status, flag_s_1, temp_s


def check_end(time):
    global end_real
    for e in end_real:
        if time == e:
            return True


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
