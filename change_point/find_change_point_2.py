from bases import *
from constant.constant_data_make import *

thr_end = 0
flag_start = 0
flag_end = 0
flag_temp = 0

# Data input,reinforcement, smoothing
def data_manipulates(data, num, time):
    reinforce(data)
    start_real, end_real = st_end_all(num, time)
    print(len(start_real), start_real)
    print(len(end_real), end_real)
    return start_real, end_real


# change point detection algorithm
def find_all(data, change_point, num, start_real, end_real):

    global thr_end
    global flag_start
    global flag_end
    global flag_temp
    thr_end = 0
    flag_start = 0
    flag_end = 0
    flag_temp = 0

    # dict of each time list
    time_dict = {'real_start_time_list': start_real, 'real_end_time_list': end_real,
                 'fixed_end_time_list': [], 'fixed_start_time_list': [], 'heat_ended_time_list': []}

    # dict of each phases list of start/end index pairs
    phase_list_dict = {'heat': [], 'hold': [], 'open': []}

    #
    find_change_point_dict = {'before_state': None, 'after_state': None, 'door_open_estimate': None,
                              'door_open_save': None, 'door_close_estimate': None, 'door_open': None,
                              'door_close_save': None, 'reheat_end_candidate': None, 'door_close': None}

    #
    heating_parameter_dict = {'heat_start_index': None, 'heat_end_index': None, 'real_heat_start_index': None,
                              'heat_start_temperature': None, 'heat_start_time': None, 'is_heat_start': False,
                              'gas_condition': False, 'time_condition': False, 'count_0_gas': 0, 'num_of_door_open': 0,
                              'last_work_end_time': None, 'wait_until_door_open_after_heating': False}

    #
    status_dict = {'furnace': 'initial', 'cycle': 'initial'}

    # 시작상태 세팅
    initialize_status(data[0]['TIME'], time_dict['real_start_time_list'][0], heating_parameter_dict, status_dict)
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
        if status_dict['furnace'] == 'wait_for_heating':
            before_work(WINDOW_DATA, i, current, num, change_point, heating_parameter_dict, status_dict, find_change_point_dict, time_dict, phase_list_dict, thr_end)
            if status_dict['furnace'] == 'after_heating_to_end':
                differential_arr = get_diff(WINDOW_DATA, current)
                door_close_index = None

        # 가열완료 - 가동종료 사이클
        elif status_dict['furnace'] == 'after_heating_to_end':
            differential_arr = get_diff(WINDOW_DATA, current)
            # print(WINDOW_DATA[current]['TIME'])
            # print(find_change_point_dict)
            if is_work_end(WINDOW_DATA, current, time_dict):
                work_end(WINDOW_DATA, current, time_dict, phase_list_dict, find_change_point_dict, heating_parameter_dict, status_dict, change_point, i)
            if is_this_point_can_be_a_change_point(differential_arr[0], differential_arr[1]):
                point_detection(WINDOW_DATA, current, change_point, differential_arr[0], differential_arr[1], num, i,
                                time_dict, status_dict, heating_parameter_dict, find_change_point_dict, phase_list_dict)
            elif is_this_point_can_be_a_change_point(differential_arr[2], differential_arr[3]):
                point_detection(WINDOW_DATA, current, change_point, differential_arr[2], differential_arr[3], num, i,
                                time_dict, status_dict, heating_parameter_dict, find_change_point_dict, phase_list_dict)
            elif status_dict['cycle'] == 'door_close':
                if num == 1:
                    thr = 40
                else:
                    thr = 70
                module_door_close(WINDOW_DATA, current, change_point, i, thr, time_dict, status_dict, heating_parameter_dict, find_change_point_dict, phase_list_dict)
    for i in range(TIME_MARGIN):
        change_point.append(None)

    for j in range(flag_start + 1, flag_end):
        change_point[j] = None

    return time_dict, phase_list_dict


def before_work(WINDOW_DATA, i, current, num, change_point, heating_parameter_dict, status_dict,
                find_change_point_dict, time_dict, phase_list_dict, thr_end):
    # 가열로 가열 시작 전
    if heating_parameter_dict['is_heat_start'] is False:
        # 가열로 시간 조건
        if is_work_start(WINDOW_DATA, current, time_dict):
            heating_parameter_dict['time_condition'] = True
            heating_parameter_dict['real_heat_start_index'] = i
        if is_work_end(WINDOW_DATA, current, time_dict) and heating_parameter_dict['real_heat_start_index'] is not None:
            heating_parameter_dict['last_work_end_time'] = WINDOW_DATA[current]['TIME']
            heating_parameter_dict['time_condition'] = False
            heating_parameter_dict['real_heat_start_index'] = None
        # 가열로 가스 조건
        check_gas_condition(WINDOW_DATA, current, i, heating_parameter_dict, status_dict, find_change_point_dict)
        # 두 조건 체크
        if heating_parameter_dict['time_condition'] and heating_parameter_dict['gas_condition']:
            heating_parameter_dict['is_heat_start'] = True
    # 가열로 가열 시작 중
    if heating_parameter_dict['is_heat_start'] is True:
        differential_arr = get_diff(WINDOW_DATA, current)
        if is_this_point_can_be_a_change_point(differential_arr[0], differential_arr[1]):
            point_detection_while_heating(WINDOW_DATA, current, change_point, differential_arr[0],
                                          differential_arr[1], num, i, time_dict, status_dict, heating_parameter_dict,
                                          find_change_point_dict)
        elif is_this_point_can_be_a_change_point(differential_arr[2], differential_arr[3]):
            point_detection_while_heating(WINDOW_DATA, current, change_point, differential_arr[2],
                                          differential_arr[3], num, i, time_dict, status_dict, heating_parameter_dict,
                                          find_change_point_dict)
        if WINDOW_DATA[current]['GAS'] == 0 and WINDOW_DATA[current]['GAS_OFF'] == 0:
            heating_parameter_dict['count_0_gas'] += 1
        if is_work_end(WINDOW_DATA, current, time_dict):
            heating_parameter_dict['last_work_end_time'] = WINDOW_DATA[current]['TIME']
            status_dict['furnace'] = 'wait_for_heating'
            reset_change_point_dict(find_change_point_dict)
            reset_heating_phase_dict(heating_parameter_dict)
        if float(WINDOW_DATA[current]['TEMPERATURE']) > thr_end:
            past_data = []
            future_data = []
            for j in range(TIME_MARGIN):
                past_data.append(float(WINDOW_DATA[current + 1 + j]['TEMPERATURE']))
                future_data.append(float(WINDOW_DATA[current - TIME_MARGIN + j]['TEMPERATURE']))
            if abs(np.mean(past_data) - np.mean(future_data)) < 1:
                change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
                heating_parameter_dict['heat_end_index'] = i
                time_dict['heat_ended_time_list'].append(WINDOW_DATA[current]['TIME'])
                if status_dict['cycle'] == 'door_close':
                    heating_parameter_dict['num_of_door_open'] += 1
                if heating_parameter_dict['real_heat_start_index'] is not None:
                    time_dict['fixed_start_time_list'].append(heating_parameter_dict['heat_start_time'])
                    change_point[heating_parameter_dict['heat_start_index']] = heating_parameter_dict['heat_start_temperature']
                    setting_heat_phase_list(phase_list_dict, heating_parameter_dict, find_change_point_dict)
                reset_change_point_dict(find_change_point_dict)
                reset_heating_phase_dict(heating_parameter_dict)
                status_dict['furnace'] = 'after_heating_to_end'
                status_dict['cycle'] = 'initial'
                heating_parameter_dict['wait_until_door_after_heating'] = True
            past_data.clear()
            future_data.clear()


def point_detection_while_heating(WINDOW_DATA, current, change_point, pp_mean, ff_mean, num, i, time_dict, status_dict,
                                  heating_parameter_dict, find_change_point_dict):
    if num == 1:
        thr = 40
    else:
        thr = 70

    global thr_end
    global flag_start
    global flag_end
    global flag_temp
    past_data = []
    future_data = []
    for j in range(TIME_MARGIN):
        future_data.append(float(WINDOW_DATA[current + 1 + j]['TEMPERATURE']))
        past_data.append(float(WINDOW_DATA[current - TIME_MARGIN + j]['TEMPERATURE']))
    if abs(np.mean(future_data) - np.mean(past_data)) < 1 and WINDOW_DATA[current]['TEMPERATURE'] < thr_end:
        if flag_start == 0:
            change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
            flag_start = i
            flag_end = i
            flag_temp = WINDOW_DATA[current]['TEMPERATURE']
        else:
            if flag_temp - thr <= WINDOW_DATA[current]['TEMPERATURE'] <= flag_temp + thr:
                change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
                flag_temp = (flag_temp + WINDOW_DATA[current]['TEMPERATURE']) / 2
                if i > flag_end + 3000:
                    for j in range(flag_start + 1, flag_end):
                        change_point[j] = None
                    flag_start = i
                flag_end = i
            else:
                for j in range(flag_start + 1, flag_end):
                    change_point[j] = None
                change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
                flag_start = i
                flag_end = i
                flag_temp = WINDOW_DATA[current]['TEMPERATURE']

    # 대분류
    categorize(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, time_dict, status_dict, find_change_point_dict)

    # 대분류를 바탕으로 change point detect

    if status_dict['cycle'] == 'initial':
        # 초기화
        if find_change_point_dict['before_state'] is None:
            find_change_point_dict['before_state'] = find_change_point_dict['after_state']
            status_dict['cycle'] = 'wait'

    elif status_dict['cycle'] == 'door_open':
        module_door_open_while_heat(WINDOW_DATA, current, i, status_dict, find_change_point_dict)

    elif status_dict['cycle'] == 'door_close':
        module_door_close_while_heat(WINDOW_DATA, current, change_point, i, thr, time_dict, status_dict, heating_parameter_dict, find_change_point_dict)

    elif status_dict['cycle'] == 'wait':
        # 대기상태
        if is_ready(find_change_point_dict):
            find_change_point_dict['before_state'] = find_change_point_dict['after_state']


# change_point Algorithm functions:
def point_detection(WINDOW_DATA, current, change_point, pp_mean, ff_mean, num, i, time_dict, status_dict,
                                  heating_parameter_dict, find_change_point_dict, phase_list_dict):
    if num == 1:
        thr = 40
    else:
        thr = 70

    # 대분류
    categorize(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, time_dict, status_dict, find_change_point_dict)

    # 대분류를 바탕으로 change point detect

    if status_dict['cycle'] == 'initial':
        # 초기화
        if find_change_point_dict['before_state'] is None:
            find_change_point_dict['before_state'] = find_change_point_dict['after_state']
            status_dict['cycle'] = 'wait'

    elif status_dict['cycle'] == 'door_open':
        module_door_open(WINDOW_DATA, current, i, status_dict, find_change_point_dict)

    elif status_dict['cycle'] == 'door_close':
        module_door_close(WINDOW_DATA, current, change_point, i, thr, time_dict, status_dict, heating_parameter_dict, find_change_point_dict, phase_list_dict)

    elif status_dict['cycle'] == 'wait':
        # 대기상태
        if is_ready(find_change_point_dict):
            find_change_point_dict['before_state'] = find_change_point_dict['after_state']


def categorize(pp_mean, ff_mean, WINDOW_DATA, current, thr, i, time_dict, status_dict, find_change_point_dict):
    #  case : + > 0
    if pp_mean > 0 and ff_mean == 0:
        find_change_point_dict['after_state'] = 'plus-to-zero'
        if status_dict['cycle'] == 'door_close' and find_change_point_dict['door_close_estimate'] is not None:
            if find_change_point_dict['reheat_end_candidate'] is None:
                find_change_point_dict['reheat_end_candidate'] = {'index': i, 'now': WINDOW_DATA[current]}
            elif find_change_point_dict['reheat_end_candidate']['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                find_change_point_dict['reheat_end_candidate'] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : + > -
    elif pp_mean > 0 and ff_mean < 0:
        find_change_point_dict['after_state'] = 'plus-to-minus'
        if not find_change_point_dict['door_open']:
            if is_door_open(WINDOW_DATA, current, thr, time_dict, find_change_point_dict, i):
                find_change_point_dict['door_open'] = True
                status_dict['cycle'] = 'door_open'
                find_change_point_dict['before_state'] = find_change_point_dict['after_state']
        if find_change_point_dict['door_open']:
            if find_change_point_dict['door_open_estimate'] is not None and \
                    find_change_point_dict['door_open_estimate']['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                find_change_point_dict['door_open_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
        if status_dict['cycle'] == 'door_close' and find_change_point_dict['door_close_estimate'] is not None:
            if find_change_point_dict['reheat_end_candidate'] is None:
                find_change_point_dict['reheat_end_candidate'] = {'index': i, 'now': WINDOW_DATA[current]}
            elif find_change_point_dict['reheat_end_candidate']['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE'] - 5:
                find_change_point_dict['reheat_end_candidate'] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : - > 0
    elif pp_mean < 0 and ff_mean == 0:
        find_change_point_dict['after_state'] = 'minus-to-zero'
        if find_change_point_dict['door_open']:
            if find_change_point_dict['door_close_estimate'] is None:
                find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
            elif find_change_point_dict['door_close_estimate']['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
            if find_change_point_dict['reheat_end_candidate'] is not None:
                find_change_point_dict['reheat_end_candidate'] = None

    # case : - > +
    elif pp_mean < 0 and ff_mean > 0:
        find_change_point_dict['after_state'] = 'minus-to-plus'
        if find_change_point_dict['door_open']:
            if find_change_point_dict['door_close_estimate'] is None:
                find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
            elif find_change_point_dict['door_close_estimate']['now']['TEMPERATURE'] > WINDOW_DATA[current]['TEMPERATURE']:
                find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
            if find_change_point_dict['reheat_end_candidate'] is not None:
                find_change_point_dict['reheat_end_candidate'] = None

    # case : 0 > -
    elif pp_mean == 0 and ff_mean < 0:
        find_change_point_dict['after_state'] = 'zero-to-minus'
        if not find_change_point_dict['door_open']:
            if is_door_open(WINDOW_DATA, current, thr, time_dict, find_change_point_dict, i):
                find_change_point_dict['door_open'] = True
                status_dict['cycle'] = 'door_open'
                find_change_point_dict['before_state'] = find_change_point_dict['after_state']
        if find_change_point_dict['door_open']:
            if find_change_point_dict['door_open_estimate'] is not None and \
                    find_change_point_dict['door_open_estimate']['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                find_change_point_dict['door_open_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}

    # case : 0 > +
    elif pp_mean == 0 and ff_mean > 0:
        find_change_point_dict['after_state'] = 'zero-to-plus'


def module_door_open_while_heat(WINDOW_DATA, current, i, status_dict, find_change_point_dict):
    # 문열림 갱신
    if find_change_point_dict['before_state'] == 'plus-to-minus' and find_change_point_dict['after_state'] == 'zero-to-minus':
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']

    # 문닫힘
    elif check_door_close(WINDOW_DATA, current, find_change_point_dict):
        if find_change_point_dict['door_open_save'] is None:
            find_change_point_dict['door_open_save'] = {'index': find_change_point_dict['door_open_estimate']['index'], 'now': find_change_point_dict['door_open_estimate']['now']}
        if find_change_point_dict['door_close_estimate'] is None:
            find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
        find_change_point_dict['door_open_estimate'] = None
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']
        status_dict['cycle'] = 'door_close'
        find_change_point_dict['door_close'] = [find_change_point_dict['door_close_estimate']['now']['TIME'],
                                                find_change_point_dict['door_close_estimate']['now']['TEMPERATURE'],
                                                find_change_point_dict['door_close_estimate']['index']]


# 문열림 - 문닫힘 구간
def module_door_open(WINDOW_DATA, current, i, status_dict, find_change_point_dict):
    # 문열림 갱신
    if find_change_point_dict['before_state'] == 'plus-to-minus' and find_change_point_dict['after_state'] == 'zero-to-minus':
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']

    # 문닫힘
    elif check_door_close(WINDOW_DATA, current, find_change_point_dict):
        if find_change_point_dict['door_open_save'] is None:
            find_change_point_dict['door_open_save'] = {'index': find_change_point_dict['door_open_estimate']['index'], 'now': find_change_point_dict['door_open_estimate']['now']}
        if find_change_point_dict['door_close_estimate'] is None:
            find_change_point_dict['door_close_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
        find_change_point_dict['door_open_estimate'] = None
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']
        status_dict['cycle'] = 'door_close'


def module_door_close_while_heat(WINDOW_DATA, current, change_point, i, thr, time_dict, status_dict, heating_parameter_dict, find_change_point_dict):
    # 문닫힘 갱신
    if is_still_lower_point(find_change_point_dict):
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']

    # 재가열 완료
    elif check_reheat_end(find_change_point_dict, WINDOW_DATA, current, i, thr, time_dict):
        # change_point[find_change_point_dict['door_close_estimate']['index']] = find_change_point_dict['door_close_estimate']['now']['TEMPERATURE']
        # change_point[find_change_point_dict['door_open_save']['index']] = find_change_point_dict['door_open_save']['now']['TEMPERATURE']
        # change_point[find_change_point_dict['reheat_end_candidate']['index']] = find_change_point_dict['reheat_end_candidate']['now']['TEMPERATURE']
        find_change_point_dict['door_open'] = False
        find_change_point_dict['door_open_save'] = None
        find_change_point_dict['reheat_end_candidate'] = None
        if not find_change_point_dict['door_open'] and find_change_point_dict['after_state'] == 'plus-to-minus':
            if is_door_open(WINDOW_DATA, current, thr, time_dict, find_change_point_dict, i):
                find_change_point_dict['door_open'] = True
                status_dict['cycle'] = 'door_open'
                find_change_point_dict['before_state'] = 'plus-to-minus'
        if not find_change_point_dict['door_open']:
            status_dict['cycle'] = 'wait'
        find_change_point_dict['door_close'] = [find_change_point_dict['door_close_estimate']['now']['TIME'],
                                                find_change_point_dict['door_close_estimate']['now']['TEMPERATURE'],
                                                find_change_point_dict['door_close_estimate']['index']]
        find_change_point_dict['door_close_estimate'] = None
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']
        heating_parameter_dict['num_of_door_open'] += 1


# 문닫힘 - 재가열완료 구간
def module_door_close(WINDOW_DATA, current, change_point, i, thr, time_dict, status_dict, heating_parameter_dict, find_change_point_dict, phase_list_dict):
    # 문닫힘 갱신
    if is_still_lower_point(find_change_point_dict):
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']

    # 재가열 완료
    elif check_reheat_end(find_change_point_dict, WINDOW_DATA, current, i, thr, time_dict):
        # change_point[find_change_point_dict['door_close_estimate']['index']] = find_change_point_dict['door_close_estimate']['now']['TEMPERATURE']
        # change_point[find_change_point_dict['door_open_save']['index']] = find_change_point_dict['door_open_save']['now']['TEMPERATURE']
        # change_point[find_change_point_dict['reheat_end_candidate']['index']] = find_change_point_dict['reheat_end_candidate']['now']['TEMPERATURE']
        phase_list_dict['open'].append([find_change_point_dict['door_open_save']['index'],
                                        find_change_point_dict['door_close_estimate']['index'],
                                        find_change_point_dict['reheat_end_candidate']['index']])
        if heating_parameter_dict['wait_until_door_open_after_heating']:
            phase_list_dict['hold'].append([heating_parameter_dict['heat_end_index'],
                                             find_change_point_dict['door_open_save']['index']])
            heating_parameter_dict['wait_until_door_open_after_heating'] = False
            heating_parameter_dict['heat_end_index'] = None
        elif find_change_point_dict['door_close'] is not None:
            phase_list_dict['hold'].append([find_change_point_dict['door_close'],
                                            find_change_point_dict['door_open_save']['index']])
            find_change_point_dict['door_close'] = None
        find_change_point_dict['door_open'] = False
        find_change_point_dict['door_open_save'] = None
        if not find_change_point_dict['door_open'] and find_change_point_dict['after_state'] == 'plus-to-minus':
            if is_door_open(WINDOW_DATA, current, thr, time_dict, find_change_point_dict, i):
                find_change_point_dict['door_open'] = True
                status_dict['cycle'] = 'door_open'
                find_change_point_dict['before_state'] = 'plus-to-minus'
        if not find_change_point_dict['door_open']:
            status_dict['cycle'] = 'wait'
            find_change_point_dict['door_close'] = find_change_point_dict['reheat_end_candidate']['index']
        find_change_point_dict['reheat_end_candidate'] = None
        find_change_point_dict['door_close_estimate'] = None
        find_change_point_dict['before_state'] = find_change_point_dict['after_state']


def initialize_status(time_zero, start_zero, heating_phase_dict, status_dict):
    if (time_zero - start_zero).total_seconds() >= 0:
        status_dict['furnace'] = 'wait_for_heating'
        heating_phase_dict['time_condition'] = True
        heating_phase_dict['real_heat_start_index'] = 0
    else:
        status_dict['furnace'] = 'wait_for_heating'


def check_gas_condition(WINDOW_DATA, current, i, heating_phase_dict, status_dict, find_change_point_dict):
    if not heating_phase_dict['gas_condition']:
        for t in range(1, GAS_CONDITION_WINDOW + 1):
            if WINDOW_DATA[current + t]['GAS'] == 0:
                return
        if heating_phase_dict['heat_start_index'] is None:
            heating_phase_dict['heat_start_index'] = i
            heating_phase_dict['heat_start_temperature'] = WINDOW_DATA[current]['TEMPERATURE']
            heating_phase_dict['heat_start_time'] = WINDOW_DATA[current]['TIME']
            heating_phase_dict['gas_condition'] = True
    else:
        sum_gas = 0
        for j in range(HALF_MARGIN):
            sum_gas += WINDOW_DATA[current + j]['GAS']
        if sum_gas == 0:
            heating_phase_dict['heat_start_index'] = None
            heating_phase_dict['gas_condition'] = False
            heating_phase_dict['num_of_door_open'] = 0
            status_dict['cycle_status'] = 'initial'
            reset_change_point_dict(find_change_point_dict)
        else:
            pass


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


def is_this_point_can_be_a_change_point(past_grad, future_grad):
    if (past_grad != 0 or future_grad != 0) and past_grad * future_grad <= 0:
        return True
    else:
        return False


def is_work_end(WINDOW_DATA, current, time_dict):
    if WINDOW_DATA[current]['TIME'] in time_dict['real_end_time_list']:
        return True
    else:
        return False


def is_work_start(WINDOW_DATA, current, time_dict):
    if WINDOW_DATA[current]['TIME'] in time_dict['real_start_time_list']:
        return True
    else:
        return False


def work_end(WINDOW_DATA, current, time_dict, phase_list_dict, find_change_point_dict, heating_parameter_dict, status_dict, change_point, i):
    heating_parameter_dict['last_work_end_time'] = WINDOW_DATA[current]['TIME']
    if not find_change_point_dict['door_open'] and find_change_point_dict['door_close'] is not None:
        phase_list_dict['hold'].append([find_change_point_dict['door_close'], i])
    if find_change_point_dict['door_open'] and (find_change_point_dict['door_open_estimate'] is not None or find_change_point_dict['door_open_save'] is not None):
        if find_change_point_dict['door_open_estimate'] is not None:
            heating_parameter_dict['last_work_end_time'] = find_change_point_dict['door_open_estimate']['now']['TIME']
        elif find_change_point_dict['door_open_save'] is not None:
            heating_parameter_dict['last_work_end_time'] = find_change_point_dict['door_open_save']['now']['TIME']
    reset_change_point_dict(find_change_point_dict)
    change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
    if is_work_start(WINDOW_DATA, current, time_dict):
        heating_parameter_dict['time_condition'] = True
        heating_parameter_dict['real_heat_start_index'] = i
    find_change_point_dict['door_close'] = None
    time_dict['fixed_end_time_list'].append(heating_parameter_dict['last_work_end_time'])
    status_dict['furnace'] = 'wait_for_heating'


def reset_change_point_dict(find_change_point_dict):
    for i in find_change_point_dict:
        find_change_point_dict[i] = None


def reset_heating_phase_dict(heating_parameter_dict):
    heating_parameter_dict['heat_start_index'] = None
    heating_parameter_dict['real_heat_start_index'] = None
    heating_parameter_dict['heat_start_temperature'] = None
    heating_parameter_dict['heat_start_time'] = None
    heating_parameter_dict['is_heat_start'] = False
    heating_parameter_dict['gas_condition'] = False
    heating_parameter_dict['time_condition'] = False
    heating_parameter_dict['count_0_gas'] = 0
    heating_parameter_dict['num_of_door_open'] = 0


def setting_heat_phase_list(phase_list_dict, heating_parameter_dict, find_change_point_dict):
    if heating_parameter_dict['num_of_door_open'] == 1:
        if find_change_point_dict['door_close'] is None:
            phase_list_dict['heat'].append([heating_parameter_dict['heat_start_index'],
                                            heating_parameter_dict['heat_end_index'],
                                            heating_parameter_dict['real_heat_start_index'],
                                            heating_parameter_dict['count_0_gas'],
                                            0,
                                            heating_parameter_dict['heat_start_time'],
                                            heating_parameter_dict['heat_start_temperature'],
                                            heating_parameter_dict['heat_start_index'],
                                            heating_parameter_dict['last_work_end_time']])
        else:
            phase_list_dict['heat'].append([heating_parameter_dict['heat_start_index'],
                                            heating_parameter_dict['heat_end_index'],
                                            heating_parameter_dict['real_heat_start_index'],
                                            heating_parameter_dict['count_0_gas'],
                                            heating_parameter_dict['num_of_door_open'],
                                            find_change_point_dict['door_close'][0],
                                            find_change_point_dict['door_close'][1],
                                            find_change_point_dict['door_close'][2],
                                            heating_parameter_dict['last_work_end_time']])
    else:
        if find_change_point_dict['door_close'] is None or heating_parameter_dict['num_of_door_open'] == 0:
            phase_list_dict['heat'].append([heating_parameter_dict['heat_start_index'],
                                            heating_parameter_dict['heat_end_index'],
                                            heating_parameter_dict['real_heat_start_index'],
                                            heating_parameter_dict['count_0_gas'],
                                            heating_parameter_dict['num_of_door_open'],
                                            0,
                                            0,
                                            None,
                                            heating_parameter_dict['last_work_end_time']])
        else:
            phase_list_dict['heat'].append([heating_parameter_dict['heat_start_index'],
                                            heating_parameter_dict['heat_end_index'],
                                            heating_parameter_dict['real_heat_start_index'],
                                            heating_parameter_dict['count_0_gas'],
                                            heating_parameter_dict['num_of_door_open'],
                                            find_change_point_dict['door_close'][0],
                                            find_change_point_dict['door_close'][1],
                                            find_change_point_dict['door_close'][2],
                                            heating_parameter_dict['last_work_end_time']])


def check_door_close(WINDOW_DATA, current, find_change_point_dict):
    if find_change_point_dict['door_open'] and \
            (find_change_point_dict['before_state'] == 'plus-to-minus' or find_change_point_dict['before_state'] == 'zero-to-minus') and\
            (find_change_point_dict['after_state'] == 'minus-to-plus' or find_change_point_dict['after_state'] == 'minus-to-zero') and\
            find_change_point_dict['door_close_save'] <= WINDOW_DATA[current]['TIME']:
        return True
    else:
        return False


def is_still_lower_point(find_change_point_dict):
    if (find_change_point_dict['before_state'] == 'minus-to-zero' or find_change_point_dict[
        'before_state'] == 'minus-to-plus') and \
            (find_change_point_dict['after_state'] == 'minus-to-zero' or find_change_point_dict[
                'before_state'] == 'minus-to-plus'):
        return True
    else:
        return False


def is_ready(find_change_point_dict):
    if (find_change_point_dict['before_state'] == 'zero-to-plus' or find_change_point_dict['before_state'] == 'plus-to-zero') and\
            find_change_point_dict['after_state'] == 'zero-to-minus':
        return True
    else:
        return False


def check_reheat_end(find_change_point_dict, WINDOW_DATA, current, i, thr, time_dict):
    if find_change_point_dict['door_open'] is True and (reheat_end(WINDOW_DATA, current, i, HALF_MARGIN, find_change_point_dict) or is_door_open(WINDOW_DATA, current, thr, time_dict)) and\
            find_change_point_dict['reheat_end_candidate'] is not None and find_change_point_dict['door_close_estimate'] is not None:
        return True
    else:
        return False


def reheat_end(WINDOW_DATA, current, i, margin, find_change_point_dict):
    p = []
    f = []
    for j in range(0, margin):
        p.append(WINDOW_DATA[current - j]['TEMPERATURE'])
        f.append(WINDOW_DATA[current + j]['TEMPERATURE'])
    if abs(np.mean(p) - np.mean(f)) < 1:
        find_change_point_dict['reheat_end_candidate'] = {'index': i, 'now': WINDOW_DATA[current]}
        return True
    else:
        return False


def is_door_open(WINDOW_DATA, current, thr, time_dict, find_change_point_dict=None, i=None):
    for l in range(1, TIME_MARGIN):
        if is_work_end(WINDOW_DATA, current, time_dict):
            return False
        elif float(WINDOW_DATA[current]['TEMPERATURE']) - float(WINDOW_DATA[current + l]['TEMPERATURE']) > thr:
            if find_change_point_dict is not None and i is not None:
                find_change_point_dict['door_close_save'] = WINDOW_DATA[current + l]['TIME']
                if find_change_point_dict['door_open_estimate'] is None or \
                    find_change_point_dict['door_open_estimate']['now']['TEMPERATURE'] < WINDOW_DATA[current]['TEMPERATURE']:
                    find_change_point_dict['door_open_estimate'] = {'index': i, 'now': WINDOW_DATA[current]}
            return True
    return False
