from change_point.change_point_calulate import *


# change_point Algorithm functions:
def point_detection(WINDOW_DATA, current, change_point, pp_mean, ff_mean, num, i, cycle_status):
    global temp_array
    if num == 1:
        thr = 40
    else:
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
    return cycle_status
