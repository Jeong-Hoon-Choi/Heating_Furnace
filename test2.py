def work_end(WINDOW_DATA, current, find_change_point_dict, hold, fixed_end_time_list, real_start_time_list, change_point, i,
             flag_s_1, temp_s, door_close_index):
    save_end = WINDOW_DATA[current]['TIME']
    if find_change_point_dict[5] == 0 and door_close_index is not None:
        hold.append([door_close_index, i])
    if find_change_point_dict['door_open'] is True:
        if find_change_point_dict['door_open_estimate'] is not None:
            save_end = find_change_point_dict[2]['now']['TIME']
        elif find_change_point_dict['door_open_save'] is not None:
            save_end = find_change_point_dict[3]['now']['TIME']
    for i in find_change_point_dict:
        find_change_point_dict[i] = None
    change_point[i] = WINDOW_DATA[current]['TEMPERATURE']
    if WINDOW_DATA[current]['TIME'] in real_start_time_list:
        flag_s_1 = 1
        temp_s = i
    door_close_index = None
    fixed_end_time_list.append(save_end)
    furnace_status = 'wait_for_heating'
    return flag_s_1, temp_s, door_close_index, furnace_status
