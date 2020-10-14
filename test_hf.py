from constant.constant_data_make import *
from learning.ffn import *
import pandas as pd
import numpy as np


choice = ''
while(choice != '0'):
    print('Model Test')
    print('1. Energy in Increasing Part')
    print('2. Energy in Holding Part')
    print('0. Exit')
    choice = input('Enter your choice: ')

    if choice == '1' or choice == '2':
        hf_choice = input('Enter Heating Furnace Number: ')

        mode = ''
        if choice == '1':
            mode = 'energy-increasing'

            if hf_choice == '1':
                time = input('Enter Total Time (second): ')
                end_temp = input('Enter End Temperature: ')
                data = pd.DataFrame({'종료온도': [end_temp], '시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
            elif hf_choice == '18' or hf_choice == '19'or hf_choice == '20':
                time = input('Enter Total Time (second): ')
                start_temp = input('Enter Start Temperature: ')
                data = pd.DataFrame({'시작온도': [start_temp], '시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
            else:
                time = input('Enter Total Time (second): ')
                data = pd.DataFrame({'시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
        elif choice == '2':
            mode = 'energy-holding'

            if hf_choice == '4':
                time = input('Enter Total Time (second): ')
                max_weight = input('Enter Max Load Weight: ')
                data = pd.DataFrame({'장입최대중량': [max_weight], '시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
            elif hf_choice == '5':
                time = input('Enter Total Time (second): ')
                start_temp = input('Enter Start Temperature: ')
                data = pd.DataFrame({'시작온도': [start_temp], '시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
            elif hf_choice == '13':
                time = input('Enter Total Time (second): ')
                end_temp = input('Enter End Temperature: ')
                data = pd.DataFrame({'종료온도': [end_temp], '시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
            else:
                time = input('Enter Total Time (second): ')
                data = pd.DataFrame({'시간(총)': [time]})
                data = data.reset_index(drop=True)
                model = FFN.load(base_path + 'save_model/model_' + mode + '_' + hf_choice, data)
                print(mode, hf_choice, '- result:', model)
    elif choice != '0':
        print('Wrong Input!')

    print()
    print('-----------------------------------------------------------------------------------------------------------')
    print()