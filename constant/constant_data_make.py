import pandas as pd


p5 = [1, 2, 3, 4, 5]
p15 = [6, 13, 17, 18, 19, 20]
p_all = [1, 2, 3, 4, 5, 6, 13, 17, 18, 19, 20]
p_all2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
p_bum = [[1], [2, 3, 4, 5, 6], [17, 18, 19, 20], [2, 3], [4, 5, 6]]

TIME_MARGIN = 10
HALF_MARGIN = 5
GAS_CONDITION_WINDOW = 3

# work_ = [19]
work_ = p_all

work_space = ''

base_path = './data_201907~202003/'
time_path = base_path + 'data/start_end_re1.csv'

df_mat = pd.read_csv(base_path + 'data/' + work_space + 'material_par3.csv', encoding='euc-kr')
