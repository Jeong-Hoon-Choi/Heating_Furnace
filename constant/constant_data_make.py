import pandas as pd


p5 = [1, 2, 3, 4, 5]
p15 = [6, 13, 17, 18, 19, 20]
p_all = [1, 2, 3, 4, 5, 6, 13, 17, 18, 19, 20]
p_all2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# p_bum = [[1], [2, 3, 4, 5, 6], [17, 18, 19, 20], [2, 3], [4, 5, 6]]
p_bum = [[1, 2, 3, 4, 5, 6, 13, 17, 18, 19, 20], [1], [2, 3, 4, 5, 6], [17, 18, 19, 20], [2, 3], [4, 5, 6]]
# p_bum = [[1, 2, 3, 4, 5, 6, 13, 17, 18, 19, 20]]

TIME_MARGIN = 10
HALF_MARGIN = 5
GAS_CONDITION_WINDOW = 3

# work_ = [19]
work_ = p_all

work_space = ''

# base_path = './data_201907~202003/'
base_path = './data_202003~202007/'
time_path = base_path + 'data/start_end_re1.csv'
sensitive_path = base_path + 'data/sensitive.csv'
clustering_condition_constant = ['민감만/1h제외', '민감만/1h포함', '민감제외/1h제외', '민감제외/1h포함', '전부/1h제외', '전부/1h포함']

df_mat = pd.read_csv(base_path + 'data/' + work_space + 'material_par3.csv', encoding='euc-kr')
