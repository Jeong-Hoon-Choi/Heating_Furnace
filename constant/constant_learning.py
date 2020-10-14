# path_1 = [['민감만/1h제외'], ['민감만/1h포함'],
#           ['민감제외/1h제외'], ['민감제외/1h포함'],
#           ['전부/1h제외'], ['전부/1h포함']]
path_1 = [['전부/1h포함']]

# feature_0323_5 = ['장입중량총합', '장입최대중량', '장입소재개수', '쉰시간',
#                   '시작온도', '종료온도', '시간(총)', '강종_ALLOY', '강종_CARBON', '강종_SUS']
# feature_0323_6 = ['장입중량총합', '장입최대중량', '장입소재개수',
#                   '시작온도', '종료온도', '시간(총)', '강종_ALLOY', '강종_CARBON', '강종_SUS']
# feature_0323_6_2 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '시간(총)',
#                     '강종_ALLOY', '강종_CARBON', '강종_SUS']
feature_0323_5 = ['장입중량총합', '장입최대중량', '장입소재개수', '쉰시간',
                  '시작온도', '종료온도', '시간(총)']
feature_0323_6 = ['장입중량총합', '장입최대중량', '장입소재개수',
                  '시작온도', '종료온도', '시간(총)']
feature_0323_7 = ['장입중량총합', '장입최대중량', '장입소재개수',
                  '시작온도', '종료온도']
feature_0323_6_2 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '시간(총)']
feature_list_0323_5 = [['에너지', feature_0323_5, None, '쉰시간포함']]
feature_list_0323_6 = [['에너지', feature_0323_6, None, '쉰시간제외']]
feature_list_0323_6_2 = [['에너지', feature_0323_6_2, None, '쉰시간제외']]
feature_list_0323_6_3 = [['에너지', feature_0323_6, None, 'All']]
feature_list_0323_6_4 = [['시간(총)', feature_0323_7, None, 'All']]
feature_list_0325_3_1 = [feature_list_0323_5, feature_list_0323_6]    # Energy Increasing Part
feature_list_0325_3_2 = [feature_list_0323_6_2]    # Energy Holding Part
feature_list_0325_3_3 = [feature_list_0323_6_3]    # All features
feature_list_0325_3_4 = [feature_list_0323_6_4]    # Time

feature_0323_time_only = ['시간(총)']
feature_list_0323_time_only = [['에너지', feature_0323_time_only, None, '시간(총)_only']]
feature_list_0325_time_only = [feature_list_0323_time_only]    # Using time only

# feature_0323_7 = ['장입중량총합', '장입최대중량', '장입소재개수', '쉰시간',
#                   '시작온도', '종료온도', '강종_ALLOY', '강종_CARBON', '강종_SUS']
# feature_0323_8 = ['장입중량총합', '장입최대중량', '장입소재개수',
#                   '시작온도', '종료온도', '강종_ALLOY', '강종_CARBON', '강종_SUS']
feature_0323_7 = ['장입중량총합', '장입최대중량', '장입소재개수', '쉰시간',
                  '시작온도', '종료온도']
feature_0323_8 = ['장입중량총합', '장입최대중량', '장입소재개수',
                  '시작온도', '종료온도']
feature_list_0323_7 = [['시간(총)', feature_0323_7, None, '쉰시간포함']]
feature_list_0323_8 = [['시간(총)', feature_0323_8, None, '쉰시간제외']]
feature_list_0325_4 = [feature_list_0323_7, feature_list_0323_8]    # Minimum Required Time

f2 = [['[1]', '에너지', '에너지'], ['[1]', '시간(총)', '시간'],
      ['[2, 3]', '에너지', '에너지'], ['[2, 3]', '시간(총)', '시간'],
      ['[4, 5, 6]', '에너지', '에너지'], ['[4, 5, 6]', '시간(총)', '시간'],
      ['[2, 3, 4, 5, 6]', '에너지', '에너지'], ['[2, 3, 4, 5, 6]', '시간(0제외)', '시간'],
      ['[17, 18, 19, 20]', '에너지', '에너지'], ['[17, 18, 19, 20]', '시간(0제외)', '시간']]

feature_0323_1 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '쉰시간']
feature_0323_2 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도']
feature_list_0323_1 = [['에너지', feature_0323_1, None, '쉰시간포함'],
                       ['시간(총)', feature_0323_1, None, '쉰시간포함'],
                       ['시간(0제외)', feature_0323_1, None, '쉰시간포함']]
feature_list_0323_2 = [['에너지', feature_0323_2, None, '쉰시간제외'],
                       ['시간(총)', feature_0323_2, None, '쉰시간제외'],
                       ['시간(0제외)', feature_0323_2, None, '쉰시간제외']]
feature_list_0325 = [feature_list_0323_1, feature_list_0323_2]

fl_0331 = [[[['에너지', '시간(총)', '시간(0제외)'], feature_0323_1, None, '쉰시간포함', '에너지']]]

f3 = [['[1]', fl_0331],
     ['[2, 3]', fl_0331],
     ['[4, 5, 6]', fl_0331],
     ['[2, 3, 4, 5, 6]', fl_0331],
     ['[17, 18, 19, 20]', fl_0331]]

f4 = [['4_filtered', fl_0331],
      ['5_filtered', fl_0331],
      ['6_filtered', fl_0331]]

feature_0323_3 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '쉰시간', '종료온도']
feature_0323_4 = ['장입중량총합', '장입최대중량', '장입소재개수', '시작온도', '종료온도']
feature_list_0323_3 = [['에너지', feature_0323_3, None, '쉰시간포함'],
                       ['시간(총)', feature_0323_3, None, '쉰시간포함'],
                       ['시간(0제외)', feature_0323_3, None, '쉰시간포함']]
feature_list_0323_4 = [['에너지', feature_0323_4, None, '쉰시간제외'],
                       ['시간(총)', feature_0323_4, None, '쉰시간제외'],
                       ['시간(0제외)', feature_0323_4, None, '쉰시간제외']]
feature_list_0325_2 = [feature_list_0323_3, feature_list_0323_4]
