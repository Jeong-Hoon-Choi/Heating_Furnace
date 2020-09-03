from module import *
from main_hf import *

if __name__ == '__main__':
    # work_start(view=False)
    # work_press()
    # work_set()
    # make_heat()
    # furnace_clustering()
    # HF_heating_module()

    model = 'energy-increasing'  # energy-increasing, energy-holding, time

    # make_heat_or_hold(model=model)
    # furnace_clustering2(model=model)
    # HF_learning(model=model)
    wrapper_feature_selection(model=model)

    # HF_learning_result_check(model=model)
