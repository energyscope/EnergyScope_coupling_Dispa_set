# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević
"""


import os
import pandas as pd
import energyscope as es


if __name__ == '__main__':

    config = {'run_ES': False,
              'import_reserves': '',
              'importing': True,
              'printing': False,
              'printing_td': False,
              'GWP_limit': 20000,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'data_folders':  ['..\\Data\\User_data', '..\\Data\\Developer_data'],
              'ES_path': '..\\STEP_2_Energy_Model',
              'ES_output_dir': '..\\STEP_2_Energy_Model\output',
              'step1_output': '..\\STEP_1_TD_selection\\TD_of_days.out',
              'all_data': pd.DataFrame(),
              'Working_directory': os.getcwd()}

   # # Reading the data
   #  config['all_data'] = es.run_ES(config)
   #  # No electricity imports
   #  config['all_data'][1].loc['ELECTRICITY', 'avail'] = 0
   #  # Printing and running
   #  config['importing'] = False
   #  config['printing'] = True
   #  config['printing_td'] = True
   #  config['run_ES'] = True
   #  config['all_data'] = es.run_ES(config)

    # config['data_folders'] = ['..\\Data\\User_data', '..\\Data\\Developer_data']
    # compute the actual average annual emission factors for each resource
    GWP_op = es.compute_gwp_op(config['data_folders'], config['ES_path'])
    GWP_op.to_csv('..\\STEP_2_Energy_Model\output\GWP_op.txt', sep='\t')
