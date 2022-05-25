# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević
"""

import os
import pandas as pd
from pathlib import Path
import energyscope as es

if __name__ == '__main__':
    # define path
    path = Path(__file__).parents[1]
    year = 2035
    data_folder = path / 'Data' / str(year)
    es_path = path / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
    # specify the configuration
    config = {'case_study': 'test',
              # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'comment': 'adding years',
              'run_ES': False,
              'import_reserves': '',
              'importing': True,
              'printing': False,
              'printing_td': False,
              'GWP_limit': 12500,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'data_folder': data_folder,  # Folders containing the csv data files
              'ES_path': es_path,  # Path to the energy model (.mod and .run files)
              'step1_output': step1_output,  # Output of the step 1 selection of typical days
              'all_data': dict(),
              'Working_directory': os.getcwd(),
              'user_defined': dict()}

    # Reading the data
    config['all_data'] = es.run_ES(config)
    # No electricity imports
    config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 0
    config['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    # No CCGT_AMMONIA
    config['all_data']['Technologies'].loc['CCGT_AMMONIA', 'f_max'] = 0

    # Printing and running
    config['importing'] = False
    config['printing'] = True
    config['printing_td'] = True
    config['run_ES'] = True
    config['all_data'] = es.run_ES(config)

    # # Example to print the sankey from this script
    sankey_path = path / 'case_studies' / config['case_study'] / 'output' / 'sankey'
    es.drawSankey(path=sankey_path)

    # compute the actual average annual emission factors for each resource
    GWP_op = es.compute_gwp_op(config['data_folder'], path / 'case_studies' / config['case_study'])
    GWP_op.to_csv(path / 'case_studies' / config['case_study'] / 'output' / 'GWP_op.txt', sep='\t')

    # compute scaled marginal cost
    # mc_scaled = es.scale_marginal_cost(config)
