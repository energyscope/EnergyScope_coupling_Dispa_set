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
    config = {'case_study': 'test_curt_perc_cap',
              # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'comment': 'test curtailment cap',
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
    # Unlimited wind
    config['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = 1e15

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

    # reading td related information to transform from td data to year data
    td = es.generate_t_h_td(config)

    # reading curtailment output and compute total curtailment over the year
    curt_td = pd.read_csv(path / 'case_studies' / config['case_study'] / 'output' / 'hourly_data' / 'curtailment.txt', sep='\t', index_col=[0,1])
    curt_yr = es.from_td_to_year(curt_td, td['t_h_td'])
    curt_tot = curt_yr.sum()

    # reading outputs
    outputs = es.read_outputs(cs=config['case_study'], hourly_data=True, layers=['layer_ELECTRICITY', 'layer_HEAT_LOW_T_DHN'])

    fig, ax = es.hourly_plot(plotdata=outputs['layer_ELECTRICITY'])
    fig, ax = es.plot_layer_elec_td(layer_elec= outputs['layer_ELECTRICITY'])
    elec_year = es.from_td_to_year(outputs['layer_ELECTRICITY'], td['t_h_td'])
    # fig, ax = es.hourly_plot(plotdata=es.from_td_to_year(outputs['layer_ELECTRICITY'], td['t_h_td']))
    outputs['layer_ELECTRICITY'].dropna(axis=1, how='all', inplace=True)


