from .usefull_functions import import_data, print_data, print_td_data, update_version
import pandas as pd
import os
from pathlib import Path
from subprocess import call
import shutil
import logging


def run_ES(config):
    if config['importing']:
        # Reading the data
        all_data = import_data(config['data_folders'])
    else:
        all_data = config['all_data']

    if config['printing']:
        # Printing ESTD_data.dat
        print_data(config)

    if config['printing_td']:
        # Printing ESD_12TD.dat
        if config['import_reserves'] == 'from_csv':
            # TODO: Make more elegant without so much processing and generalize country names (not only ES)
            reserves = pd.read_csv('Reserves.csv')
            reserves.index = range(1, len(reserves) + 1)
            reserves = reserves.loc[:, 'ES']
            reserves = pd.DataFrame(reserves / 1000)
            reserves.rename(columns={'ES': 'end_uses_reserve'}, inplace=True)
            print_td_data(config, nbr_td=12, end_uses_reserve=reserves)
        elif config['import_reserves'] == 'from_df':
            print_td_data(config, nbr_td=12, end_uses_reserve=config['reserves'])
        else:
            print_td_data(config, nbr_td=12)

    # run the energy system optimisation model with AMPL
    if config['run_ES']:
        two_up = Path(__file__).parents[2]

        cs = two_up / 'case_studies'

        # TODO make the case_study folder containing all runs with input, model and outputs
        shutil.copyfile(config['ES_path'] / 'ESTD_model.mod',
                        cs / config['case_study'] / 'ESTD_model.mod')
        shutil.copyfile(config['ES_path'] / 'ESTD_main.run',
                        cs / config['case_study'] / 'ESTD_main.run')
        # creating output directory
        (cs / config['case_study'] / 'output').mkdir(parents=True, exist_ok=True)
        (cs / config['case_study'] / 'output' / 'hourly_data').mkdir(parents=True, exist_ok=True)
        (cs / config['case_study'] / 'output' / 'sankey').mkdir(parents=True, exist_ok=True)
        os.chdir(cs / config['case_study'])
        # running ES
        logging.info('Running EnergyScope')
        call('ampl ESTD_main.run', shell=True)
        os.chdir(config['Working_directory'])

        update_version(config)

        logging.info('End of run')

    return all_data
