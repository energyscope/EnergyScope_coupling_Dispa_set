from .usefull_functions import import_data,print_data, print_td_data, make_dir
import pandas as pd
import os



def run_ES(config):
    
    if config['importing'] :
        # Reading the data
        (Eud, Resources, Technologies, End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in,
         Storage_eff_out, Time_Series) = import_data(config['data_folders'])
    else:
        (Eud, Resources, Technologies, End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in,
         Storage_eff_out, Time_Series) = config['all_data'] 


    all_df = (Eud, Resources, Technologies, End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in,
              Storage_eff_out, Time_Series)
    
    if config['printing']:
        # Printing ESTD_data.dat
        print_data(all_df, config['ES_path'], config['GWP_limit'])
    
    if config['printing_td']:
        # Printing ESD_12TD.dat
        if config['import_reserves']=='from_csv':
            # TODO: Make more elegant without so much processing and generalize country names (not only ES)
            reserves = pd.read_csv('Reserves.csv')
            reserves.index = range(1, len(reserves) + 1)
            reserves = reserves.loc[:, 'ES']
            reserves = pd.DataFrame(reserves / 1000)
            reserves.rename(columns={'ES': 'end_uses_reserve'}, inplace=True)
            print_td_data(Time_Series, config['ES_path'], config['step1_output'], end_uses_reserve=reserves)
        elif config['import_reserves']=='from_df':
            print_td_data(Time_Series, config['ES_path'], config['step1_output'], end_uses_reserve=config['reserves'])
        else:
            print_td_data(Time_Series, config['ES_path'], config['step1_output'])
    
    # run the energy system optimisation model with AMPL
    if config['run_ES']:
        #os.chdir(config['ES_path'])
        make_dir(config['ES_output_dir'])
        make_dir(config['ES_output_dir']+'/hourly_data')
        make_dir(config['ES_output_dir']+'/sankey')
        os.chdir(config['ES_output_dir'])
        os.system('cmd /c "ampl ../ESTD_main.run"')
        os.chdir(config['Working_directory'])

    return all_df