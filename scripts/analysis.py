from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import energyscope as es
import pickle

if __name__ == '__main__':

    case_study = 'GwpLimit=40000_ElecImport=0'
    n_iter = 6
    plot = True
    hourly_plot = True
    show_plots = True
    save_plots = True

    path = Path(__file__).parents[1]
    user_data = path/'Data'/'User_data'
    developer_data = path/'Data'/'Developer_data'
    es_path = path/'energyscope'/'STEP_2_Energy_Model'
    step1_output = path/'energyscope'/'STEP_1_TD_selection'/'TD_of_days.out'
    # specify the configuration
    config = {'case_study': case_study, # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'comment': 'This is a test of versionning',
              'run_ES': False,
              'import_reserves': '',
              'importing': True,
              'printing': False,
              'printing_td': False,
              'GWP_limit': 12500,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'import_capacity': 9.72,  # [GW] Electrical interconnections with neighbouring countries
              'data_folders':  [user_data, developer_data],  # Folders containing the csv data files
              'ES_path':  es_path,  # Path to the energy model (.mod and .run files)
              'step1_output': step1_output, # Output of the step 1 selection of typical days
              'all_data': dict()}

    cs_dir = path/'case_studies'

    # reading the input data
    config['all_data'] = es.import_data([user_data, developer_data])

    # reading the outputs
    iters = np.arange(n_iter-1,-1,-1)
    results_es = dict()
    inputs = dict()
    results = dict()
    results[0] = dict()
    inputs[0] = dict()
    for i in range(n_iter):
        # read ES results
        results_es[i] = es.read_outputs(config['case_study']+'_loop_'+str(i), hourly_data=True)
        # read DS results
        with open(cs_dir / (config['case_study'] + '_loop_' + str(i)) / 'output' / 'DS_Results.p','rb') as file:
            try:
                inputs[i] = pickle.load(file)
                results[i] = pickle.load(file)
            except EOFError:
                pass

    # extracting the yearly info for each loop
    eff_tech = config['all_data']['Layers_in_out'].drop(config['all_data']['Resources'].index)
    elec_tech = list(eff_tech.loc[config['all_data']['Layers_in_out'].loc[:,'ELECTRICITY']>0.1,:].index)
    elec_cons_tech = list(eff_tech.loc[config['all_data']['Layers_in_out'].loc[:,'ELECTRICITY']<-0.1,:].index)
    sto_tech = list(config['all_data']['Storage_characteristics'].index)
    resources_names = list(config['all_data']['Resources'].index)

    assets_elec = pd.DataFrame()
    assets_cons_elec = pd.DataFrame()
    assets_sto = pd.DataFrame()
    resources = pd.DataFrame()
    res_max = pd.DataFrame()
    cp = pd.DataFrame()

    for i in range(n_iter):
        assets_elec[i] = results_es[i]['assets'].loc[elec_tech,'f']
        cp[i] = results_es[i]['assets'].loc[:, 'c_p']
        assets_cons_elec[i] = results_es[i]['assets'].loc[elec_cons_tech,'f']
        assets_sto[i] = results_es[i]['assets'].loc[sto_tech,'f']
        resources[i] = results_es[i]['resources_breakdown'].loc[:,'Used']
        res_max[i] = results_es[i]['resources_breakdown'].loc[:,'Potential']

    cp.to_csv(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' /'cp.csv')

    iter_to_plot = 3
    # exploring storage level for last loop
    sto_ts = results_es[(iter_to_plot)]['energy_stored']
    sto_lvl = sto_ts.loc[:,sto_tech]
    sto_lvl = sto_lvl.loc[:,sto_lvl.sum()>0.1]

    # looking at elec layer of last run
    layer_elec = results_es[iter_to_plot]['layer_ELECTRICITY'].dropna(axis=1)
    elec_prod = layer_elec.loc[:,layer_elec.sum(axis=0)>0]
    elec_cons = layer_elec.loc[:,layer_elec.sum(axis=0)<0]
    elec_peak = elec_prod.sum(axis=1).max()
    peak_cons = elec_cons.min(axis=0).sort_values()
    peak_prod = elec_prod.max(axis=0).sort_values(ascending=False)
    peak_prod.to_csv(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' /'peak_prod.csv')
    peak_cons.to_csv(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' /'peak_cons.csv')
    layer_reserve_elec = results_es[iter_to_plot]['layer_reserve_ELECTRICITY'].dropna(axis=1)


    # DS outputs
    LL = pd.DataFrame()
    Curtailment = pd.DataFrame()
    for i in np.arange(1,n_iter):
        LL = pd.concat([LL, results[i]['OutputShedLoad']], axis=1)
        Curtailment = pd.concat([Curtailment, results[i]['OutputCurtailedPower']], axis=1)
    LL.columns = np.arange(1,n_iter)
    Curtailment.columns = np.arange(1,n_iter)
    ENS_max = LL.max()

    # %% #######
    # Plotting #
    ############
    if plot:
        # plotting elec assets
        fig, ax = plt.subplots(figsize=(5, 3))
        assets_elec.loc[assets_elec.sum(axis=1)>0.1,iters].rename(index=lambda x: x.capitalize().replace("_"," "))\
            .sort_values(by=(n_iter-1)).plot(kind='barh', ax=ax)
        ax.set_title('Installed capacity of electricity assets')
        ax.legend(loc='lower right')
        ax.set_xlabel('Capacity [GW]')
        fig.tight_layout()
        fig.show()

        # plotting resources used all and with a zoom
        fig, ax = plt.subplots(figsize=(5, 3))
        (resources.loc[resources.sum(axis=1)>1.0,iters]/1000).rename(index=lambda x: x.capitalize().replace("_"," "))\
            .sort_values(by=(n_iter-1)).plot(kind='barh', ax=ax)
        ax.set_title('Primary resources')
        ax.legend(loc='lower right')
        ax.set_xlabel('Primary resources [TWh]')
        ax.set_ylabel('')
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 3))
        (resources.loc[(resources.sum(axis=1)>1.0) & (resources.sum(axis=1)<100000),iters]/1000).rename(index=lambda x: x.capitalize().replace("_"," "))\
            .sort_values(by=(n_iter-1)).plot(kind='barh', ax=ax)
        ax.set_title('Primary resources (zoom<100TWh)')
        ax.legend(loc='lower right')
        ax.set_xlabel('Primary resources [TWh]')
        ax.set_ylabel('')
        fig.tight_layout()
        fig.show()

        # plotting storage assets into 2 parts
        fig, ax = plt.subplots(figsize=(5, 3))
        assets_sto.loc[assets_sto.sum(axis=1) > 1000, iters].rename(index=lambda x: x.capitalize().replace("_", " ")) \
            .sort_values(by=(n_iter - 1)).plot(kind='barh', ax=ax)
        ax.set_title('Installed capacity of storage assets (>1000 GWh)')
        ax.legend(loc='lower right')
        ax.set_xlabel('Capacity [GWh]')
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 3))
        assets_sto.loc[(assets_sto.sum(axis=1) > 1.0) & (assets_sto.sum(axis=1) < 1000), iters] \
            .rename(index=lambda x: x.capitalize().replace("_", " ")).sort_values(by=(n_iter - 1)) \
            .plot(kind='barh', ax=ax)
        ax.set_title('Installed capacity of storage assets (<1000 GWh)')
        ax.legend(loc='lower right')
        ax.set_xlabel('Capacity [GWh]')
        fig.tight_layout()
        fig.show()

        #plotting storage level
        fig, ax = plt.subplots(figsize=(5, 3))
        (sto_lvl.loc[:,(sto_lvl.max()>1000)]).rename(columns=lambda x: x.capitalize().replace("_"," ")).plot(kind='line', ax=ax)
        ax.set_title('Storage level (max>1000 GWh)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 3))
        (sto_lvl.loc[:,(sto_lvl.max()>50) & (sto_lvl.max()<1000)]).rename(columns=lambda x: x.capitalize().replace("_"," ")).plot(kind='line', ax=ax)
        ax.set_title('Storage level (50<max<1000 GWh)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 3))
        (sto_lvl.loc[:,(sto_lvl.max()>10) & (sto_lvl.max()<50)]).rename(columns=lambda x: x.capitalize().replace("_"," ")).plot(kind='line', ax=ax)
        ax.set_title('Storage level (10<max<50)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(figsize=(5, 3))
        (sto_lvl.loc[:,(sto_lvl.max()>0.2) & (sto_lvl.max()<7)]).rename(columns=lambda x: x.capitalize().replace("_"," ")).plot(kind='line', ax=ax)
        ax.set_title('Storage level (max<10)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        fig.show()

        # colors = {'ELECTRICITY', 'ELEC_EXPORT', 'CCGT', 'PV', 'WIND_ONSHORE',
        #    'WIND_OFFSHORE', 'HYDRO_RIVER', 'IND_COGEN_GAS', 'IND_COGEN_WOOD',
        #    'IND_COGEN_WASTE', 'IND_DIRECT_ELEC', 'DHN_HP_ELEC', 'DHN_COGEN_GAS',
        #    'DHN_COGEN_WOOD', 'DHN_COGEN_WASTE', 'DHN_COGEN_WET_BIOMASS',
        #    'DHN_COGEN_BIO_HYDROLYSIS', 'DEC_HP_ELEC', 'DEC_COGEN_GAS',
        #    'DEC_COGEN_OIL', 'DEC_ADVCOGEN_GAS', 'DEC_ADVCOGEN_H2',
        #    'DEC_DIRECT_ELEC', 'TRAMWAY_TROLLEY', 'TRAIN_PUB', 'CAR_PHEV',
        #    'CAR_BEV', 'TRAIN_FREIGHT', 'TRUCK_ELEC', 'HABER_BOSCH',
        #    'SYN_METHANOLATION', 'BIOMASS_TO_METHANOL', 'OIL_TO_HVC', 'GAS_TO_HVC',
        #    'BIOMASS_TO_HVC', 'H2_ELECTROLYSIS', 'BIO_HYDROLYSIS',
        #    'PYROLYSIS_TO_LFO', 'PYROLYSIS_TO_FUELS', 'ATM_CCS', 'INDUSTRY_CCS',
        #    'PHS_Pin', 'PHS_Pout', 'BATT_LI_Pin', 'BATT_LI_Pout', 'BEV_BATT_Pin',
        #    'BEV_BATT_Pout', 'PHEV_BATT_Pin', 'PHEV_BATT_Pout', 'END_USE'}
        if hourly_plot:
            # plot elec layer
            d1  = es.plot_layer_elec_td(layer_elec=layer_elec, title='Layer electricity', tds = np.arange(1, 13), reorder_elec = None, figsize = (13, 7))
            # d2  = es.plot_layer_elec_td(layer_elec=layer_elec, title='Layer electricity (4 TDs)', tds = np.array([1,5,9,12]), reorder_elec = None, figsize = (13, 7))
            d3  = es.plot_layer_elec_td(layer_elec=layer_reserve_elec, title='Layer electricity reserve', tds = np.arange(1, 13), reorder_elec = None, figsize = (13, 7))
            d4  = es.plot_layer_elec_td(layer_elec=layer_reserve_elec-layer_elec, title='Difference reserve and real operation', tds = np.arange(1, 13), reorder_elec = None, figsize = (13, 7))


        #TODO add other interesting plots -> energy stored, fourrier transform, layer (colors!!),...
