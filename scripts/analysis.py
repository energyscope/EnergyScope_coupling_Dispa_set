from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import energyscope as es
import dispaset as ds
import pickle


if __name__ == '__main__':

    case_study = 'PAC2022_scenarios/6250_ElecImport=0'
    n_iter = 4
    plot = True
    draw_sankey = True
    hourly_plot = True
    ds_plots = False
    show_plots = True
    save_plots = False

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
              'GWP_limit': 37500,  # [ktCO2-eq./year]	# Minimum GWP reduction
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
    conv = pd.DataFrame(index=['OutputOptimizationError'])

    td_data = es.generate_t_h_td(config)
    for i in range(n_iter):
        # read ES results
        results_es[i] = es.read_outputs(config['case_study']+'_loop_'+str(i), hourly_data=True,
                                        layers=['layer_ELECTRICITY', 'layer_reserve_ELECTRICITY'])

        # read DS results
        with open(cs_dir / (config['case_study'] + '_loop_' + str(i)) / 'output' / 'DS_Results.p','rb') as file:
            try:
                inputs[i] = pickle.load(file)
                results[i] = pickle.load(file)
            except EOFError:
                pass

        conv.loc[:,i] = results[i]['OutputOptimizationError'].abs().sum()
        if (results[i]['OutputOptimizationError'].abs() > results[i]['OutputOptimalityGap']).any():
            print('Another iteration required')
        else:
            print('Final convergence occurred in loop: ' + str(i) + '. Soft-linking is now complete')
            n_iter=i+1
            iters = np.arange(n_iter - 1, -1, -1)
            break

    # extracting the yearly info for each loop
    resources_names = list(config['all_data']['Resources'].index)
    eff_tech = config['all_data']['Layers_in_out'].drop(resources_names)
    elec_tech = list(eff_tech.loc[config['all_data']['Layers_in_out'].loc[:,'ELECTRICITY']>0.1,:].index)
    elec_cons_tech = list(eff_tech.loc[config['all_data']['Layers_in_out'].loc[:,'ELECTRICITY']<-0.1,:].index)
    sto_tech = list(config['all_data']['Storage_characteristics'].index)

    assets_elec = pd.DataFrame()
    assets_cons_elec = pd.DataFrame()
    assets_sto = pd.DataFrame()
    resources = pd.DataFrame()
    res_max = pd.DataFrame()
    cp = pd.DataFrame()
    total_cost = pd.DataFrame(columns=results_es[0]['cost_breakdown'].columns)

    for i in range(n_iter):
        assets_elec.loc[:,i] = es.get_assets_l('ELECTRICITY',eff_tech, results_es[i]['assets'],treshold=0.05).loc[:,'f'] #results_es[i]['assets'].loc[elec_tech,'f']*config['all_data']['Layers_in_out'].loc[elec_tech,'ELECTRICITY']
        cp[i] = results_es[i]['assets'].loc[:, 'c_p']
        assets_cons_elec[i] = results_es[i]['assets'].loc[elec_cons_tech,'f']
        assets_sto[i] = results_es[i]['assets'].loc[sto_tech,'f']
        resources[i] = results_es[i]['resources_breakdown'].loc[:,'Used']
        res_max[i] = results_es[i]['resources_breakdown'].loc[:,'Potential']
        total_cost.loc[i,:] = results_es[i]['cost_breakdown'].sum(axis=0)
    total_cost.loc[:,'Total cost'] = total_cost.sum(axis=1)

    cp.to_csv(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' /'cp.csv')

    iter_to_plot = n_iter-1
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


    # # DS outputs
    LL = pd.DataFrame()
    Curtailment = pd.DataFrame()
    for i in np.arange(0,n_iter):
        LL = pd.concat([LL, results[i]['OutputShedLoad']], axis=1)
        Curtailment = pd.concat([Curtailment, results[i]['OutputCurtailedPower']], axis=1)
    LL.columns = np.arange(0,n_iter)
    Curtailment.columns = np.arange(0,n_iter)
    ENS_max = LL.max()

    # Compute curtailment in DS and ES and share renewables in elec and primary
    yr_curt_ds = Curtailment.sum() / 1000
    re_assets = ['PV', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO_RIVER']
    yr_elec_es = pd.DataFrame(index=results_es[i]['year_balance'].index)
    for i in range(n_iter):
        yr_elec_es[i] = results_es[i]['year_balance'].loc[:,'ELECTRICITY']
    yr_elec_es.loc['END_USES_DEMAND',:] = -yr_elec_es.loc['END_USES_DEMAND',:]
    yr_elec_es = yr_elec_es.loc[yr_elec_es.abs().max(axis=1)>1.0,:]
    yr_elec_prod = yr_elec_es.loc[yr_elec_es.max(axis=1)>0.0,:]
    yr_elec_re_prod = yr_elec_es.loc[re_assets,:]
    yr_elec_cons = yr_elec_es.loc[yr_elec_es.max(axis=1)<0.0,:]
    total_prod_elec = yr_elec_prod.sum()
    total_prod_elec_re = yr_elec_re_prod.sum()
    share_re_elec = total_prod_elec_re/total_prod_elec
    share_re_primary = resources.loc[config['all_data']['Resources'].loc[:,'Category']=='Renewable',:].sum()/resources.sum()
    cp_max_re = config['all_data']['Time_series'].loc[:,['PV', 'Wind_onshore', 'Wind_offshore', 'Hydro_river']].sum()/8760
    cp_max_re.index = re_assets
    curt_es = -((cp.loc[re_assets,:].sub(cp_max_re.loc[re_assets], axis=0))*assets_elec).sum()*8760


    yr_bal_diff = results_es[n_iter-1]['year_balance']-results_es[0]['year_balance']

    # %% #######
    # Plotting #
    ############
    if plot:
        # Update font sizes in plots
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # List to rename the iteration
        iter_names = ['No reserve', 'Initialization'] + ['Iteration ' + str(i) for i in np.arange(1, n_iter - 1, 1)]

        # # Example to print the sankey from this script
        if draw_sankey:
            # For loop 0
            es.drawSankey(path= cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'sankey')
            # For converged solution
            es.drawSankey(path= cs_dir / (config['case_study'] + '_loop_' + str(n_iter-1)) / 'output' / 'sankey')

        # # plotting convergence TODO
        fig, ax = plt.subplots(figsize=(13, 7))
        # conv.columns = iter_names
        conv.loc['OutputOptimizationError',1:].plot(logy=True,ax=ax)
        ax.set_title('Evolution of the convergence criteria (optimization error)')
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], iter_names, loc='lower right', frameon=False)
        # ax.set_xlabel('Capacity [GW]')
        # ax.set_ylim([0, conv.max().max()])
        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'convergence.png')


        # plotting assets elec
        fig,ax = es.plot_barh(assets_elec,treshold=0.15,
                     title='Electricity assets', x_label='Capacity [GW]', legend={'labels':iter_names},
                     show_plot=show_plots)

        # plotting cp of elec assets
        assets_to_plot = assets_elec.loc[assets_elec.max(axis=1) > 0.15, iters].sort_values(by=(n_iter - 1))
        fig,ax = es.plot_barh(cp.loc[assets_to_plot.index,:],treshold=0.15,
                     title='Capacity factor of electricity assets', x_label='Capacity factor',
                     legend={'labels':iter_names}, show_plot=show_plots)


        # assets and cp on subplots
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 7))
        assets_to_plot = assets_elec.loc[assets_elec.max(axis=1) > 0.15, iters].sort_values(by=(n_iter - 1))
        assets_to_plot.rename(index=es.plotting_names).plot(kind='barh', width=0.8, legend=False,
                                                            colormap='viridis', ax=axes[0])
        axes[0].set_title('Installed capacity')
        axes[0].set_xlabel('Capacity [GW]')

        cp_elec = cp.loc[assets_to_plot.index, :]
        cp_elec.loc[cp_elec.max(axis=1) > 0.01, iters].rename(index=es.plotting_names) \
            .plot(kind='barh', width=0.8, legend='reverse', colormap='viridis', ax=axes[1])
        axes[1].set_title('Capacity factor')
        axes[1].set_xlabel('Capacity factor')
        axes[1].set_xlim([0, 1])

        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles[::-1], iter_names, loc='upper right', frameon=False)

        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'capacity_cp_elec.png')

        # plotting resources used all and with a zoom
        fig, ax = es.plot_barh(resources/1000,treshold=0.001,
                     title='Primary resources', x_label='Primary resources [TWh]',
                     legend={'labels':iter_names}, show_plot=show_plots)
        fig, ax = es.plot_barh((resources.loc[(resources.sum(axis=1)>1.0) & (resources.sum(axis=1)<100000),iters]/1000), treshold=0.001,
                     title='Primary resources', x_label='Primary resources [TWh]',
                     legend={'labels': iter_names}, show_plot=show_plots)




        # plotting storage assets
        fig, ax = es.plot_barh(assets_sto.drop(index='GAS_STORAGE'), treshold=1.0,
                     title='Installed capacity of storage assets', x_label='Capacity [GWh]',
                     legend={'labels': iter_names}, show_plot=show_plots)
        # if save_plots:
        #     fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'assets_sto.png')


        #plotting storage level
        fig, ax = plt.subplots(figsize=(13, 7))
        (sto_lvl.loc[:,(sto_lvl.max()>1000)].drop(columns=['GAS_STORAGE'])).rename(columns=es.plotting_names).plot(kind='line', ax=ax)
        ax.set_title('Storage level (max>1000 GWh)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'sto_1000GWh_lvl.png')

        fig, ax = plt.subplots(figsize=(13, 7))
        (sto_lvl.loc[:,(sto_lvl.max()>50) & (sto_lvl.max()<1000)]).rename(columns=es.plotting_names).plot(kind='line', ax=ax)
        ax.set_title('Storage level (50<max<1000 GWh)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'sto_50_lvl_1000GWh.png')

        fig, ax = plt.subplots(figsize=(13, 7))
        (sto_lvl.loc[:,(sto_lvl.max()>10) & (sto_lvl.max()<50)]).rename(columns=es.plotting_names).plot(kind='line', ax=ax)
        ax.set_title('Storage level (10<max<50)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'sto_10_lvl_50GWh.png')

        fig, ax = plt.subplots(figsize=(13, 7))
        (sto_lvl.loc[:,(sto_lvl.max()>0.2) & (sto_lvl.max()<7)]).rename(columns=es.plotting_names).plot(kind='line', ax=ax)
        ax.set_title('Storage level (max<10)')
        # ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('Hour')
        ax.set_ylabel('Storage level [GWh]')
        fig.tight_layout()
        if show_plots:
            fig.show()
        if save_plots:
            fig.savefig(cs_dir / (config['case_study'] + '_loop_' + str(0)) / 'output' / 'sto_lvl_10GWh.png')

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
            d1  = es.plot_layer_elec_td(layer_elec=layer_elec, title='Layer electricity', tds = np.arange(1, 13), reorder_elec = None, figsize = (15, 7))
            # d2  = es.plot_layer_elec_td(layer_elec=layer_elec, title='Layer electricity (4 TDs)', tds = np.array([1,5,9,12]), reorder_elec = None, figsize = (13, 7))
            d3  = es.plot_layer_elec_td(layer_elec=layer_reserve_elec, title='Layer electricity reserve', tds = np.arange(1, 13), reorder_elec = None, figsize = (15, 7))
            d4  = es.plot_layer_elec_td(layer_elec=layer_reserve_elec-layer_elec, title='Difference reserve and real operation', tds = np.arange(1, 13), reorder_elec = None, figsize = (15, 7))
            # TODO do the same for heating layer, add possibility to savefig?
            # layer_elec_yr = es.from_td_to_year(layer_elec, t_h_td)

            results_es[0]['layer_HEAT_HIGH_T'] = es.read_layer(config['case_study'] + '_loop_' + str(0),
                                                               'layer_HEAT_HIGH_T')
            results_es[n_iter - 1]['layer_HEAT_HIGH_T'] = es.read_layer(
                config['case_study'] + '_loop_' + str(n_iter - 1), 'layer_HEAT_HIGH_T')

            d5 = es.hourly_plot(plotdata=results_es[0]['layer_HEAT_HIGH_T'], title='HT heat Loop 0', nbr_tds=12)
            d6 = es.hourly_plot(plotdata=results_es[3]['layer_HEAT_HIGH_T'], title='HT heat Loop 3', nbr_tds=12)
            d6 = es.hourly_plot(plotdata=results_es[3]['layer_HEAT_HIGH_T'] - results_es[0]['layer_HEAT_HIGH_T'],
                                title='HT heat Loop 3 - Loop 0', nbr_tds=12)

            # #example to plot the full year
            # d7 = es.hourly_plot(plotdata=es.from_td_to_year(results_es[0]['layer_HEAT_HIGH_T'], td_data['t_h_td']), xticks= np.arange(1,8761,288), title='HT heat loop 0 year')

        if ds_plots:

            rng = pd.date_range('2015-1-01', '2015-12-31', freq='H')
            # Generate country-specific plots
            ds.plot_zone(inputs[iter_to_plot], results[iter_to_plot], rng=rng, z_th='ES_DHN')

            # Bar plot with the installed capacities in all countries:
            cap = ds.plot_zone_capacities(inputs[iter_to_plot], results[iter_to_plot])

            # Bar plot with installed storage capacity
            sto = ds.plot_tech_cap(inputs[iter_to_plot])

            # Violin plot for CO2 emissions
            ds.plot_co2(inputs[iter_to_plot], results[iter_to_plot], figsize=(9, 6), width=0.9)

            # Bar plot with the energy balances in all countries:
            ds.plot_energy_zone_fuel(inputs[iter_to_plot], results[iter_to_plot], ds.get_indicators_powerplant(inputs[2], results[2]))

            # Analyse the results for each country and provide quantitative indicators:
            r = ds.get_result_analysis(inputs[iter_to_plot], results[iter_to_plot])

        #TODO add other interesting plots -> energy stored, fourrier transform, layer (colors!!),...





