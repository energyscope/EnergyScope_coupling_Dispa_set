# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:26:29 2020

Contains functions to read data in csv files and print it with AMPL syntax in ESTD_data.dat
Also contains functions to analyse input data

@author: Paolo Thiran
"""
import logging

import numpy as np
import pandas as pd
import csv
import os
import shutil
from subprocess import call
import json
import git
from datetime import datetime

from pathlib import Path


# Useful functions for printing in AMPL syntax #
def ampl_syntax(df, comment):
    # adds ampl syntax to df
    df2 = df.copy()
    df2.rename(columns={df2.columns[df2.shape[1] - 1]: str(df2.columns[df2.shape[1] - 1]) + ' ' + ':= ' + comment},
               inplace=True)
    return df2


def print_set(my_set, name, out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['set ' + name + ' := \t' + '\t'.join(my_set) + ';'])


def print_df(name, df, out_path):
    df.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=name, quoting=csv.QUOTE_NONE)

    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([';'])


def newline(out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([''])


def print_param(name, param, comment, out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        if comment == '':
            writer.writerow(['param ' + str(name) + ' := ' + str(param) + ';'])
        else:
            writer.writerow(['param ' + str(name) + ' := ' + str(param) + '; # ' + str(comment)])

def print_json(my_sets, file):  # printing the dictionary containing all the sets into directory/sets.json
    with open(file, 'w') as fp:
        json.dump(my_sets, fp, indent=4, sort_keys=True)
    return

def read_json(file):
    # reading the saved dictionary containing all the sets from directory/sets.json
    with open(file, 'r') as fp:
        data = json.load(fp)
    return data


# Function to import the data from the CSV data files #
def import_data(import_folders):
    logging.info('Importing data files')
    # Reading CSV #
    # Reading User CSV to build dataframes
    eud = pd.read_csv(import_folders[0]/'Demand.csv', sep=';', index_col=2, header=0)
    resources = pd.read_csv(import_folders[0]/'Resources.csv', sep=';', index_col=2, header=2)
    technologies = pd.read_csv(import_folders[0]/'Technologies.csv', sep=';', index_col=3, header=0, skiprows=[1])

    # Reading Developer CSV to build dataframes
    end_uses_categories = pd.read_csv(import_folders[1]/'END_USES_CATEGORIES.csv', sep=';')
    layers_in_out = pd.read_csv(import_folders[1]/'Layers_in_out.csv', sep=';', index_col=0)
    storage_characteristics = pd.read_csv(import_folders[1]/'Storage_characteristics.csv', sep=';', index_col=0)
    storage_eff_in = pd.read_csv(import_folders[1]/'Storage_eff_in.csv', sep=';', index_col=0)
    storage_eff_out = pd.read_csv(import_folders[1]/'Storage_eff_out.csv', sep=';', index_col=0)
    time_series = pd.read_csv(import_folders[1]/'Time_series.csv', sep=';', header=0, index_col=0)

    # Pre-processing #
    resources.drop(columns=['Comment'], inplace=True)
    resources.dropna(axis=0, how='any', inplace=True)
    technologies.drop(columns=['Comment'], inplace=True)
    technologies.dropna(axis=0, how='any', inplace=True)
    # cleaning indices and columns

    all_df = {'Demand': eud, 'Resources': resources, 'Technologies': technologies,
              'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out,
              'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in,
              'Storage_eff_out': storage_eff_out, 'Time_series': time_series}

    for key in all_df:
        if type(all_df[key].index[0]) == str:
            all_df[key].index = all_df[key].index.str.strip()
        if type(all_df[key].columns[0]) == str:
            all_df[key].columns = all_df[key].columns.str.strip()

    return all_df


# Function to print the ESTD_data.dat file #
def print_data(config):
    # two_up = os.path.dirname(os.path.dirname(__file__))
    two_up = Path(__file__).parents[2]

    cs = two_up / 'case_studies'
    cs.mkdir(parents=True, exist_ok=True)

    (cs / config['case_study']).mkdir(parents=True, exist_ok=True)

    data = config['all_data']

    eud = data['Demand']
    resources = data['Resources']
    technologies = data['Technologies']
    end_uses_categories = data['End_uses_categories']
    layers_in_out = data['Layers_in_out']
    storage_characteristics = data['Storage_characteristics']
    storage_eff_in = data['Storage_eff_in']
    storage_eff_out = data['Storage_eff_out']
    time_series = data['Time_series']

    logging.info('Printing ESTD_data.dat')

    # Prints the data into .dat file (out_path) with the right syntax for AMPL
    out_path = cs / config['case_study'] / 'ESTD_data.dat'
    # config['ES_path'] + '/ESTD_data.dat'
    gwp_limit = config['GWP_limit']
    import_capacity = config['import_capacity']  # [GW] Maximum power of electrical interconnections

    # Pre-processing df #

    # pre-processing resources
    resources_simple = resources.loc[:, ['avail', 'gwp_op', 'c_op']]
    resources_simple.index.name = 'param :'
    resources_simple = resources_simple.astype('float')
    # pre-processing eud
    eud_simple = eud.drop(columns=['Category', 'Subcategory', 'Units'])
    eud_simple.index.name = 'param end_uses_demand_year:'
    eud_simple = eud_simple.astype('float')
    # pre_processing technologies
    technologies_simple = technologies.drop(columns=['Category', 'Subcategory', 'Technologies name'])
    technologies_simple.index.name = 'param:'
    technologies_simple = technologies_simple.astype('float')

    # Developer defined parameters #
    # Economical inputs

    # TODO put everything into a config file so that we can modify it from outside the function
    i_rate = 0.015  # [-]
    # Political inputs
    re_share_primary = 0  # [-] Minimum RE share in primary consumption
    solar_area = 250  # [km^2]
    power_density_pv = 0.2367  # PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2
    power_density_solar_thermal = 0.2857  # Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2

    # Technologies shares
    share_mobility_public_min = 0.199
    share_mobility_public_max = 0.5
    share_freight_train_min = 0.109
    share_freight_train_max = 0.25
    share_freight_road_min = 0
    share_freight_road_max = 1
    share_freight_boat_min = 0.156
    share_freight_boat_max = 0.3
    share_heat_dhn_min = 0.02
    share_heat_dhn_max = 0.37

    share_ned = pd.DataFrame([0.779, 0.029, 0.192], index=['HVC', 'METHANOL', 'AMMONIA'], columns=['share_ned'])

    # Electric vehicles :
    # km-pass/h/veh. : Gives the equivalence between capacity and number of vehicles.
    # ev_batt, size [GWh]: Size of batteries per car per technology of EV
    evs = pd.DataFrame({'EVs_BATT': ['PHEV_BATT', 'BEV_BATT'], 'vehicule_capacity': [5.04E+01, 5.04E+01],
                        'batt_per_car': [4.40, 24.0]}, index=['CAR_PHEV', 'CAR_BEV'])
    a = np.zeros((2, 24))
    a[0, 6] = 0.6
    a[1, 6] = 0.6
    state_of_charge_ev = pd.DataFrame(a, columns=np.arange(1, 25), index=['PHEV_BATT', 'BEV_BATT'])
    # Network
    loss_network = {'ELECTRICITY': 4.7E-02, 'HEAT_LOW_T_DHN': 5.0E-02}
    c_grid_extra = 367.8  # cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2

    # Storage daily
    STORAGE_DAILY = ['TS_DEC_HP_ELEC', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL',
                     'TS_DEC_ADVCOGEN_GAS',
                     'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL',
                     'TS_DEC_DIRECT_ELEC', 'TS_DHN_DAILY', 'BATT_LI', 'TS_HIGH_TEMP']  # TODO automatise

    # Building SETS from data #
    SECTORS = list(eud_simple.columns)
    END_USES_INPUT = list(eud_simple.index)
    END_USES_CATEGORIES = list(end_uses_categories.loc[:, 'END_USES_CATEGORIES'].unique())
    RESOURCES = list(resources_simple.index)
    RES_IMPORT_CONSTANT = ['GAS', 'GAS_RE', 'H2_RE', 'H2']  # TODO automatise
    RENEWABLE_FUELS = list(resources[resources.loc[:, 'Subcategory'] == 'Biofuel'].index)
    RE_RESOURCES = list(
        resources.loc[(resources['Category'] == 'Renewable'), :].index)
    EXPORT = list(resources.loc[resources['Category'] == 'Export', :].index)

    END_USES_TYPES_OF_CATEGORY = []
    for i in END_USES_CATEGORIES:
        li = list(end_uses_categories.loc[
                      end_uses_categories.loc[:, 'END_USES_CATEGORIES'] == i, 'END_USES_TYPES_OF_CATEGORY'])
        END_USES_TYPES_OF_CATEGORY.append(li)

    # TECHNOLOGIES_OF_END_USES_TYPE -> # METHOD 2 (uses layer_in_out to determine the END_USES_TYPE)
    END_USES_TYPES = list(end_uses_categories.loc[:, 'END_USES_TYPES_OF_CATEGORY'])

    ALL_TECHS = list(technologies_simple.index)

    layers_in_out_tech = layers_in_out.loc[~layers_in_out.index.isin(RESOURCES), :]
    TECHNOLOGIES_OF_END_USES_TYPE = []
    for i in END_USES_TYPES:
        li = list(layers_in_out_tech.loc[layers_in_out_tech.loc[:, i] == 1, :].index)
        TECHNOLOGIES_OF_END_USES_TYPE.append(li)

    # STORAGE and INFRASTRUCTURES
    ALL_TECH_OF_EUT = [item for sublist in TECHNOLOGIES_OF_END_USES_TYPE for item in sublist]

    STORAGE_TECH = list(storage_eff_in.index)
    INFRASTRUCTURE = [item for item in ALL_TECHS if item not in STORAGE_TECH and item not in ALL_TECH_OF_EUT]

    # EVs
    EVs_BATT = list(evs.loc[:, 'EVs_BATT'])
    V2G = list(evs.index)

    # STORAGE_OF_END_USES_TYPES ->  #METHOD 2 (using storage_eff_in)
    STORAGE_OF_END_USES_TYPES_DHN = []
    STORAGE_OF_END_USES_TYPES_DEC = []
    STORAGE_OF_END_USES_TYPES_ELEC = []
    STORAGE_OF_END_USES_TYPES_HIGH_T = []

    for i in STORAGE_TECH:
        if storage_eff_in.loc[i, 'HEAT_LOW_T_DHN'] > 0:
            STORAGE_OF_END_USES_TYPES_DHN.append(i)
        elif storage_eff_in.loc[i, 'HEAT_LOW_T_DECEN'] > 0:
            STORAGE_OF_END_USES_TYPES_DEC.append(i)
        elif storage_eff_in.loc[i, 'ELECTRICITY'] > 0:
            STORAGE_OF_END_USES_TYPES_ELEC.append(i)
        elif storage_eff_in.loc[i, 'HEAT_HIGH_T'] > 0:
            STORAGE_OF_END_USES_TYPES_HIGH_T.append(i)

    STORAGE_OF_END_USES_TYPES_ELEC.remove('BEV_BATT')
    STORAGE_OF_END_USES_TYPES_ELEC.remove('PHEV_BATT')

    # etc. still TS_OF_DEC_TECH and EVs_BATT_OF_V2G missing... -> hard coded !

    COGEN = []
    BOILERS = []

    for i in ALL_TECH_OF_EUT:
        if (layers_in_out.loc[i, 'HEAT_HIGH_T'] == 1 or layers_in_out.loc[i, 'HEAT_LOW_T_DHN'] == 1 or
                layers_in_out.loc[i, 'HEAT_LOW_T_DECEN'] == 1):
            if layers_in_out.loc[i, 'ELECTRICITY'] > 0:
                COGEN.append(i)
            else:
                BOILERS.append(i)

    # Adding AMPL syntax #
    # creating Batt_per_Car_df for printing
    batt_per_car_df = evs[['batt_per_car']]
    vehicule_capacity_df = evs[['vehicule_capacity']]
    state_of_charge_ev = ampl_syntax(state_of_charge_ev, '')
    loss_network_df = pd.DataFrame(data=loss_network.values(), index=loss_network.keys(), columns=[' '])
    # Putting all the df in ampl syntax
    batt_per_car_df = ampl_syntax(batt_per_car_df,
                                  '# ev_batt,size [GWh]: Size of batteries per car per technology of EV')
    vehicule_capacity_df = ampl_syntax(vehicule_capacity_df, '# km-pass/h/veh. : Gives the equivalence between '
                                                             'capacity and number of vehicles.')
    eud_simple = ampl_syntax(eud_simple, '')
    share_ned = ampl_syntax(share_ned, '')
    layers_in_out = ampl_syntax(layers_in_out, '')
    technologies_simple = ampl_syntax(technologies_simple, '')
    technologies_simple[technologies_simple > 1e+14] = 'Infinity'
    resources_simple = ampl_syntax(resources_simple, '')
    resources_simple[resources_simple > 1e+14] = 'Infinity'
    storage_eff_in = ampl_syntax(storage_eff_in, '')
    storage_eff_out = ampl_syntax(storage_eff_out, '')
    storage_characteristics = ampl_syntax(storage_characteristics, '')
    loss_network_df = ampl_syntax(loss_network_df, '')

    # Printing data #
    # printing signature of data file
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['# ------------------------------------------------------------------------------'
                         '-------------------------------------------	'])
        writer.writerow(['#	EnergyScope TD is an open-source energy model suitable for country scale analysis.'
                         ' It is a simplified representation of an urban or national energy system accounting for the'
                         ' energy flows'])
        writer.writerow(
            ['#	within its boundaries. Based on a hourly resolution, it optimises the design and operation '
             'of the energy system while minimizing the cost of the system.'])
        writer.writerow(['#	'])
        writer.writerow(['#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), '
                         'Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
        writer.writerow(['#	'])
        writer.writerow(['#	'])
        writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
        writer.writerow(['#	you may not use this file except in compliance with the License.'])
        writer.writerow(['#	You may obtain a copy of the License at'])
        writer.writerow(['#	'])
        writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
        writer.writerow(['#	'])
        writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
        writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
        writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
        writer.writerow(['#	See the License for the specific language governing permissions and'])
        writer.writerow(['#	limitations under the License.'])
        writer.writerow(['#	'])
        writer.writerow(['#	Description and complete License: see LICENSE file.'])
        writer.writerow(
            ['# -------------------------------------------------------------------------------------------'
             '------------------------------	'])
        writer.writerow(['	'])
        writer.writerow(['# UNIT MEASURES:'])
        writer.writerow(['# Unless otherwise specified units are:'])
        writer.writerow(
            [
                '# Energy [GWh], Power [GW], Cost [Meuro], Time [h], Passenger transport [Mpkm], Freight Transport [Mtkm]'])
        writer.writerow(['	'])
        writer.writerow(['# References based on Supplementary material'])
        writer.writerow(['# --------------------------	'])
        writer.writerow(['# SETS not depending on TD	'])
        writer.writerow(['# --------------------------	'])
        writer.writerow(['	'])

    # printing sets
    print_set(SECTORS, 'SECTORS', out_path)
    print_set(END_USES_INPUT, 'END_USES_INPUT', out_path)
    print_set(END_USES_CATEGORIES, 'END_USES_CATEGORIES', out_path)
    print_set(RESOURCES, 'RESOURCES', out_path)
    print_set(RES_IMPORT_CONSTANT, 'RES_IMPORT_CONSTANT', out_path)
    print_set(RENEWABLE_FUELS, 'RENEWABLE_FUELS', out_path)
    print_set(RE_RESOURCES, 'RE_RESOURCES', out_path)
    print_set(EXPORT, 'EXPORT', out_path)
    newline(out_path)
    n = 0
    for j in END_USES_TYPES_OF_CATEGORY:
        print_set(j, 'END_USES_TYPES_OF_CATEGORY' + '["' + END_USES_CATEGORIES[n] + '"]', out_path)
        n += 1
    newline(out_path)
    n = 0
    for j in TECHNOLOGIES_OF_END_USES_TYPE:
        print_set(j, 'TECHNOLOGIES_OF_END_USES_TYPE' + '["' + END_USES_TYPES[n] + '"]', out_path)
        n += 1
    newline(out_path)
    print_set(STORAGE_TECH, 'STORAGE_TECH', out_path)
    print_set(INFRASTRUCTURE, 'INFRASTRUCTURE', out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Storage subsets'])
    print_set(EVs_BATT, 'EVs_BATT', out_path)
    print_set(V2G, 'V2G', out_path)
    print_set(STORAGE_DAILY, 'STORAGE_DAILY', out_path)
    newline(out_path)
    print_set(STORAGE_OF_END_USES_TYPES_DHN, 'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DHN"]', out_path)
    print_set(STORAGE_OF_END_USES_TYPES_DEC, 'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DECEN"]', out_path)
    print_set(STORAGE_OF_END_USES_TYPES_ELEC, 'STORAGE_OF_END_USES_TYPES ["ELECTRICITY"]', out_path)
    print_set(STORAGE_OF_END_USES_TYPES_HIGH_T, 'STORAGE_OF_END_USES_TYPES ["HEAT_HIGH_T"]', out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Link between storages & specific technologies	'])
    # Hardcoded
    print_set(['TS_DEC_HP_ELEC'], 'TS_OF_DEC_TECH ["DEC_HP_ELEC"]', out_path)
    print_set(['TS_DEC_DIRECT_ELEC'], 'TS_OF_DEC_TECH ["DEC_DIRECT_ELEC"]', out_path)
    print_set(['TS_DEC_THHP_GAS'], 'TS_OF_DEC_TECH ["DEC_THHP_GAS"]', out_path)
    print_set(['TS_DEC_COGEN_GAS'], 'TS_OF_DEC_TECH ["DEC_COGEN_GAS"]', out_path)
    print_set(['TS_DEC_ADVCOGEN_GAS'], 'TS_OF_DEC_TECH ["DEC_ADVCOGEN_GAS"]', out_path)
    print_set(['TS_DEC_COGEN_OIL'], 'TS_OF_DEC_TECH ["DEC_COGEN_OIL"]', out_path)
    print_set(['TS_DEC_ADVCOGEN_H2'], 'TS_OF_DEC_TECH ["DEC_ADVCOGEN_H2"]', out_path)
    print_set(['TS_DEC_BOILER_GAS'], 'TS_OF_DEC_TECH ["DEC_BOILER_GAS"]', out_path)
    print_set(['TS_DEC_BOILER_WOOD'], 'TS_OF_DEC_TECH ["DEC_BOILER_WOOD"]', out_path)
    print_set(['TS_DEC_BOILER_OIL'], 'TS_OF_DEC_TECH ["DEC_BOILER_OIL"]', out_path)
    print_set(['PHEV_BATT'], 'EVs_BATT_OF_V2G ["CAR_PHEV"]', out_path)
    print_set(['BEV_BATT'], 'EVs_BATT_OF_V2G ["CAR_BEV"]', out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Additional sets, just needed for printing results	'])
    print_set(COGEN, 'COGEN', out_path)
    print_set(BOILERS, 'BOILERS', out_path)
    newline(out_path)

    # printing parameters
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# -----------------------------'])
        writer.writerow(['# PARAMETERS NOT DEPENDING ON THE NUMBER OF TYPICAL DAYS : '])
        writer.writerow(['# -----------------------------	'])
        writer.writerow([''])
        writer.writerow(['## PARAMETERS presented in Table 2.	'])
    # printing i_rate, re_share_primary,gwp_limit,solar_area
    print_param('i_rate', i_rate, 'part [2.7.4]', out_path)
    print_param('re_share_primary', re_share_primary, 'Minimum RE share in primary consumption', out_path)
    print_param('gwp_limit', gwp_limit, 'gwp_limit [ktCO2-eq./year]: maximum GWP emissions', out_path)
    print_param('solar_area', solar_area, '', out_path)
    print_param('power_density_pv', power_density_pv, 'PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2',
                out_path)
    print_param('power_density_solar_thermal', power_density_solar_thermal,
                'Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2', out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Part [2.4]	'])
    print_df('param:', batt_per_car_df, out_path)
    newline(out_path)
    print_df('param:', vehicule_capacity_df, out_path)
    newline(out_path)
    print_df('param state_of_charge_ev :', state_of_charge_ev, out_path)
    newline(out_path)

    # printing c_grid_extra and import_capacity
    print_param('c_grid_extra', c_grid_extra,
                'cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2', out_path)
    print_param('import_capacity', import_capacity, '', out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# end_Uses_year see part [2.1]'])
    print_df('param end_uses_demand_year : ', eud_simple, out_path)
    newline(out_path)
    print_param('share_mobility_public_min', share_mobility_public_min, '', out_path)
    print_param('share_mobility_public_max', share_mobility_public_max, '', out_path)
    newline(out_path)
    print_param('share_freight_train_min', share_freight_train_min, '', out_path)
    print_param('share_freight_train_max', share_freight_train_max, '', out_path)
    newline(out_path)
    print_param('share_freight_road_min', share_freight_road_min, '', out_path)
    print_param('share_freight_road_max', share_freight_road_max, '', out_path)
    newline(out_path)
    print_param('share_freight_boat_min', share_freight_boat_min, '', out_path)
    print_param('share_freight_boat_max', share_freight_boat_max, '', out_path)
    newline(out_path)
    print_param('share_heat_dhn_min', share_heat_dhn_min, '', out_path)
    print_param('share_heat_dhn_max', share_heat_dhn_max, '', out_path)
    newline(out_path)
    print_df('param:', share_ned, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Link between layers  (data from Tables 19,21,22,23,25,29,30)'])
    print_df('param layers_in_out : ', layers_in_out, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['# Technologies data from Tables (10,19,21,22,23,25,27,28,29,30) and part [2.2.1.1] for hydro'])
    print_df('param :', technologies_simple, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# RESOURCES: part [2.5] (Table 26)'])
    print_df('param :', resources_simple, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['# Storage inlet/outlet efficiency : part [2.6] (Table 28) and part [2.2.1.1] for hydro.	'])
    print_df('param storage_eff_in :', storage_eff_in, out_path)
    newline(out_path)
    print_df('param storage_eff_out :', storage_eff_out, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# Storage characteristics : part [2.6] (Table 28) and part [2.2.1.1] for hydro.'])
    print_df('param :', storage_characteristics, out_path)
    newline(out_path)
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['# [A.6]'])
    print_df('param loss_network ', loss_network_df, out_path)

    return


# Function to print the ESTD_12TD.dat file from timeseries and STEP1 results #
def print_td_data(config, nbr_td=12, end_uses_reserve=pd.DataFrame(np.zeros((8760, 1)), columns=['end_uses_reserve'],
                                                                  index=np.arange(1, 8761, 1))):
    # two_up = os.path.dirname(os.path.dirname(__file__))
    two_up = Path(__file__).parents[2]

    cs = two_up / 'case_studies'
    cs.mkdir(parents=True, exist_ok=True)

    (cs / config['case_study']).mkdir(parents=True, exist_ok=True)

    data = config['all_data']

    time_series = data['Time_series']

    out_path = cs / config['case_study']  # config['ES_path']
    step1_out = config['step1_output']

    logging.info('Printing ESTD_' + str(nbr_td) + 'TD.dat')

    # DICTIONARIES TO TRANSLATE NAMES INTO AMPL SYNTAX #
    # for EUD timeseries
    eud_params = {'Electricity (%_elec)': 'param electricity_time_series :',
                  'Space Heating (%_sh)': 'param heating_time_series :',
                  'Passanger mobility (%_pass)': 'param mob_pass_time_series :',
                  'Freight mobility (%_freight)': 'param mob_freight_time_series :'}
    # for resources timeseries that have only 1 tech linked to it
    res_params = {'PV': 'PV', 'Wind_onshore': 'WIND_ONSHORE', 'Wind_offshore': 'WIND_OFFSHORE',
                  'Hydro_river': 'HYDRO_RIVER'}
    # for resources timeseries that have several techs linked to it
    res_mult_params = {'Solar': ['DHN_SOLAR', 'DEC_SOLAR']}

    # Merge end_uses__reserve to the other timeseries #
    time_series = time_series.merge(end_uses_reserve, left_index=True, right_index=True)

    # Redefine the output file from the out_path given #
    out_path = out_path/('ESTD_' + str(nbr_td) + 'TD.dat')

    # READING OUTPUT OF STEP1 #
    td_of_days = pd.read_csv(step1_out, names=['TD_of_days'])
    td_of_days['day'] = np.arange(1, 366, 1)  # putting the days of the year beside

    # COMPUTING NUMBER OF DAYS REPRESENTED BY EACH TD #
    sorted_td = td_of_days.groupby('TD_of_days').count()
    sorted_td.rename(columns={'day': '#days'}, inplace=True)
    sorted_td.reset_index(inplace=True)
    sorted_td.set_index(np.arange(1, nbr_td + 1), inplace=True)  # adding number of TD as index

    # BUILDING T_H_TD MATRICE #
    # generate T_H_TD
    td_and_hour_array = np.ones((24 * 365, 2))
    for i in range(365):
        td_and_hour_array[i * 24:(i + 1) * 24, 0] = np.arange(1, 25, 1)
        td_and_hour_array[i * 24:(i + 1) * 24, 1] = td_and_hour_array[i * 24:(i + 1) * 24, 1] * sorted_td[
            sorted_td['TD_of_days'] == td_of_days.loc[i, 'TD_of_days']].index.values
    t_h_td = pd.DataFrame(td_and_hour_array, index=np.arange(1, 8761, 1), columns=['H_of_D', 'TD_of_day'])
    t_h_td = t_h_td.astype('int64')
    # giving the right syntax
    t_h_td.reset_index(inplace=True)
    t_h_td.rename(columns={'index': 'H_of_Y'}, inplace=True)
    t_h_td['par_g'] = '('
    t_h_td['par_d'] = ')'
    t_h_td['comma1'] = ','
    t_h_td['comma2'] = ','
    # giving the right order to the columns
    t_h_td = t_h_td[['par_g', 'H_of_Y', 'comma1', 'H_of_D', 'comma2', 'TD_of_day', 'par_d']]

    # COMPUTING THE NORM OVER THE YEAR ##
    norm = time_series.sum(axis=0)
    norm.index.rename('Category', inplace=True)
    norm.name = 'Norm'

    # BUILDING TD TIMESERIES #
    # creating df with 2 columns : day of the year | hour in the day
    day_and_hour_array = np.ones((24 * 365, 2))
    for i in range(365):
        day_and_hour_array[i * 24:(i + 1) * 24, 0] = day_and_hour_array[i * 24:(i + 1) * 24, 0] * (i + 1)
        day_and_hour_array[i * 24:(i + 1) * 24, 1] = np.arange(1, 25, 1)
    day_and_hour = pd.DataFrame(day_and_hour_array, index=np.arange(1, 8761, 1), columns=['D_of_H', 'H_of_D'])
    day_and_hour = day_and_hour.astype('int64')
    time_series = time_series.merge(day_and_hour, left_index=True, right_index=True)

    # selecting time series of TD only
    td_ts = time_series[time_series['D_of_H'].isin(sorted_td['TD_of_days'])]

    # COMPUTING THE NORM_TD OVER THE YEAR FOR CORRECTION #
    # computing the sum of ts over each TD
    agg_td_ts = td_ts.groupby('D_of_H').sum()
    agg_td_ts.reset_index(inplace=True)
    agg_td_ts.set_index(np.arange(1, nbr_td + 1), inplace=True)
    agg_td_ts.drop(columns=['D_of_H', 'H_of_D'], inplace=True)
    # multiplicating each TD by the number of day it represents
    for c in agg_td_ts.columns:
        agg_td_ts[c] = agg_td_ts[c] * sorted_td['#days']
    # sum of new ts over the whole year
    norm_td = agg_td_ts.sum()

    # BUILDING THE DF WITH THE TS OF EACH TD FOR EACH CATEGORY #
    # pivoting TD_ts to obtain a (24,Nbr_TD*Nbr_ts*N_c)
    all_td_ts = td_ts.pivot(index='H_of_D', columns='D_of_H')

    # COMPUTE peak_sh_factor #
    max_sh_td = td_ts.loc[:, 'Space Heating (%_sh)'].max()
    max_sh_all = time_series.loc[:, 'Space Heating (%_sh)'].max()
    peak_sh_factor = max_sh_all / max_sh_td

    # PRINTING #
    # printing description of file
    with open(out_path, mode='w', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        # Comments and license
        td_writer.writerow([
            '# -------------------------------------------------------------------------------------------------------------------------	'])
        td_writer.writerow([
            '#	EnergyScope TD is an open-source energy model suitable for country scale analysis. It is a simplified representation of an urban or national energy system accounting for the energy flows'])
        td_writer.writerow([
            '#	within its boundaries. Based on a hourly resolution, it optimises the design and operation of the energy system while minimizing the cost of the system.'])
        td_writer.writerow(['#	'])
        td_writer.writerow([
            '#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
        td_writer.writerow(['#	'])
        td_writer.writerow(['#	'])
        td_writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
        td_writer.writerow(['#	you may not use this file except in compliance with the License.'])
        td_writer.writerow(['#	You may obtain a copy of the License at'])
        td_writer.writerow(['#	'])
        td_writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
        td_writer.writerow(['#	'])
        td_writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
        td_writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
        td_writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
        td_writer.writerow(['#	See the License for the specific language governing permissions and'])
        td_writer.writerow(['#	limitations under the License.'])
        td_writer.writerow(['#	'])
        td_writer.writerow(['#	Description and complete License: see LICENSE file.'])
        td_writer.writerow([
            '# -------------------------------------------------------------------------------------------------------------------------	'])
        td_writer.writerow(['	'])
        # peak_sh_factor
        td_writer.writerow(['# SETS depending on TD	'])
        td_writer.writerow(['# --------------------------	'])
        td_writer.writerow(['param peak_sh_factor	:=	' + str(peak_sh_factor)])
        td_writer.writerow([';		'])
        td_writer.writerow(['		'])

        # printing T_H_TD param
        td_writer.writerow(['#SETS [Figure 3]		'])
        td_writer.writerow(['set T_H_TD := 		'])

    t_h_td.to_csv(out_path, sep='\t', header=False, index=False, mode='a', quoting=csv.QUOTE_NONE)

    # printing interlude
    with open(out_path, mode='a', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        td_writer.writerow([';'])
        td_writer.writerow([''])
        td_writer.writerow(['# -----------------------------'])
        td_writer.writerow(['# PARAMETERS DEPENDING ON NUMBER OF TYPICAL DAYS : '])
        td_writer.writerow(['# -----------------------------'])
        td_writer.writerow([''])

    # printing EUD timeseries param
    for k in eud_params.keys():
        ts = all_td_ts[k]
        ts.columns = np.arange(1, nbr_td + 1)
        ts = ts * norm[k] / norm_td[k]
        ts.fillna(0, inplace=True)

        ts = ampl_syntax(ts, '')
        print_df(eud_params[k], ts, out_path)
        newline(out_path)

    # printing c_p_t param #
    with open(out_path, mode='a', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        td_writer.writerow(['param c_p_t:='])
        # printing c_p_t part where 1 ts => 1 tech
    for k in res_params.keys():
        ts = all_td_ts[k]
        ts.columns = np.arange(1, nbr_td + 1)
        ts = ts * norm[k] / norm_td[k]
        ts.fillna(0, inplace=True)

        ts = ampl_syntax(ts, '')
        s = '["' + res_params[k] + '",*,*]:'
        ts.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        newline(out_path)

    # printing c_p_t part where 1 ts => more then 1 tech
    for k in res_mult_params.keys():
        for j in res_mult_params[k]:
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            s = '["' + j + '",*,*]:'
            ts.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)

    # printing end_uses_reserve ts
    with open(out_path, mode='a', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        td_writer.writerow([';'])
        td_writer.writerow([''])
        td_writer.writerow(['param end_uses_reserve:='])

    ts = all_td_ts['end_uses_reserve']
    ts.columns = np.arange(1, nbr_td + 1)
    ts = ts * norm['end_uses_reserve'] / norm_td['end_uses_reserve']
    ts.fillna(0, inplace=True)
    ts = ampl_syntax(ts, '')
    print_df('["ELECTRICITY",*,*]:', ts, out_path)
    newline(out_path)

    return


def update_version(config):
    """

    Updating the version.json file into case_studies directory to add the description of this run

    """
    # path of case_studies dir
    two_up = Path(__file__).parents[2]
    cs_versions = two_up / 'case_studies/versions.json'

    # get git commit used
    repo = git.Repo(search_parent_directories=True)
    commit_name = repo.head.commit.summary

    # get current datetime
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # read versions dict
    try:
        versions = read_json(cs_versions)
    except:
        versions = dict()

    # update the key for this case_study
    keys_to_extract = ['comment', 'printing', 'printing_td', 'GWP_limit']
    versions[config['case_study']] = {key: config[key] for key in keys_to_extract}
    versions[config['case_study']]['commit_name'] = commit_name
    versions[config['case_study']]['datetime'] = now

    print_json(versions, cs_versions)
    return


# Function to compute the annual average emission factors of each resource from the outputs #
# Function to compute the annual average emission factors of each resource from the outputs #
def compute_gwp_op(import_folders, out_path):
    # import data and model outputs
    resources = pd.read_csv(import_folders[0] / 'Resources.csv', sep=';', index_col=2, header=2)
    yb = pd.read_csv((out_path / 'output') / 'year_balance.txt', sep='\t', index_col=0)

    # clean df and get useful data
    yb.rename(columns=lambda x: x.strip(), inplace=True)
    yb.rename(index=lambda x: x.strip(), inplace=True)
    resources.rename(columns=lambda x: x.strip(), inplace=True)
    resources.rename(index=lambda x: x.strip(), inplace=True)
    gwp_op_data = resources['gwp_op'].dropna()
    res_names = list(gwp_op_data.index)
    res_names_red = list(set(res_names) & set(list(yb.columns)))  # resources that are a layer
    yb2 = yb.drop(index='END_USES_DEMAND')
    tot_year = yb2.mul(yb2.gt(0)).sum()[res_names_red]

    # compute the actual resources used to produce each resource
    res_used = pd.DataFrame(0, columns=res_names_red, index=res_names)
    fuel_mapping = {'BIOETHANOL': 'GASOLINE', 'BIODIESEL': 'DIESEL', 'GAS_RE': 'GAS', 'H2_RE' : 'H2',
                    'AMMONIA_RE': 'AMMONIA', 'METHANOL_RE': 'METHANOL'}
    for r in res_names_red:
        yb_r = yb2.loc[yb2.loc[:, r] > 0, :]
        for i, j in yb_r.iterrows():
            if i in res_names:
                if i not in fuel_mapping.keys():
                    res_used.loc[i, r] = res_used.loc[i, r] + j[i]
                else:
                    res_used.loc[i, r] = res_used.loc[i, r] + j[fuel_mapping[i]]
            else:
                s = list(j[j < 0].index)[0]
                res_used.loc[s, r] = res_used.loc[s, r] - j[s]

    # differentiate the imported resources from the ones that are the mix between the imported ones and the produced ones
    gwp_op_imp = gwp_op_data.copy()
    gwp_op_imp.rename(index=lambda x: x + '_imp', inplace=True)
    gwp_op = pd.concat([gwp_op_data.copy(), gwp_op_imp])
    res_used_imp = pd.DataFrame(0, index=res_used.index, columns=res_used.columns)
    for i, j in res_used.iteritems():
        res_used_imp.loc[i, i] = j[i]
        res_used.loc[i, i] = 0
    res_used_imp.rename(index=lambda x: x + '_imp', inplace=True)
    all_res_used = pd.concat([res_used, res_used_imp])

    # compute the gwp_op of each mix through looping over the equations
    gwp_op_new = gwp_op.copy()
    conv = 100
    count = 0
    while conv > 1e-6:
        gwp_op = gwp_op_new
        gwp_op_new = pd.concat([(all_res_used.mul(gwp_op, axis=0).sum() / tot_year).fillna(0), gwp_op_imp])
        conv = (gwp_op_new - gwp_op).abs().sum()
        count += 1

    gwp_op_final = gwp_op_new[res_names_red]

    return gwp_op_final.combine_first(gwp_op_data)
