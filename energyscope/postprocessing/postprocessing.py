import logging

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ..common import *

def read_outputs(cs, hourly_data=False):
    """Reads the EnergyScope outputs in the case study (cs) specified
    Parameters
    ----------
    cs : str
    Case study to read output from

    Returns
    -------
    outputs: dict()
    Dictionnary containing the different output dataframes as pd.DataFrame
    """
    path = Path(__file__).parents[2]/'case_studies'/str(cs)/'output'

    logging.info('Reading outputs from: '+str(path))
    outputs = dict()
    outputs['assets'] = pd.read_csv(path/'assets.txt', sep="\t", skiprows=[1], index_col=0)
    outputs['assets'].columns = list(outputs['assets'].columns)[1:]+['']
    outputs['assets'].dropna(how='all', axis=1, inplace=True)
    outputs['CO2_cost'] = pd.read_csv(path/'CO2_cost.txt', sep="\t", header=None, index_col=False)
    outputs['CO2_cost'].index = ['CO2_cost']
    outputs['CO2_cost'].columns = ['CO2_cost']
    outputs['cost_breakdown'] = pd.read_csv(path/'cost_breakdown.txt', sep='\t', index_col=0)
    outputs['gwp_breakdown'] = pd.read_csv(path/'gwp_breakdown.txt', sep='\t', index_col=0)
    outputs['GWP_op'] = pd.read_csv(path/'GWP_op.txt', sep='\t', index_col=0)
    outputs['losses'] = pd.read_csv(path/'losses.txt', sep='\t', index_col=0)
    outputs['resources_breakdown'] = pd.read_csv(path/'resources_breakdown.txt', sep='\t', index_col=0)
    outputs['year_balance'] = pd.read_csv(path/'year_balance.txt', sep='\t', index_col=0).dropna(how='all', axis=1)

    if hourly_data:
        outputs['energy_stored'] = pd.read_csv(path/'hourly_data'/'energy_stored.txt', sep='\t', index_col=0)
        outputs['layer_ELECTRICITY'] = pd.read_csv(path/'hourly_data'/'layer_ELECTRICITY.txt', sep='\t', index_col=[0,1])
        outputs['layer_reserve_ELECTRICITY'] = pd.read_csv(path/'hourly_data'/'layer_reserve_ELECTRICITY.txt', sep='\t', index_col=[0,1])

        # TODO addother layers

    for o in outputs:
        outputs[o] = clean_col_and_index(outputs[o])



    return outputs

def clean_col_and_index(df):
    """Strip the leading and trailing white space in columns and index
    Parameters
    ----------
    df: pd.DataFrame()
    Dataframe to be cleaned

    Returns
    -------
    df2: pd.DataFrame()
    The stripped dataframe
    """
    df2 = df.copy()
    if df2.columns.inferred_type == 'string':
        df2.rename(columns=lambda x: x.strip(), inplace=True)
    if df2.index.inferred_type == 'string':
        df2.rename(index=lambda x: x.strip(), inplace=True)
    return df2

def scale_marginal_cost(config: dict):
    """Reads the marginal cost, scale it according to the number of days represented by each TD and prints it as 'mc_scaled.txt'
    Parameters
    ----------
    config: dict()
    Dictionnary of configuration of the EnegyScope case study

    Returns
    -------
    mc_sclaed: pd.DataFrame()
    Scaled dataframe of marginal cost

    """
    # Compute the number of days represented by each TD
    td = pd.read_csv(config['step1_output'], header=None)
    td[1] = 1 # add a column of 1 to sum
    a = td.groupby(0).sum() # count the number of occurence of each TD
    #TODO use Nbr_TD as an input
    a = a.set_index(np.arange(1,13)) # renumber from 1 to 12 (=Nbr_TD)
    b = np.repeat(a[1],24) # repeat the value for each TD 24 times (for each hour of the day)
    h = np.resize(np.arange(1, 25), 12 * 24)  # hours of each of the 12 TDs
    b = b.reset_index()  # put the TD number as a column
    b['hour'] = h
    b = b.set_index(['index', 'hour'])  # set multi-index as (TD number, hour)
    # Read marginal cost and rescale it
    cs = Path(__file__).parents[2]/'case_studies'/config['case_study']/'output'
    mc = pd.read_csv(cs/'marginal_cost.txt', sep='\t', index_col=[0,1])
    mc_scaled = mc.div(b[1],axis=0)
    mc_scaled.to_csv(cs / 'mc_scaled.txt', sep='\t')
    return mc_scaled

def hourly_plot(plotdata: pd.DataFrame, title='', xticks=None, figsize=(13,7), colors=None, show=True):
    """Cleans and plot the hourly data
    Drops the null columns and plots the hourly data in plotdata dataframe as stacked bars

    Parameters
    ----------
    plotdata: pandas.DataFrame
    Hourly dataframe with producing (>0) and consumming (<0) technologies (columns) at each hour (rows)

    xticks: numpy.ndarray
    Array of xticks for the plot

    figsize: tuple
    Figure size for the plot

    show: Boolean
    Show or not the graph

    Returns
    -------
     fig: matplotlib.figure.Figure
    Figure object of the plot

    ax: matplotlib.axes._subplots.AxesSubplot
    Ax object of the plot
    """

    # select columns with non-null prod or cons
    plotdata = plotdata.loc[:, plotdata.sum().abs() > 1.0]
    # default xticks
    if xticks is None:
        xticks = np.arange(0, plotdata.shape[0]+1, 8)

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        plotdata.plot(kind='bar', position=0, width=1.0, stacked=True, ax=ax, legend=True, xticks=xticks,
                      colormap='tab20')
    else:
        plotdata.plot(kind='bar', position=0, width=1.0, stacked=True, ax=ax, legend=True, xticks=xticks,
                      color=colors)
    ax.set_title(title)
    ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power [GW]')
    fig.tight_layout()
    fig.show()
    return fig, ax

def plot_layer_elec_td(layer_elec: pd.DataFrame, title='Layer electricity', tds = np.arange(1,13), reorder_elec=None, figsize=(13,7), xticks=None):
    """Cleans and plots the layer electricity
    Select the rows linked with specific TD, reorder the columns for the plot,
    merge the EVs columns with the batteries output, drop the null columns and plots

    Parameters
    ----------
    layer_elec: pandas.DataFrame
    Multiindex dataframe of hourly production (>0) and consumption (<0) of each technology (columns) for each hour of each typical day (rows)

    tds: numpy.ndarray
    Array containing the numbers of the TDs to plot

    reorder_elec: list
    Ordered list with all the columns names of layer_elec ordered in the way to be plotted
    (e.g. 'END_USES' should be the first consummer to be the one the closest to the x acis)

    figsize: tuple
    Size of the figure

    Returns
    -------
    Dict with:
        fig: matplotlib.figure.Figure
        Figure object of the plot

        ax: matplotlib.axes._subplots.AxesSubplot
        Ax object of the plot

        other_prods: list
        List of producing technologies with max<0.02*biggest producer (or consummer)

        other_cons: list
        List of cons technologies with max(abs)<0.02*biggest producer  (or consummer)
    """
    #TODO add colors, add printing names from Technologies dataframe
    plotdata = layer_elec.copy()
    # select specified TDs
    plotdata = plotdata.loc[(tds, slice(None)),:]

    # default reordering
    if reorder_elec is None:
        reorder_elec = elec_order_graphs
    # reorder the columns for the plot
    plotdata = plotdata[reorder_elec]
    # Grouping some tech for plot readability
        # Public mobility
    plotdata.loc[:,'TRAMWAY_TROLLEY'] = plotdata.loc[:,['TRAMWAY_TROLLEY', 'TRAIN_PUB']].sum(axis=1)
    plotdata.rename(columns={'TRAMWAY_TROLLEY': 'Public mobility'}, inplace=True)
    plotdata.drop(columns=['TRAIN_PUB'], inplace=True)
        # Freight mobility
    plotdata.loc[:,'TRAIN_FREIGHT'] = plotdata.loc[:,['TRAIN_FREIGHT', 'TRUCK_ELEC']].sum(axis=1)
    plotdata.rename(columns={'TRAIN_FREIGHT': 'Freight'}, inplace=True)
    plotdata.drop(columns=['TRUCK_ELEC'], inplace=True)

        # sum CAR_BEV and BEV_BATT_Pout into 1 column for easier reading of the impact of BEV on the grid
    plotdata.loc[:, 'BEV_BATT_Pout'] = plotdata.loc[:, 'BEV_BATT_Pout'] + plotdata.loc[:, 'CAR_BEV']
    plotdata.drop(columns=['CAR_BEV'], inplace=True)
        # same for PHEV
    plotdata.loc[:, 'PHEV_BATT_Pout'] = plotdata.loc[:, 'PHEV_BATT_Pout'] + plotdata.loc[:, 'CAR_PHEV']
    plotdata.drop(columns=['CAR_PHEV'], inplace=True)
        # treshold to group other tech
    treshold = 0.02*plotdata.abs().max().max()
        # Other prod. -> the ones with max<treshold
    other_prods = list(plotdata.loc[:,(plotdata.max()>0.0) & (plotdata.max()<treshold)].columns)
    if other_prods:
        plotdata.loc[:,other_prods[0]] = plotdata.loc[:,other_prods].sum(axis=1)
        plotdata.rename(columns={other_prods[0]: 'Other prod.'}, inplace=True)
        plotdata.drop(columns=other_prods[1:], inplace=True)
        # Other cons. -> the ones with abs(max)<treshold
    other_cons = list(plotdata.loc[:,(plotdata.min()<0.0) & (plotdata.min()>-treshold)].columns)
    if other_cons:
        plotdata.loc[:,other_cons[0]] = plotdata.loc[:,other_cons].sum(axis=1)
        plotdata.rename(columns={other_cons[0]: 'Other cons.'}, inplace=True)
        plotdata.drop(columns=other_cons[1:], inplace=True)

    # Change names before plotting
    plotdata.rename(columns=plotting_names, inplace=True)
    plotdata.rename(columns=lambda x: rename_storage_power(x) if x.endswith('Pin') or x.endswith('Pout') else x, inplace=True)

    fig, ax = hourly_plot(plotdata=plotdata, title=title, xticks=xticks, figsize=figsize, colors=colors_elec)

    return {'fig': fig, 'ax': ax, 'other_prods': other_prods, 'other_cons': other_cons}

def rename_storage_power(s):

    l = s.rsplit(sep='_')
    name = plotting_names['_'.join(l[:-1])]
    suffix = l[-1]
    return name + ' ' + suffix

# def from_td_to_year(ts_td, t_h_td):
#     """Converts time series on TDs to yearly time series
#
#     Parameters
#     ----------
#     ts_td: pandas.DataFrame
#     Multiindex dataframe of hourly data for each hour of each TD.
#     The index should be of the form (TD_number, hour_of_the_day).
#
#     t_h_td: pandas.DataFrame
#
#
#     """
#     #TODO finish and test
#     h2_layer = pd.read_csv(ES_folder / 'case_studies' / config_es['case_study'] / 'output' / 'hourly_data' /
#                            'layer_H2.txt', delimiter='\t', index_col=[0, 1])
#     h2_layer.rename(columns=lambda x: x.strip(), inplace=True)
#     h2_layer.drop(columns=['H2_STORAGE_Pin', 'H2_STORAGE_Pout'], inplace=True)
#     # computing consumption of H2
#     h2_td = pd.DataFrame(-h2_layer[h2_layer < 0].sum(axis=1), columns=['ES_H2'])  # TODO automatise name zone assignment
#     # transforming TD time series into yearly time series
#     td_final = pd.read_csv(config_es['step1_output'], header=None)
#     TD_DF = dst.process_TD(td_final)
#     h2_ts = TD_DF.loc[:, ['TD', 'hour']]
#     td_h = t_h_td.loc[:,['TD_number','H_of_D']]
#     ts_yr = td_h.merge(ts_td, left_on=['TD_number','H_of_D'], right_index=True).sort_index()
#     h2_ts.drop(columns=['TD', 'hour'], inplace=True)
#     return