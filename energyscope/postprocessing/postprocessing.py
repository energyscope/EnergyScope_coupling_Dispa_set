import logging

import pandas as pd
import numpy as np
from pathlib import Path

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
