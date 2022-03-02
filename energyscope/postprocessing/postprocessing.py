import logging

import pandas as pd
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

    for o in outputs:
        outputs[o] = clean_col_and_index(outputs[o])

    #TODO add possibility to import hourly data aswell

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
    df2.rename(columns=lambda x: x.strip(), inplace=True)
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
    td = pd.read_csv(config['step1_output'], header=None)
    td[1] = 1
    a = td.groupby(0).sum()
    a = a.set_index(np.arange(1,13))
    b = np.repeat(a[1],24)
    path = Path(__file__).parents[2]
    cs = path/'case_studies'/config['case_study']/'output'
    mc = pd.read_csv(cs/'marginal_cost.txt', sep='\t', index_col=[0,1])
    h = np.resize(np.arange(1,25),288)
    b = b.reset_index()
    b['hour'] = h
    b = b.set_index(['index','hour'])
    mc_scaled = mc.div(b[1],axis=0)
    mc_scaled.to_csv(cs / 'mc_scaled.txt', sep='\t')
    return mc_scaled
