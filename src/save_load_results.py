from datetime import datetime
import os
import pandas as pd
from typing import Tuple

DIRECTORY_DIC = {'standard' : '../Results/Standard_Prompting/',
                 'cot' : '../Results/Random_Manual_CoT/'}

def save_results_different_contexts(dataset_list: list, dataset_name: str, strategy: str, seed: int):
    """
        Export the results from the pandas dataframes into csv files.

        Args:
            dataset_name (str): name of the dataset
            dataset_list (list): a list of pandas dataframes
            strategy (str): strategy should be 'standard' or 'cot'

        Returns:
            return_msg (str): a message whether the data storing was done successfully or not
    """


    strategy_directory = DIRECTORY_DIC[strategy]
    now = datetime.now()
    date, time = now.strftime("%Y-%m-%d %H:%M:%S").replace('-', '_').replace(':', '_').split(' ')
    identifier =  dataset_name + '_d_' + date + '_t_' + time + '_seed_' + str(seed)
    
    all_runs_directory = strategy_directory + identifier + '/'
    if not os.path.exists(all_runs_directory):
        os.mkdir(all_runs_directory)
    
    for i, dataset in enumerate(dataset_list):
        csv_path = all_runs_directory + f'df_run_{i+1}.csv'
        try:
            # export dataframe into csv file
            dataset.to_csv(csv_path, index=False)
        except Exception as e:
            print(f'Dataframe of Run {i} could not be exported into csv file!')
            print(f'Error msg: {e}')


def load_results_different_contexts(identifier: str, strategy: str) -> list:
    """
        Load the results from the csv files into pandas dataframes. 

        Args:
            identifier (str): the identifier of the run
            strategy (str): strategy should be 'standard' or 'cot'
        
        Returns:
            df_list (list): a list of pandas dataframes
    """

    directory = DIRECTORY_DIC[strategy] + identifier 

    # get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    df_list = []
    # loop over the CSV files and read them into pandas dataframes
    for csv_file in csv_files:
        csv_file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(csv_file_path)
        df_list.append(df)

    return df_list


def save_results_same_contexts(dataset_name: str, dataframes_dict : dict, strategy: str) -> str :
    """
        Export the results from the pandas dataframes into csv files.

        Args:
            dataset_name (str): name of the dataset
            dataframes_dict (dict): a dictionary which includes for each seed the corresponding result dataframe and context
            strategy (str): strategy should be 'standard' or 'cot'

        Returns:
            return_msg (str): a message whether the data storing was done successfully or not
    """
    
    directory = DIRECTORY_DIC[strategy]
    now = datetime.now()
    date, time = now.strftime("%Y-%m-%d %H:%M:%S").replace('-', '_').replace(':', '_').split(' ')
    identifier =  dataset_name + '_d_' + date + '_t_' + time
    
    run_directory = directory + identifier + '/'
    os.mkdir(run_directory)
    
    
    for seed in dataframes_dict:
        csv_path = run_directory + f'df_seed_{seed}.csv'
        txt_path = run_directory + f'context_seed_{seed}.txt'
        try:
            # export dataframe into csv file
            dataframes_dict[seed][0].to_csv(csv_path, index=False)
        except Exception as e:
            print(f'Dataframe for seed {seed} could not be exported into csv file!')
            print(f'Error msg: {e}')
           
        try:
            # export few-shot demonstrations (context) into txt file
            with open(txt_path, "w") as f:
                f.write(dataframes_dict[seed][1])

        except Exception as e:
            print(f'Context for seed {seed} could not be exported into txt file!')
            print(f'Error msg: {e}')

        
def load_results_same_contexts(identifier: str, strategy: str) -> Tuple[list, list]:
    """
        Load the results from the csv files into pandas dataframes. 

        Args:
            identifier (str): the identifier of the run
            strategy (str): strategy should be 'standard' or 'cot'
        
        Returns:
            df_list (list): a list of pandas dataframes
            text_list (list): a list of contexts
    """

    directory = DIRECTORY_DIC[strategy] + identifier 

    # get a list of all CSV files in the directory
    csv_list = [f for f in os.listdir(directory) if f.endswith('.csv')]
    txt_list = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    df_list = []
    text_list = []
    # loop over the CSV files and read them into pandas dataframes
    for csv_file, txt_file in zip(csv_list, txt_list):
        csv_file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(csv_file_path)
        df_list.append(df)

        txt_file_path = os.path.join(directory, txt_file)
        with open(txt_file_path, 'r') as file:
            txt_content = file.read()        
        text_list.append(txt_content)
        
    return df_list, text_list