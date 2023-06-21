from typing import List, Dict, Any
import json 

def load_gsm8k(path: str) -> List[Dict[str, Any]]:
    """
        Load GSM8K dataset as json object.

        Args:
            path (str): the path where the data is located

        Returns:
            list of dictionaries where each dictionary is an example from the data
    """
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def load_aqua(path: str) -> List[Dict[str, Any]]:
    """
        Load AQUA dataset as json object.

        Args:
            path (str): the path where the data is located

        Returns:
            dataset (list): list of dictionaries where each dictionary is an example from the data
    """
    with open(path) as fh:
        dataset = [json.loads(line) for line in fh.readlines() if line]
    
    for example in dataset:
        example['answer'] = example['correct']
    
    return dataset 

def load_strategyqa(path: str) -> List[Dict[str, Any]]:
    """
        Load the StrategyQA dataset as json object.

        Args:
            path (str): the path where the data is located
        
        Returns:
            a list of dictionaries where each dictionary is an example from the data   
        
    """
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)