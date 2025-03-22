"""
This document includes functions that are (frequently)
used as tools
"""

import json

def predictions_json(model, predictions:dict):
    """
    This function takes in a dictionary with predictions
    of a model and stores them within a .json-file.
    """

    if "deepseek" in model:
        model_name = "deepseek"
    elif "llama" in model:
        model_name = "llama"
    # define outputfile path
    output_filename = f"./WPS-HRI/predictions/{model_name}_predictions.json"

    with open(output_filename, "w") as f:    
        json.dump(predictions, f, indent=4)