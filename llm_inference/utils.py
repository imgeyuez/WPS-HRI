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

    # define outputfile path
    output_filename = f"C:/Users/imgey/Desktop/MASTER_POTSDAM/WiSe2425/PM1_argument_mining/WPS-HRI/predictions/{model}_predictions.json"

    with open(output_filename, "w") as f:    
        json.dump(predictions, f, indent=4)
