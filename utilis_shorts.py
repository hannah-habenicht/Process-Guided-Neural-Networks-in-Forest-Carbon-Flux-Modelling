# short usful code snippets
import json
import os
import shutil

def save_json(file, path, name):
    '''saves dictionary'''
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise
    name= name + '.json'
    with open(os.path.join(path, name),'w') as json_file:    
        json_=json.dump(file,json_file )
    return

def save_frozen_config(config, output_path="configs"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)

def json_open(path):
    '''opens json'''
    with open(path) as json_file:
        dict_ = json.load(json_file)
    return dict_

NORMALIZATION_FILENAME = "normalization.json"


def save_normalization(normalization, path, filename=NORMALIZATION_FILENAME):
    """Persist a fold's fitted z-score constants next to its checkpoint.

    The constants are fitted on the TRAIN split and are part of the model: without
    them a checkpoint cannot be turned back into physical units without rebuilding
    the training data. Values are numpy arrays, so coerce to plain lists for JSON.

    Returns the written path, or None if there was nothing to write.
    """
    if normalization is None:
        return None

    serialisable = {}
    for key, value in dict(normalization).items():
        if hasattr(value, "tolist"):
            serialisable[key] = value.tolist()
        else:
            serialisable[key] = value

    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, filename)
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=4)
    return out_path


def load_normalization(path, filename=NORMALIZATION_FILENAME):
    """Read back constants written by `save_normalization`, or None if absent.

    Lists are returned as-is; `Basic_dataset._apply_normalization` coerces them
    with `np.asarray`, so no conversion is needed here.
    """
    full = path if path.endswith(".json") else os.path.join(path, filename)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def splits_path(path, steps=3):
    parts = path.split(os.sep)
    first_steps = parts[:steps+1] if parts[0] == '' else parts[:steps]
    return os.sep.join(first_steps)

def errotype_to_string(errortype):
    if errortype == 'test_mse':
        return 'Mean Squared Error'
    elif errortype == 'test_mae':
        return 'Mean Absolute Error'
    elif errortype == 'test_mse_et':
        return 'MSE ET'
    elif errortype == 'test_mae_et':
        return 'MAE ET'
    elif errortype == 'test_mse_gpp':
        return 'MSE GPP'
    elif errortype == 'test_mae_gpp':
        return 'MAE GPP'
    return