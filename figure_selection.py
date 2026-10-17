"""Resolve the pinned figure instances in figure_selection.json to instance ids."""
import json
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
SELECTION_JSON = os.path.join(_HERE, 'figure_selection.json')


def resolve(figure, data_dir, inst_csv_name='non_exercise.csv'):
    """Return {role: {'uid': int, 'eid': str}} for one figure ('qualitative' or 'scorecard')."""
    with open(SELECTION_JSON) as f:
        spec = json.load(f)[figure]
    inst = pd.read_csv(os.path.join(data_dir, inst_csv_name),
                       usecols=['ex', 'id', 'time_series_start'])
    inst['id'] = inst['id'].astype(float).astype(int)
    inst['time_series_start'] = pd.to_datetime(inst['time_series_start'])
    out = {}
    for role, s in spec.items():
        hit = inst[(inst['id'] == int(s['user_id'])) &
                   (inst['time_series_start'] == pd.Timestamp(s['time_series_start']))]
        if len(hit) != 1:
            raise LookupError(f'{figure}/{role}: expected one instance for {s}, found {len(hit)}')
        out[role] = {'uid': int(s['user_id']), 'eid': str(int(hit['ex'].iloc[0]))}
    return out
