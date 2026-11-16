"""Training curves for the report: every run appends one row per epoch."""
import json
import os
import time

DIR = 'runs/curves'


def log_curve(run, **row):
    os.makedirs(DIR, exist_ok=True)
    path = f'{DIR}/{run}.json'
    rows = json.load(open(path)) if os.path.exists(path) else []
    rows.append(dict(row, time=time.strftime('%Y-%m-%d %H:%M:%S')))
    json.dump(rows, open(path, 'w'), indent=1)
