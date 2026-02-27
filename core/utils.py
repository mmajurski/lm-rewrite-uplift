"""Utility functions for managing inspect-ai evaluation log files."""



def remove_empty_logs(fp: str):
    import os
    import json
    fns = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith('.json')]
    for fn in fns:
        with open(fn, 'r') as f:
            data = json.load(f)
            if 'results' not in data:
                os.remove(fn)

def get_completed_logs(fp: str) -> list[dict]:
    import os
    import json
    if not os.path.exists(fp):
        return [], []
    fns = [os.path.join(fp, fn) for fn in os.listdir(fp) if fn.endswith('.json')]
    fns.sort()
    completed_logs = []
    completed_fns = []
    for fn in fns:
        with open(fn, 'r') as f:
            data = json.load(f)
        if 'results' in data:
            completed_logs.append(data['eval'])
            completed_fns.append(fn)
        else:
            os.remove(fn)
    return completed_logs, completed_fns
