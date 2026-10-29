"""Ordinary MaxEntSlop endpoint entropy solves and linear kernel annealing.

No kappa target, curvature estimate, or stability-boundary matching is used.
"""
import json
from pathlib import Path
import torch

def load_schedule(path,solve,desiderata_class):
    config=json.loads(Path(path).read_text());wps=config['waypoints']
    if len(wps)<2 or not all(a['iteration']<b['iteration'] for a,b in zip(wps,wps[1:])):
        raise ValueError('expected at least two strictly ordered waypoints')
    kernels=[]
    for w in wps:
        s=w['shape']
        d=desiderata_class(mean_lag=s['mean_lag'],log_moment=s['log_moment'],
                          newest_weight=s['c0'],second_newest_weight=s['c1'])
        k=solve(d)
        if not torch.isfinite(k).all() or k.min()<0:raise ValueError('invalid entropy kernel')
        kernels.append(k)
    return torch.stack(kernels),config

def interpolate_waypoint_kernels(kernels,waypoints,step):
    if step<=waypoints[0]['iteration']:return kernels[0]
    if step>=waypoints[-1]['iteration']:return kernels[-1]
    for i,(a,b) in enumerate(zip(waypoints,waypoints[1:])):
        if a['iteration']<=step<=b['iteration']:
            return kernels[i].lerp(kernels[i+1],(step-a['iteration'])/(b['iteration']-a['iteration']))
    raise ValueError('iteration outside schedule')
