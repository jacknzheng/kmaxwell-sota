#!/usr/bin/env python3
"""Read captured controls and use the production frozen-kernel bubbleforce prediction."""
import argparse,json,pathlib
import torch
from frozen_prediction import predict_kernel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True,type=pathlib.Path);p.add_argument('--update',required=True,type=int);a=p.parse_args()
f=a.capture/'rank0'/f'boundary_{a.update:06d}.pt';x=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
groups=[g for s in x['settings'] if s['optimizer']=='muon' for g in s['groups']]
if not groups:raise ValueError('No Muon group in capture')
rates={(float(g['lr']),float(g['weight_decay']))for g in groups}
if len(rates)!=1:raise ValueError('Heterogeneous Muon groups need a group-specific prediction')
lr,wd=next(iter(rates));tokens=x.get('training_batch_tokens')
if tokens is None:tokens=x.get('batch_receipts',{}).get('used',{}).get('batch_tokens',524288)
if tokens!=524288:raise ValueError(f'This starter expects training batch524288, got{tokens}')
row=predict_kernel(x['kernel_weights_used'].cpu().numpy(),learning_rate=lr,weight_decay=wd,train_tokens=tokens)
row.update(update=a.update,bubbleforce=row['kappa_star']);print(json.dumps(row,indent=2))
