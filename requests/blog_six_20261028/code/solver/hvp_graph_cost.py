"""Inspect an existing fake-traced graph; do not execute or alter its operators."""
import os
from pathlib import Path
import sys
import time
from collections import Counter
from common import write


def analyze(graph_module):
    nodes=list(graph_module.graph.nodes)
    placeholders=[n for n in nodes if n.op=='placeholder']
    directions=set(placeholders[2:])
    dependent={}
    counts={False:Counter(),True:Counter()}
    flops={False:0,True:0}
    for node in nodes:
        dependent[node]=node in directions or any(dependent[n] for n in node.all_input_nodes)
        if node.op!='call_function':continue
        name=getattr(node.target,'__name__',str(node.target))
        counts[dependent[node]][name]+=1
        if name in ('bf16_mm','mm.default','bf16_bmm','bmm.default','bf16_addmm','addmm.default'):
            operands=node.args[1:3] if 'addmm' in name else node.args[:2]
            shapes=[tuple(int(x) for x in a.meta['val'].shape) for a in operands]
            a,b=shapes
            batch=a[0] if len(a)==3 else 1
            flops[dependent[node]]+=2*batch*a[-2]*a[-1]*b[-1]
    boundary=[];storages={}
    for node in nodes:
        if dependent[node] or not any(dependent[user] for user in node.users):continue
        value=node.meta.get('val')
        if not hasattr(value,'numel'):continue
        size=int(value.numel()*value.element_size())
        resident=node.op in ('get_attr','placeholder')
        boundary.append(dict(name=node.name,operation=str(node.target),shape=list(value.shape),
            dtype=str(value.dtype),logical_bytes=size,already_resident_input_or_weight=resident))
        if not resident:
            storage=value.untyped_storage()
            storages[storage._cdata]=max(storages.get(storage._cdata,0),storage.nbytes())
    total=sum(flops.values())
    return dict(input_shapes=[list(n.meta['val'].shape) for n in placeholders[:2]],
        direction_inputs=len(directions),node_counts={str(k):dict(v) for k,v in counts.items()},
        gemm_flops_primal=flops[False],gemm_flops_direction_dependent=flops[True],
        modelled_primal_gemm_share=flops[False]/total if total else None,
        boundary=boundary,nonresident_boundary_storage_bytes=sum(storages.values()),
        boundary_logical_bytes=sum(v['logical_bytes'] for v in boundary if not v['already_resident_input_or_weight']),
        note='Syntactic direction dependence and GEMM operation counts from fake metadata; not measured runtime. '
             'Boundary storage excludes already resident input tensors and weights. Compiler fusion/lifetimes may differ.')


def install():
    from records.track_3_optimization.analysis import bf16_gemm
    original=bf16_gemm.rewrite_bf16_gemms
    def rewrite(gm):
        result=original(gm)
        if int(os.environ.get('RANK','0'))==0:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            try:
                record=analyze(gm)
                shape='_'.join(str(n) for n in record['input_shapes'][0])
                record['time']=time.time()
                write(output/f'hvp_graph_cost_{shape}.json',record)
            except Exception as error:
                write(output/'hvp_graph_cost_error.json',dict(error=repr(error)))
        return result
    bf16_gemm.rewrite_bf16_gemms=rewrite
