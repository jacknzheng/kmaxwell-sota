"""Plot qualified coupled eigenvalues, without confusing residual and sampling."""
import argparse
import csv
import html
import json
from pathlib import Path


def qualified_rows(progress):
    rows=[]
    for probe in progress.get('probes',[]):
        if not probe.get('qualified') or probe.get('mu') is None:
            continue
        histories=[s['history'] for s in probe.get('searches',[]) if s.get('qualified')]
        rows.append(dict(tokens=probe['tokens'],offset=probe['offset'],mu=probe['mu'],
            seconds=probe['seconds'],operator_calls=probe.get('operator_calls'),
            residual=max((h[-1]['residual'] for h in histories),default=None),
            dimension=max((h[-1]['dimension'] for h in histories),default=None)))
    return sorted(rows,key=lambda p:(p['offset'],p['tokens']))


def preliminary_row(progress):
    """Keep an in-progress single-start estimate out of qualified CSV data."""
    probe, search = progress.get('current_probe', {}), progress.get('current_search', {})
    if (probe.get('tokens') is None or search.get('mu') is None or
            search.get('tokens', probe.get('tokens')) != probe.get('tokens') or
            search.get('offset', probe.get('offset', 0)) != probe.get('offset', 0) or
            any(p.get('qualified') and p.get('tokens') == probe['tokens'] and
                p.get('offset') == probe.get('offset', 0) for p in progress.get('probes', []))):
        return None
    return dict(tokens=probe['tokens'], offset=probe.get('offset', 0),
                mu=search['mu'], residual=search.get('residual'),
                dimension=search.get('dimension'), seed=search.get('seed'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--progress',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-preliminary',action='store_true',
                   help='Show current single-start estimate separately; never qualify it')
    p.add_argument('--predicted-mu',type=float,help='Optional prediction in the same operator/normalization units')
    args=p.parse_args()
    progress=json.loads(args.progress.read_text())
    rows=qualified_rows(progress)
    preliminary=preliminary_row(progress) if args.include_preliminary else None
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'coupled_token_convergence.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['tokens','offset','mu','seconds','operator_calls','residual','dimension'])
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4.5))
    offsets=sorted({r['offset'] for r in rows})
    for offset in offsets:
        selected=[r for r in rows if r['offset']==offset]
        axes[0].plot([r['tokens'] for r in selected],[r['mu'] for r in selected],'o-',label=f'offset {offset:,}')
    nested=[r for r in rows if r['offset']==0]
    if preliminary is not None:
        axes[0].scatter([preliminary['tokens']], [preliminary['mu']], marker='D',
                        facecolors='none', edgecolors='darkorange', s=70, label='Preliminary: checks unfinished')
        if nested and preliminary['offset'] == 0 and preliminary['tokens'] > nested[-1]['tokens']:
            last=nested[-1]
            axes[0].plot([last['tokens'],preliminary['tokens']],[last['mu'],preliminary['mu']],
                         ':',color='darkorange')
            change=100*abs(preliminary['mu']-last['mu'])/max(abs(preliminary['mu']),abs(last['mu']))
            axes[1].scatter([preliminary['tokens']],[change],marker='D',facecolors='none',
                            edgecolors='darkorange',s=70,label='Preliminary change')
    if args.predicted_mu is not None:
        axes[0].axhline(args.predicted_mu,color='green',linestyle='--',label='Frozen-law prediction')
    changes=[100*abs(b['mu']-a['mu'])/max(abs(a['mu']),abs(b['mu'])) for a,b in zip(nested,nested[1:])]
    axes[1].plot([r['tokens'] for r in nested[1:]],changes,'o-')
    axes[1].axhline(progress.get('token_rtol',.01)*100,color='red',linestyle='--',label='1% sampling gate')
    axes[0].set_ylabel('Largest positive real eigenvalue μ (token-mean units)')
    axes[1].set_ylabel('Relative change from preceding nested probe (%)')
    for axis in axes:
        axis.set_xscale('log',base=2);axis.set_xlabel('Validation tokens');axis.grid(alpha=.25);axis.legend()
        ticks=sorted({r['tokens'] for r in rows} | ({preliminary['tokens']} if preliminary else set()))
        axis.set_xticks(ticks, [f'{t/1048576:g}M' for t in ticks])
    status='Sampling convergence established' if progress.get('empirical_convergence') else 'Sampling convergence NOT established'
    decay='decay included' if progress.get('include_weight_decay',False) else 'polar only; decay excluded'
    figure.suptitle(f'Checkpoint {progress.get("update",1250)} · all 72 Muon matrices · {decay}\n{status}')
    figure.tight_layout()
    figure.savefig(args.output/'coupled_token_convergence.png',dpi=160)
    figure.savefig(args.output/'coupled_token_convergence.svg')
    plt.close(figure)
    cells=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in
        ('tokens','offset','mu','seconds','residual','dimension'))+'</tr>' for r in rows)
    preliminary_note=(f'<p><strong>Preliminary, not independently qualified:</strong> '
        f'{preliminary["tokens"]:,} tokens, μ={preliminary["mu"]:.10g}; '
        f'seed {preliminary["seed"]}, dimension {preliminary["dimension"]}, '
        f'residual {preliminary["residual"]}. This point is not in the qualified CSV.</p>'
        if preliminary is not None else '')
    body=f'''<!doctype html><html><meta charset="utf-8"><title>Coupled Muon convergence</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;text-align:right}}table{{border-collapse:collapse}}tr{{border-bottom:1px solid #ccc}}</style>
<h1>Coupled Muon μ versus validation tokens</h1><p>{status}. The connected blue points are numerically qualified completed probes; {decay}.
Adam parameters are frozen; all 72 Muon matrices and their cross curvature are included.</p>
<img src="coupled_token_convergence.svg" alt="Coupled Muon eigenvalues and nested relative changes">
{preliminary_note}
<p>The sampling gate needs two consecutive nested changes ≤1% and three disjoint probe estimates agreeing within 1%.
A small Arnoldi residual alone does not establish sampling convergence. These points share nested data and are not independent confidence intervals.</p>
<table><tr><th>Tokens</th><th>Offset</th><th>μ</th><th>Seconds</th><th>Residual</th><th>Dimension</th></tr>{cells}</table>
<p><a href="coupled_token_convergence.csv">CSV</a> · <a href="coupled_token_convergence.png">PNG</a></p></html>'''
    (args.output/'index.html').write_text(body)


if __name__=='__main__':main()
