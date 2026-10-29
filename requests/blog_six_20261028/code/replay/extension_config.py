"""Write one seed0 continuation config using the pinned eight-rank trainer."""
import argparse
import json
from pathlib import Path
import yaml


def build(home, parent, output, last, factor, window_start=None):
    home, parent, output = map(Path, (home, parent, output))
    receipt = json.loads((parent / 'COMPLETE.json').read_text())
    start = receipt['boundary']
    first = last - 9 if window_start is None else window_start
    if receipt['world_size'] != 8 or not start <= first <= last or last <= start:
        raise ValueError('Expected an eight-rank continuation with a later capture window')
    for rank in range(8):
        if not (parent / f'rank{rank}.pt').is_file():
            raise FileNotFoundError(parent / f'rank{rank}.pt')
    cfg = yaml.safe_load((home / 'parent.yaml').read_text())
    cfg.update(run_id=output.name, output_dir=str(output / 'training'),
               checkpoint_dir=str(output / 'restart'), checkpoint_every=last,
               resume_checkpoint=str(parent), resume_protocol_sha256=receipt['protocol_sha256'],
               resume_anchor=None, stop_step=last + 1, require_world_size=8,
               train_data='data/fineweb10B/fineweb_train_*.bin',
               val_data='data/fineweb10B/fineweb_val_*.bin')
    cfg.pop('start_step', None)
    cfg['capture'] = dict(root=str(output / 'captures'), windows=[[first, last]],
                          anchors=[], window_gradient_only=True)
    for group in cfg['optimizer_groups']:
        group['lr_schedule'] = dict(kind='piecewise_constant', boundaries=[2009], factors=[1., factor])
    for hook in cfg['post_optimizer']:
        if hook['name'] == 'validate_at_step_boundaries':
            hook['hyperparams'] = dict(every=last, final_tenth_every=last)
    return cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--home', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--last', type=int, required=True)
    p.add_argument('--factor', type=float, choices=[.5, 2.], required=True)
    p.add_argument('--window-start', type=int)
    a = p.parse_args()
    if (a.output / 'captures/index.jsonl').exists():
        raise RuntimeError('Existing capture window: preserve it; choose a new output directory')
    cfg = build(a.home, a.parent, a.output, a.last, a.factor, a.window_start)
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / 'training.yaml'
    target.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(target)
