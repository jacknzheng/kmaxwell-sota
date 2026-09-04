"""REQ-046 code changes at 25d3208: PerMatrixClipMuon (grad clip pre-momentum) + registry + probe field."""
import sys
MUON="records/track_3_optimization/optimizers/muon.py"
INIT="records/track_3_optimization/optimizers/__init__.py"
PROBE="records/track_3_optimization/offline_analysis/measure_per_matrix_curvature.py"

CLIP_CLASS='''
class PerMatrixClipMuon(BimaxwellMuon):
    """BimaxwellMuon that scales each matrix's gradient by a per-matrix clip
    multiplier BEFORE the momentum lerp. A scale applied inside polar_express
    cancels in the unit-spectral-norm normalisation (X/(c*||X||)); applied to the
    gradient before the momentum update it enters the buffer trajectory instead.
    ``clip_multipliers`` follows the size-descending sorted-param order used for
    lr_multipliers. Momentum state/buffers are exactly BimaxwellMuon's, so
    serialized bi-Maxwell fork states load unchanged."""
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95,
                 fast_decay=PR340_BIMAXWELL["fast_decay"],
                 slow_decay=PR340_BIMAXWELL["slow_decay"],
                 fast_weight=PR340_BIMAXWELL["fast_weight"],
                 switch_step=PR340_BIMAXWELL["switch_step"],
                 clip_multipliers=()):
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu,
                         fast_decay=fast_decay, slow_decay=slow_decay,
                         fast_weight=fast_weight, switch_step=switch_step)
        if len(clip_multipliers) != len(self.sorted_params()):
            raise ValueError(f"clip_multipliers has {len(clip_multipliers)} entries"
                             f" for {len(self.sorted_params())} matrices")
        self.clip_multipliers = tuple(float(c) for c in clip_multipliers)
        self._clip_by_id = {id(p): self.clip_multipliers[i]
                            for i, p in enumerate(self.sorted_params())}

    def compute_polar_input(self, p, state, group):
        c = self._clip_by_id[id(p)]
        if c != 1.0:
            p.grad.mul_(c)   # scale grad BEFORE the momentum lerp inside super()
        return super().compute_polar_input(p, state, group)


'''

def patch(path, subs, label):
    s=open(path).read()
    for old,new in subs:
        if old not in s:
            print(f"FAIL {label}: anchor not found:\n  {old[:80]!r}"); sys.exit(2)
        if s.count(old)!=1:
            print(f"FAIL {label}: anchor not unique ({s.count(old)}x):\n  {old[:80]!r}"); sys.exit(2)
        s=s.replace(old,new,1)
    open(path,"w").write(s); print(f"OK {label}")

# 1. muon.py: insert class before ScheduledWeightsMuon
patch(MUON, [("\nclass ScheduledWeightsMuon(", CLIP_CLASS+"\nclass ScheduledWeightsMuon(")], "muon.py")
# 2. __init__.py: import + registry entry
patch(INIT, [
    ("PerMatrixLrMuon, ScheduledWeightsMuon, SgdBlocks,",
     "PerMatrixClipMuon, PerMatrixLrMuon, ScheduledWeightsMuon, SgdBlocks,"),
    ('    "per_matrix_lr_muon": PerMatrixLrMuon,\n',
     '    "per_matrix_lr_muon": PerMatrixLrMuon,\n    "per_matrix_clip_muon": PerMatrixClipMuon,\n'),
], "__init__.py")
# 3. probe: --clip_json arg + load + clipped_gradient_block_norm field
patch(PROBE, [
    ('    ap.add_argument("--out_tag", type=str, default="per_matrix_curvature")',
     '    ap.add_argument("--out_tag", type=str, default="per_matrix_curvature")\n'
     '    ap.add_argument("--clip_json", type=str, default=None,\n'
     '                    help="JSON {matrix_name: clip_multiplier}; adds clipped_gradient_block_norm")'),
    ("    all_names = target_matrix_names()",
     "    clip_map = {}\n    if args.clip_json:\n        import json as _cj\n        clip_map = _cj.load(open(args.clip_json))\n    all_names = target_matrix_names()"),
    ("                gradient_block_norm=float(g[i].norm()),",
     "                gradient_block_norm=float(g[i].norm()),\n"
     "                clipped_gradient_block_norm=float(clip_map.get(name, 1.0)) * float(g[i].norm()),"),
], "probe")
print("ALL PATCHES APPLIED")
