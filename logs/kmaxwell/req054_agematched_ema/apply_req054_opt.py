"""REQ-054: add AgeMatchedEmaMuon (single EMA, decay follows K-Maxwell's scheduled average age) + register."""
import sys
MUON="records/track_3_optimization/optimizers/muon.py"; INIT="records/track_3_optimization/optimizers/__init__.py"
s=open(MUON).read()
CLS='''
class AgeMatchedEmaMuon(AnnealedDecayMuon):
    """Single-EMA control whose decay tracks K-Maxwell's scheduled average memory age.
    A(t) = sum_i w_i(t) * beta_i/(1-beta_i) with w_i(t) linearly interpolated start_weights->end_weights
    over [switch_step, anneal_end_step] (exactly annealed_weights_muon's weight schedule); beta(t)=A(t)/(1+A(t)).
    Only the single buffer's decay is scheduled; the outer Nesterov blend mu is fixed. Baseline Muon through switch_step."""
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95,
                 decays=(), start_weights=(), end_weights=(),
                 switch_step=1000, anneal_end_step=3249):
        super().__init__(params, lr=lr, weight_decay=weight_decay, mu=mu,
                         beta_start=0.95, beta_end=0.95, switch_step=switch_step, anneal_end_step=anneal_end_step)
        assert len(decays)==len(start_weights)==len(end_weights) and len(decays)>=1
        self.decays=tuple(float(d) for d in decays)
        self.start_weights=tuple(float(w) for w in start_weights)
        self.end_weights=tuple(float(w) for w in end_weights)

    def scheduled_age(self):
        a=self.interpolation_fraction()
        w=[s+a*(e-s) for s,e in zip(self.start_weights,self.end_weights)]
        num=sum(wi*bi/(1.0-bi) for wi,bi in zip(w,self.decays)); den=sum(w)
        return num/den if den>0 else 0.0

    def current_beta(self):
        if self._muon_steps_seen <= self.switch_step:
            return self.beta_start
        A=self.scheduled_age(); return A/(1.0+A)

'''
anchor="\ndef compile_weighted_decays_kernel("
assert anchor in s and "class AgeMatchedEmaMuon" not in s
s=s.replace(anchor, CLS+anchor,1); open(MUON,"w").write(s)
i=open(INIT).read()
i=i.replace("AnnealedDecayMuon,","AnnealedDecayMuon, AgeMatchedEmaMuon," ,1) if "AgeMatchedEmaMuon" not in i.split("_REGISTRY")[0] else i
# add import (AnnealedDecayMuon is imported already) + registry
if "AgeMatchedEmaMuon" not in i:
    i=i.replace("AnnealedDecayMuon", "AnnealedDecayMuon, AgeMatchedEmaMuon",1)
i=i.replace('    "annealed_decay_muon": AnnealedDecayMuon,',
            '    "annealed_decay_muon": AnnealedDecayMuon,\n    "age_matched_ema_muon": AgeMatchedEmaMuon,')
open(INIT,"w").write(i); print("AgeMatchedEmaMuon added + registered")
