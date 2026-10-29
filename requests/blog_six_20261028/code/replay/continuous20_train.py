"""Pinned trainer; save only the requested pre-update continuation boundary."""
from records.track_3_optimization.harness import hooks
from records.track_3_optimization.harness.checkpoint import save_checkpoint
from records.track_3_optimization import run

def checkpoint_state():
    def hook(config,state):
        boundary=state['step']+1
        if boundary==config['checkpoint_every']:
            save_checkpoint(config,state,boundary)
        return state
    return hook

hooks.checkpoint_state=checkpoint_state
run.main()
