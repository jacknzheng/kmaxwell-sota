"""
train_gpt_kmaxwell.py

K-Maxwell generalization of the Track-3 bi-Maxwell Muon baseline
(records/track_3_optimization/results/20260715_bimaxwell_baseline_3210).

From --start onward, Muon's single-EMA first moment is replaced by a mix of K
fixed-rate EMA buffers whose ages are log-spaced in [tau_min, tau_max]. Mix
weights are a deterministic bell-curve score on log(tau), centered at the
geometric midpoint of the window, then normalized. Mean age is derived, not
pinned.

K=2 with --bimaxwell-exact recovers the submission recipe bit-identically:

    M_fast <- lerp(M_fast, g, 1 - 0.85)
    M_slow <- lerp(M_slow, g, 1 - 0.98)
    M_eff  = 0.4385 * M_fast + 0.5615 * M_slow
    update = g.lerp(M_eff, mu)

Stage 0 identity:
    torchrun --standalone --nproc_per_node=1 \\
        records/track_3_optimization/results/20260715_bimaxwell_baseline_3210/train_gpt_kmaxwell.py \\
        --seed 0 --k 2 --bimaxwell-exact

Sweeps: kmaxwell_sweep.py / scripts/kmaxwell_sweep.sh

Val-time --probe-ema (default on) scores each live EMA unit against a holdout
grad. It does not change the Muon update; --no-probe-ema disables it.
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist

# torchrun executes this file with cwd = repo root; keep sibling imports working
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kmaxwell_kernel import (
    BIMAXWELL_START,
    BIMAXWELL_TAU_MAX,
    BIMAXWELL_TAU_MIN,
    build_kmaxwell_kernel,
    format_kmaxwell_recipe,
    parse_weights,
)


def parse_kmaxwell_args(argv=None):
    parser = argparse.ArgumentParser(description="K-Maxwell Muon training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2, help="number of EMA units")
    parser.add_argument("--tau-min", type=float, default=BIMAXWELL_TAU_MIN)
    parser.add_argument("--tau-max", type=float, default=BIMAXWELL_TAU_MAX)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="width of the log(tau) scoring bump (not a sampling stddev)")
    parser.add_argument("--start", type=int, default=BIMAXWELL_START,
                        help="first Muon step on the K-Maxwell path")
    parser.add_argument("--bimaxwell-exact", action="store_true",
                        help="K=2 only: use the exact bi-Maxwell betas/weights")
    parser.add_argument("--weights", default=None,
                        help="comma mix weights, e.g. 0.35,0.25,0.25,0.15 (overrides Gaussian sigma)")
    parser.add_argument("--probe-ema", action=argparse.BooleanOptionalAction, default=True,
                        help="val-time holdout grad vs live EMA units (no change to the update)")
    parser.add_argument("--probe-ema-mbs", type=int, default=4,
                        help="holdout val microbatches for --probe-ema")
    args, _unknown = parser.parse_known_args(argv)
    return args


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def kmaxwell_momentum(grad, m, betas, weights, mu=0.95):
    """In-place K-EMA mix. m has shape (K, *grad.shape). Mutates m and grad."""
    lerp_coeff = (1 - betas).reshape(-1, *([1] * grad.ndim))
    m.lerp_(grad.unsqueeze(0), lerp_coeff)
    w = weights.reshape(-1, *([1] * grad.ndim))
    m_eff = (w * m).sum(dim=0)
    return grad.lerp_(m_eff, mu)

@torch.compile
def muon_update_kmaxwell(grad, m, betas, weights, mu=0.95):
    update = kmaxwell_momentum(grad, m, betas, weights, mu)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95,
                 kmaxwell_betas=None, kmaxwell_weights=None, kmaxwell_start=BIMAXWELL_START):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        self._step = 0
        assert kmaxwell_betas is not None and kmaxwell_weights is not None
        self.km_betas = torch.as_tensor(kmaxwell_betas, dtype=torch.float32).cpu()
        self.km_weights = torch.as_tensor(kmaxwell_weights, dtype=torch.float32).cpu()
        self.km_k = int(self.km_betas.numel())
        self.km_start = int(kmaxwell_start)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    if self._step >= self.km_start:
                        if "m" not in state:
                            # switch step: advance the single-EMA momentum once more and
                            # lazy-init all K units from it -> this step's update is
                            # bit-identical to the baseline's, by construction
                            update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                            state["m"] = state["momentum"].unsqueeze(0).repeat(self.km_k, *([1] * p.ndim))
                        else:
                            betas = self.km_betas.to(device=p.device, dtype=p.dtype)
                            weights = self.km_weights.to(device=p.device, dtype=p.dtype)
                            update = muon_update_kmaxwell(
                                p.grad, state["m"], betas, weights, mu=group["mu"])
                    else:
                        update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        self._step += 1


def probe_kmaxwell_ema(muon_opt, model, val_inputs, val_targets, mbs, probe_mbs, print0):
    """Score each live EMA unit against a holdout val grad. Does not step the opt.

    Runs only after K-Maxwell buffers exist. Clock must already be stopped.
    """
    device = val_inputs.device
    has = torch.zeros((), device=device, dtype=torch.int32)
    for group in muon_opt.param_groups:
        for p in group["params"]:
            if "m" in muon_opt.state.get(p, {}):
                has.fill_(1)
                break
        if int(has.item()) == 1:
            break
    dist.all_reduce(has, op=dist.ReduceOp.MAX)
    if int(has.item()) == 0:
        return

    n_mbs = min(int(probe_mbs), len(val_inputs) // mbs)
    if n_mbs < 1:
        return
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        for i in range(n_mbs):
            model(val_inputs[i * mbs:(i + 1) * mbs], val_targets[i * mbs:(i + 1) * mbs]).backward()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    k_dim = int(muon_opt.km_k)
    acc_align = torch.zeros(k_dim, device=device, dtype=torch.float64)
    acc_mse = torch.zeros(k_dim, device=device, dtype=torch.float64)
    acc_n = torch.zeros((), device=device, dtype=torch.float64)
    for group in muon_opt.param_groups:
        for p in group["params"]:
            state = muon_opt.state.get(p, {})
            if "m" not in state or p.grad is None:
                continue
            m = state["m"].float()
            g = p.grad.float()
            g_norm = torch.linalg.vector_norm(g)
            g_norm_sq = g_norm * g_norm
            flat = m.reshape(k_dim, -1)
            dots = (flat * g.reshape(1, -1)).sum(-1)
            m_norms = torch.linalg.vector_norm(flat, dim=-1)
            n = float(p.numel())
            acc_n += n
            acc_align += n * (dots / (m_norms * g_norm + 1e-12)).double()
            acc_mse += n * ((flat - g.reshape(1, -1)).pow(2).sum(-1) / (g_norm_sq + 1e-12)).double()
    dist.all_reduce(acc_align, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_mse, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_n, op=dist.ReduceOp.SUM)
    model.zero_grad(set_to_none=True)
    if float(acc_n.item()) <= 0:
        return
    acc_align /= acc_n
    acc_mse /= acc_n
    align_s = ",".join(f"{x:.5f}" for x in acc_align.tolist())
    mse_s = ",".join(f"{x:.5f}" for x in acc_mse.tolist())
    print0(f"probe_ema: n={int(acc_n.item())} align={align_s} rel_mse={mse_s}", console=True)


def main():
    with open(sys.argv[0]) as f:
        code = f.read() # read the code of this file ASAP, for logging

    args = parse_kmaxwell_args()
    SEED = args.seed
    explicit = parse_weights(args.weights) if args.weights else None
    tau, betas, weights, mean_age = build_kmaxwell_kernel(
        args.k, args.tau_min, args.tau_max, args.sigma, args.bimaxwell_exact,
        weights=explicit)

    # torchrun sets these env variables
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0

    # logging setup
    if dist.get_rank() == 0:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{uuid.uuid4()}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
                with open(logfile, "a") as f:
                    print(s, file=f)

    # we begin by logging this file itself
    print0(code)
    print0("="*100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
    print0(format_kmaxwell_recipe(
        args.k, args.tau_min, args.tau_max, args.sigma, args.start, SEED,
        args.bimaxwell_exact, tau, betas, weights, mean_age,
        weights_explicit=explicit is not None))
    print0(f"probe_ema={args.probe_ema} probe_ema_mbs={args.probe_ema_mbs}")
    print0("="*100)

    val_tokens = 20 * 524288
    batch_size = 8 * 64 * 1024
    mbs = 64
    val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    model.compile(dynamic=False)


    num_trials = 1

    for _ in range(num_trials):


        ########################################
        #       Init & Optim Hyperparams       #
        ########################################

        # we want to minimize this while still reaching 3.28 val loss
        train_steps = 3250

        # deterministic paired trials: seed the global RNG right before param init
        torch.manual_seed(1337 + SEED)

        # initialize model parameters
        for name, p in model.named_parameters():
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()  # default torch init
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
            elif name.endswith("bias"):
                w.zero_()
            elif name.endswith("gains"):
                w.normal_(mean=1, std=0)
            else:
                raise Exception(f"Uninitialized parameter: {name}")

        # create the optimizer(s)
        optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                            dict(params=[model.proj.weight], lr=0.004),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                           betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
        optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                          lr=0.025, weight_decay=0.05,
                          kmaxwell_betas=betas, kmaxwell_weights=weights, kmaxwell_start=args.start)
        optimizers = [optimizer1, optimizer2]
        assert set(p for opt in optimizers for group in opt.param_groups
                   for p in group["params"]) == set(model.parameters())
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # learning rate schedule: stable then decay
        def set_hparams(step, cooldown_frac=0.7):
            progress = step / train_steps
            assert 0 <= progress < 1
            if progress < 1 - cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / cooldown_frac
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * eta


        ########################################
        #        Training and Validation       #
        ########################################

        train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)
        # start the clock
        training_time = 0
        last_val_step = 0
        dist.barrier()
        t0 = time.perf_counter()
        for step in range(train_steps + 1):

            # --------------- VALIDATION SECTION -----------------
            # dense eval-only validation over the crossing zone (uniform across all seeds;
            # the earliest formally-passing step is selected the same way for every trial)
            dense = 2900 <= step <= 3250 and step % 10 == 0
            val_step_freq = 125 if step / train_steps < 0.9 else 25
            if step == train_steps or step % val_step_freq == 0 or dense:
                # stop the clock
                dist.barrier()
                time_since_last_val = time.perf_counter() - t0
                step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
                last_val_step = step
                training_time += time_since_last_val
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    assert len(val_inputs) % mbs == 0
                    for i in range(len(val_inputs) // mbs):
                        val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= val_tokens
                print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms", console=True)
                if args.probe_ema:
                    probe_kmaxwell_ema(
                        optimizer2, model, val_inputs, val_targets, mbs,
                        args.probe_ema_mbs, print0)
                model.train()
                # start the clock again
                dist.barrier()
                t0 = time.perf_counter()

            if step == train_steps:
                break

            # --------------- TRAINING SECTION -----------------
            inputs, targets = next(train_loader)
            # accumulate across microbatches in case we are running with fewer than 8 gpus
            assert len(inputs) % mbs == 0
            for i in range(len(inputs) // mbs):
                model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, name
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            # set optimization hyperparameters and take a step
            set_hparams(step)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            approx_training_time = training_time + (time.perf_counter() - t0)
            print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
