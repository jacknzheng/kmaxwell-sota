import torch

HEAD_DIM = 128
_CGI_HEAD_MEAN_SHRINK = 0.50


def headmean_antithetic_pair(shape, *, device, layer_idx: int, seed: int, head_dim: int = HEAD_DIM):
    width = shape[-1]
    if width % head_dim != 0:
        s0 = (torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2 - 1)
        return s0, -s0
    gen = torch.Generator(device=device)
    gen.manual_seed(0xA9170000 + seed * 1009 + layer_idx * 9176)
    heads0 = []
    heads1 = []
    for _ in range(width // head_dim):
        h0 = (torch.randint(0, 2, (head_dim,), device=device, generator=gen, dtype=torch.int64) * 2 - 1).float()
        plus0 = int((h0 > 0).sum().item())
        plus0 = int(round(head_dim / 2 + _CGI_HEAD_MEAN_SHRINK * (plus0 - head_dim / 2)))
        plus0 = max(0, min(head_dim, plus0))
        h0 = torch.cat([
            torch.ones(plus0, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus0, device=device, dtype=torch.float32),
        ])
        h0 = h0[torch.randperm(head_dim, device=device, generator=gen)]
        plus1 = head_dim - plus0
        h1 = torch.cat([
            torch.ones(plus1, device=device, dtype=torch.float32),
            -torch.ones(head_dim - plus1, device=device, dtype=torch.float32),
        ])
        h1 = h1[torch.randperm(head_dim, device=device, generator=gen)]
        heads0.append(h0)
        heads1.append(h1)
    return torch.cat(heads0).reshape(shape), torch.cat(heads1).reshape(shape)


@torch.no_grad()
def initialize_soap341(model, *, seed):
    """Call after seeded model construction; do not reseed or reset its weights."""
    for name, p in model.named_parameters():
        if "proj" in name:
            p.data.zero_()
    num_blocks = len(model.blocks)
    for layer, block in enumerate(model.blocks):
        ramp = layer / (num_blocks - 1) if num_blocks > 1 else 0.0
        block.mlp.fc.weight.data.mul_(1.0 - 0.30 * ramp)
    pair_next = None
    for layer, block in enumerate(model.blocks):
        if layer >= 6 and (layer - 6) % 2 == 1:
            signs = pair_next
            pair_next = None
        elif layer >= 6:
            signs, pair_next = headmean_antithetic_pair(
                block.norm1.gains.shape, device=block.norm1.gains.device,
                layer_idx=layer, seed=seed)
        else:
            signs = (torch.randint(0, 2, block.norm1.gains.shape,
                                  device=block.norm1.gains.device,
                                  dtype=torch.float32) * 2 - 1)
        block.norm1.gains.data.copy_((1.0 - 0.125 * signs).to(block.norm1.gains.dtype))
        block.norm2.gains.data.copy_((1.0 + 0.125 * signs).to(block.norm2.gains.dtype))


def soap341_schedule_end(step):
    return 2900 - (60 if step >= 1400 else 0)


def soap341_mu(step):
    end = soap341_schedule_end(step)
    if step < 300:
        return 0.85 + step / 300 * (0.95 - 0.85)
    if step > end - 200:
        return 0.95 - (step - (end - 200)) / 200 * (0.95 - 0.85)
    return 0.95


def soap341_validation_due(boundary, *, stop_step=2600):
    """Original pre-update boundary predicate, including no initial validation."""
    return (boundary == 2900 or boundary == stop_step
            or (boundary > 0 and boundary % 125 == 0)
            or (2500 <= boundary <= 2800 and boundary % 5 == 0)
            or boundary in {2810, 2820, 2830, 2840, 2850, 2860, 2870,
                            2880, 2890, 2895, 2900})
