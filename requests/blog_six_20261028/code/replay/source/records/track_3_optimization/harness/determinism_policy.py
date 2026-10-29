# Verbatim copy of muoff/boxlogs/jstudy/determinism_policy.py (sha256 ee297649fbc7c3962f56c0bec9cd5d9e64f92bdbf71d8d84c57a8bbd7c6de410),
# the accepted recipe of RES-DETERMINISM-RECIPE-PRINTED-REPEAT-20260928 (gate A PASS). Installed by run.py
# when config determinism: true or env CAPTURE_DETERMINISM=1; do not edit here, edit the source.
import os

def apply():
    # Set the launch environment before importing Torch or initializing CUDA.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TORCHINDUCTOR_DETERMINISTIC'] = '1'
    import torch
    import torch._inductor.config as config
    if not torch.__version__.startswith('2.11.'):
        raise RuntimeError('Reviewed determinism policy requires Torch 2.11.x')
    if torch.cuda.is_initialized():
        raise RuntimeError('Determinism policy loaded after CUDA initialization')
    torch.use_deterministic_algorithms(True, warn_only=False)
    config.deterministic = True
    config.max_autotune = False
    config.max_autotune_gemm = False
    config.coordinate_descent_tuning = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    # Preserve production TF32 and dtype choices. Offline HVP explicitly selects math.
    torch._muoff_determinism_installed = True

def verify(require_launch_env=False):
    import torch
    import torch._inductor.config as config
    checks = {
        'installed': getattr(torch, '_muoff_determinism_installed', False),
        'strict': torch.are_deterministic_algorithms_enabled(),
        'not_warn_only': not torch.is_deterministic_algorithms_warn_only_enabled(),
        'inductor': config.deterministic,
        'workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
        'flash_only': torch.backends.cuda.flash_sdp_enabled() and not any([
            torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled()]),
    }
    if require_launch_env:
        checks['python_hash_seed'] = os.environ.get('PYTHONHASHSEED') == '0'
    if not all(checks.values()): raise RuntimeError(str(checks))
    return dict(checks, torch=str(torch.__version__), cuda=torch.version.cuda,
                training_attention='strict deterministic Flash only',
                offline_attention='explicit math SDPA context for double backward')
