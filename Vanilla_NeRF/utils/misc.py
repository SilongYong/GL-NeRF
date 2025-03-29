from copy import deepcopy
import numpy as np
import random
import torch
import torch.distributed as dist


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def overwrite_config(args, past_args):
    for k, v in past_args.items():
        if hasattr(args, k): # skip if args has past_args
            continue
        setattr(args, k, v)
    return args

def anytype2bool_dict(s):
    # check str
    if not isinstance(s, str):
        return s
    else:
        # try int
        try:
            ret = int(s)
        except:
            # try bool
            if s.lower() in ('true', 'false'):
                ret = s.lower() == 'true'
            # try float
            else:
                try:
                    ret = float(s)
                except:
                    ret = s
        return ret

def parse_string_to_dict(field_name, value):
    fields = field_name.split('.')
    for fd in fields[::-1]:
        res = {fd: anytype2bool_dict(value)}
        value = res
    return res

def merge_to_dicts(a, b):
    if isinstance(b, dict) and isinstance(a, dict):
        a_and_b = set(a.keys()) & set(b.keys())
        every_key = set(a.keys()) | set(b.keys())
        return {k: merge_to_dicts(a[k], b[k]) if k in a_and_b else
                   deepcopy(a[k] if k in a else b[k]) for k in every_key}
    return deepcopy(type(a)(b))

def override_cfg_from_list(cfg, opts):
    assert len(opts) % 2 == 0, 'Paired input must be provided to override config, opts: {}'.format(opts)
    for ix in range(0, len(opts), 2):
        opts_dict = parse_string_to_dict(opts[ix], opts[ix + 1])
        cfg = merge_to_dicts(cfg, opts_dict)
    return cfg