import torch

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def batchify_dict(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        rgbs = []
        xs = []
        sigmas = []
        for i in range(0, inputs.shape[0], chunk):
            outs = fn(inputs[i:i+chunk]) # {'rgb' : rgb, 'x' : outs['x'], 'sigma' : outs['sigma']}
            rgbs.append(outs['rgb'])
            xs.append(outs['x'])
            sigmas.append(outs['sigma'])
        rgbs = torch.cat(rgbs, 0)
        xs = torch.cat(xs, 0)
        sigmas = torch.cat(sigmas, 0)
        return {'rgb' : rgbs, 'x' : xs, 'sigma' : sigmas}
    return ret