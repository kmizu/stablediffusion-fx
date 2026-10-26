"""Native positive/negative IP-Adapter features for Neutral Prompt's internal guidance layouts."""
from .protocol import EngineError, UNSUPPORTED


def prepare(pipe, image, batch, guided):
    import torch
    with torch.no_grad():
        once = pipe.prepare_ip_adapter_image_embeds(image, None, pipe._execution_device, 1, True)
    canonical, native = [], []
    for value in once:
        if value.shape[0] != 2:
            raise EngineError(UNSUPPORTED, "Neutral Prompt expected native negative/positive IP-Adapter features")
        negative, positive = value.chunk(2)
        repeats = [batch] + [1] * (value.ndim - 1)
        canonical.append(torch.cat([negative.repeat(*repeats), positive.repeat(*repeats)]))
        native.append(value if guided else positive)
    return canonical, native


def align(embeds, batch, layout, device, dtype):
    import torch
    out = []
    for value in embeds:
        if value.shape[0] != 2 * batch:
            raise EngineError(UNSUPPORTED, "Neutral Prompt received mismatched IP-Adapter image batches")
        negative, positive = value.chunk(2)
        out.append(torch.cat([positive if p else negative for p in layout]).to(device, dtype))
    return out
