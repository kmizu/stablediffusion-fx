"""VAE method-hook input placement and exceptional upcast restoration."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from sdfx.extensions import encode_image

@pytest.mark.parametrize('idle_device', ['cpu', 'meta'])
@pytest.mark.parametrize('upcast', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_encoding_uses_execution_device_and_restores_dtype(idle_device, upcast, fail):
    # A method hook materializes an offloaded VAE on the execution device.
    # The CPU-only fixture represents a dormant offload device with meta.
    class Vae:
        device = torch.device(idle_device)
        dtype = torch.float16
        config = SimpleNamespace(force_upcast=upcast, scaling_factor=0.5)
        def to(self, dtype): self.dtype = dtype; return self
        def encode(self, pixels):
            assert pixels.device == torch.device('cpu'), 'input followed dormant VAE storage instead of execution device'
            assert pixels.dtype == (torch.float32 if upcast else torch.float16)
            if fail: raise RuntimeError('injected encode failure')
            return SimpleNamespace(latent_dist=SimpleNamespace(mean=pixels))
    vae = Vae()
    pipe = SimpleNamespace(vae=vae, image_processor=SimpleNamespace(preprocess=lambda _: torch.ones(1, 4, 2, 2)))
    engine = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float16)
    if fail:
        with pytest.raises(RuntimeError, match='injected encode failure'): encode_image(engine, pipe, object())
    else:
        value = encode_image(engine, pipe, object())
        assert value.device == engine.device and value.dtype == engine.dtype
        assert torch.equal(value, torch.full((1, 4, 2, 2), 0.5, dtype=torch.float16))
    assert vae.dtype == torch.float16
