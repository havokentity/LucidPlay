def test_pick_device_returns_valid_torch_device():
    import torch
    from src.device import pick_device

    d = pick_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cuda", "mps", "cpu")


def test_pin_memory_only_for_cuda():
    import torch
    from src.device import pin_memory_for

    assert pin_memory_for(torch.device("cuda")) is True
    assert pin_memory_for(torch.device("mps")) is False
    assert pin_memory_for(torch.device("cpu")) is False
