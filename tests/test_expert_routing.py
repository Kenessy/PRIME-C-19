import torch

from prime_c19.tp6.experts import LocationExpertRouter


def test_location_expert_router_routes_by_pointer_modulo():
    router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=3)

    # Make each expert output a constant equal to its index via bias.
    for idx, expert in enumerate(router.experts):
        with torch.no_grad():
            expert.weight.zero_()
            expert.bias.fill_(float(idx))

    x = torch.randn(6, 2)
    ptr = torch.tensor([0, 1, 2, 3, 4, 5])
    out = router(x, ptr)

    expected = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    assert torch.allclose(out.cpu(), expected)


def test_location_expert_router_defaults_to_expert_zero_when_no_pointer():
    router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=3)
    for idx, expert in enumerate(router.experts):
        with torch.no_grad():
            expert.weight.zero_()
            expert.bias.fill_(float(idx))

    x = torch.randn(4, 2)
    out = router(x, pointer_addresses=None)
    expected = torch.zeros(4, 1)
    assert torch.allclose(out.cpu(), expected)


def test_location_expert_router_single_expert_ignores_pointer():
    router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=1)
    with torch.no_grad():
        router.single.weight.zero_()
        router.single.bias.fill_(3.14)

    x = torch.randn(3, 2)
    ptr = torch.tensor([0, 1, 2])
    out = router(x, ptr)
    expected = torch.full((3, 1), 3.14)
    assert torch.allclose(out.cpu(), expected)
