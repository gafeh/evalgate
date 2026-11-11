from evalgate.judge.pricing import PRICES, cost_usd, is_priced


def test_cost_scales_with_tokens() -> None:
    model = "claude-haiku-5-5"
    price = PRICES[model]
    expected = (1_000_000 * price.input_per_mtok_usd) / 1_000_000
    assert cost_usd(model, 1_000_000, 0) == expected


def test_input_and_output_are_priced_separately() -> None:
    model = "claude-sonnet-5-5"
    only_input = cost_usd(model, 1000, 0)
    only_output = cost_usd(model, 0, 1000)
    assert only_output > only_input
    assert cost_usd(model, 1000, 1000) == only_input + only_output


def test_unknown_model_costs_zero_rather_than_crashing() -> None:
    assert is_priced("some-future-model") is False
    assert cost_usd("some-future-model", 1000, 1000) == 0.0