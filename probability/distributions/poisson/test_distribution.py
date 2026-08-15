import math
import pytest

from probability.distributions.poisson import Poisson


class TestPoisson:

    # -------------------------
    # Constructor
    # -------------------------

    def test_constructor(self):
        distribution = Poisson(3)

        assert distribution.lam == 3
        assert distribution.mu == 3
        assert distribution.variance == 3

    def test_constructor_different_lambda(self):
        distribution = Poisson(0.5)

        assert distribution.lam == 0.5
        assert distribution.mu == 0.5
        assert distribution.variance == 0.5

    # -------------------------
    # Constructor edge cases
    # -------------------------

    def test_zero_lambda(self):
        with pytest.raises(ValueError):
            Poisson(0)

    def test_negative_lambda(self):
        with pytest.raises(ValueError):
            Poisson(-1)

    # -------------------------
    # PMF
    # -------------------------

    def test_pmf_zero(self):
        distribution = Poisson(3)

        expected = math.exp(-3)

        assert math.isclose(
            distribution.pmf(0),
            expected
        )

    def test_pmf_one(self):
        distribution = Poisson(3)

        expected = 3 * math.exp(-3)

        assert math.isclose(
            distribution.pmf(1),
            expected
        )

    def test_pmf_two(self):
        distribution = Poisson(3)

        expected = (3 ** 2 * math.exp(-3)) / math.factorial(2)

        assert math.isclose(
            distribution.pmf(2),
            expected
        )

    def test_pmf_large_value(self):
        distribution = Poisson(3)

        expected = (
            math.exp(-3)
            * 3 ** 10
            / math.factorial(10)
        )

        assert math.isclose(
            distribution.pmf(10),
            expected
        )

    def test_pmf_negative(self):
        distribution = Poisson(3)

        assert distribution.pmf(-1) == 0

    def test_pmf_non_integer(self):
        distribution = Poisson(3)

        with pytest.raises(TypeError):
            distribution.pmf(1.5)

    # -------------------------
    # CDF
    # -------------------------

    def test_cdf_zero(self):
        distribution = Poisson(3)

        assert math.isclose(
            distribution.cdf(0),
            distribution.pmf(0)
        )

    def test_cdf_one(self):
        distribution = Poisson(3)

        expected = (
            distribution.pmf(0)
            + distribution.pmf(1)
        )

        assert math.isclose(
            distribution.cdf(1),
            expected
        )

    def test_cdf_two(self):
        distribution = Poisson(3)

        expected = sum(
            distribution.pmf(x)
            for x in range(3)
        )

        assert math.isclose(
            distribution.cdf(2),
            expected
        )

    def test_cdf_negative(self):
        distribution = Poisson(3)

        assert distribution.cdf(-1) == 0

    def test_cdf_non_integer(self):
        distribution = Poisson(3)

        with pytest.raises(TypeError):
            distribution.cdf(1.5)

    def test_cdf_approaches_one(self):
        distribution = Poisson(3)

        assert math.isclose(
            distribution.cdf(30),
            1,
            rel_tol=1e-10
        )

    # -------------------------
    # Mathematical properties
    # -------------------------

    def test_pmf_is_non_negative(self):
        distribution = Poisson(3)

        for k in range(20):
            assert distribution.pmf(k) >= 0

    def test_pmf_sums_to_one(self):
        distribution = Poisson(3)

        total = sum(
            distribution.pmf(k)
            for k in range(50)
        )

        assert math.isclose(
            total,
            1,
            rel_tol=1e-10
        )

    def test_cdf_is_between_zero_and_one(self):
        distribution = Poisson(3)

        for k in range(20):
            assert 0 <= distribution.cdf(k) <= 1

    def test_cdf_is_increasing(self):
        distribution = Poisson(3)

        assert distribution.cdf(1) < distribution.cdf(2)
        assert distribution.cdf(2) < distribution.cdf(3)

    # -------------------------
    # Sampling
    # -------------------------

    def test_sample_returns_integer(self):
        distribution = Poisson(3)

        sample = distribution.sample()

        assert isinstance(sample, int)

    def test_sample_is_non_negative(self):
        distribution = Poisson(3)

        for _ in range(1000):
            assert distribution.sample() >= 0

    def test_sample_mean(self):
        distribution = Poisson(3)

        samples = [
            distribution.sample()
            for _ in range(10000)
        ]

        sample_mean = sum(samples) / len(samples)

        assert math.isclose(
            sample_mean,
            distribution.mu,
            rel_tol=0.1
        )

    def test_sample_variance(self):
        distribution = Poisson(3)

        samples = [
            distribution.sample()
            for _ in range(10000)
        ]

        sample_mean = sum(samples) / len(samples)

        sample_variance = sum(
            (x - sample_mean) ** 2
            for x in samples
        ) / len(samples)

        assert math.isclose(
            sample_variance,
            distribution.variance,
            rel_tol=0.1
        )
