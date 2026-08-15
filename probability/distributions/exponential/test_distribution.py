import math
import pytest

from probability.distributions.exponential import Exponential


class TestExponential:

    # -------------------------
    # Constructor
    # -------------------------

    def test_constructor(self):
        distribution = Exponential(2)

        assert distribution.lam == 2
        assert distribution.mu == 0.5
        assert distribution.variance == 0.25

    def test_constructor_different_lambda(self):
        distribution = Exponential(0.5)

        assert distribution.lam == 0.5
        assert distribution.mu == 2
        assert distribution.variance == 4

    # -------------------------
    # Constructor edge cases
    # -------------------------

    def test_zero_lambda(self):
        with pytest.raises(ValueError):
            Exponential(0)

    def test_negative_lambda(self):
        with pytest.raises(ValueError):
            Exponential(-1)

    # -------------------------
    # PDF
    # -------------------------

    def test_pdf_at_zero(self):
        distribution = Exponential(2)

        assert distribution.pdf(0) == 2

    def test_pdf_positive_value(self):
        distribution = Exponential(2)

        expected = 2 * math.exp(-4)

        assert math.isclose(
            distribution.pdf(2),
            expected
        )

    def test_pdf_negative_value(self):
        distribution = Exponential(2)

        assert distribution.pdf(-1) == 0

    # -------------------------
    # CDF
    # -------------------------

    def test_cdf_at_zero(self):
        distribution = Exponential(2)

        assert distribution.cdf(0) == 0

    def test_cdf_positive_value(self):
        distribution = Exponential(2)

        expected = 1 - math.exp(-4)

        assert math.isclose(
            distribution.cdf(2),
            expected
        )

    def test_cdf_negative_value(self):
        distribution = Exponential(2)

        assert distribution.cdf(-1) == 0

    def test_cdf_approaches_one(self):
        distribution = Exponential(2)

        assert math.isclose(
            distribution.cdf(100),
            1,
            rel_tol=1e-10
        )

    # -------------------------
    # Mathematical properties
    # -------------------------

    def test_pdf_is_positive(self):
        distribution = Exponential(2)

        assert distribution.pdf(1) > 0
        assert distribution.pdf(10) > 0

    def test_cdf_is_between_zero_and_one(self):
        distribution = Exponential(2)

        for x in [0, 0.1, 1, 5, 10]:
            assert 0 <= distribution.cdf(x) <= 1

    def test_cdf_is_increasing(self):
        distribution = Exponential(2)

        assert distribution.cdf(1) < distribution.cdf(2)
        assert distribution.cdf(2) < distribution.cdf(3)

    # -------------------------
    # Sampling
    # -------------------------

    def test_sample_returns_number(self):
        distribution = Exponential(2)

        sample = distribution.sample()

        assert isinstance(sample, float)

    def test_sample_is_non_negative(self):
        distribution = Exponential(2)

        for _ in range(1000):
            assert distribution.sample() >= 0

    def test_sample_mean(self):
        distribution = Exponential(2)

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
        distribution = Exponential(2)

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
