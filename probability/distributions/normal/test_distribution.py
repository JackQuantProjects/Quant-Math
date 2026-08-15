import math
import pytest
import numpy as np

from probability.distributions.normal import Normal


class TestNormal:

    # --------------------
    # Constructor
    # --------------------

    def test_constructor(self):
        distribution = Normal(0, 1)

        assert distribution.mu == 0
        assert distribution.sigma == 1
        assert distribution.variance == 1

    def test_negative_sigma(self):
        with pytest.raises(ValueError):
            Normal(0, -1)

    def test_zero_sigma(self):
        with pytest.raises(ValueError):
            Normal(0, 0)

    # --------------------
    # PDF
    # --------------------

    def test_pdf_at_mean(self):
        distribution = Normal(0, 1)

        expected = 1 / math.sqrt(2 * math.pi)

        assert distribution.pdf(0) == pytest.approx(expected)

    def test_pdf_one_std_above_mean(self):
        distribution = Normal(0, 1)

        expected = (
            1 / math.sqrt(2 * math.pi)
            * math.exp(-0.5)
        )

        assert distribution.pdf(1) == pytest.approx(expected)

    def test_pdf_symmetry(self):
        distribution = Normal(0, 1)

        assert distribution.pdf(-1) == pytest.approx(
            distribution.pdf(1)
        )

    def test_pdf_non_standard_normal(self):
        distribution = Normal(10, 2)

        expected = 1 / (2 * math.sqrt(2 * math.pi))

        assert distribution.pdf(10) == pytest.approx(expected)

    # --------------------
    # CDF
    # --------------------

    def test_cdf_at_mean(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(0) == pytest.approx(0.5)

    def test_cdf_one_std_above_mean(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(1) == pytest.approx(
            0.841344746
        )

    def test_cdf_one_std_below_mean(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(-1) == pytest.approx(
            0.158655254
        )

    def test_cdf_symmetry(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(-1) == pytest.approx(
            1 - distribution.cdf(1)
        )

    def test_cdf_non_standard_normal(self):
        distribution = Normal(10, 2)

        assert distribution.cdf(10) == pytest.approx(0.5)

    def test_cdf_far_below(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(-100) == pytest.approx(0)

    def test_cdf_far_above(self):
        distribution = Normal(0, 1)

        assert distribution.cdf(100) == pytest.approx(1)

    # --------------------
    # Sampling
    # --------------------

    def test_sample_negative(self):
        distribution = Normal(0, 1)

        with pytest.raises(ValueError):
            distribution.sample(-1)

    def test_sample_non_integer(self):
        distribution = Normal(0, 1)

        with pytest.raises(TypeError):
            distribution.sample(2.5)

    def test_sample_zero(self):
        distribution = Normal(0, 1)

        samples = distribution.sample(0)

        assert len(samples) == 0

    def test_sample_size(self):
        distribution = Normal(0, 1)

        samples = distribution.sample(100)

        assert len(samples) == 100

    def test_sample_type(self):
        distribution = Normal(0, 1)

        samples = distribution.sample(100)

        assert isinstance(samples, np.ndarray)

    def test_sample_mean_and_std(self):
        distribution = Normal(10, 2)

        samples = distribution.sample(100000)

        assert np.mean(samples) == pytest.approx(
            10,
            abs=0.05
        )

        assert np.std(samples) == pytest.approx(
            2,
            abs=0.05
        )
