import pytest
import numpy as np

from probability.distributions.uniform import Uniform

class TestUniform:

    # --------------------
    # Constructor
    # --------------------

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Uniform(5, 2)

        with pytest.raises(ValueError):
            Uniform(5, 5)

    # --------------------
    # PDF
    # --------------------

    def test_pdf_inside_range(self):
        distribution = Uniform(2, 8)

        expected = 1 / (8 - 2)

        assert distribution.pdf(2) == expected
        assert distribution.pdf(5) == expected
        assert distribution.pdf(8) == expected

    def test_pdf_outside_range(self):
        distribution = Uniform(2, 8)

        assert distribution.pdf(1) == 0
        assert distribution.pdf(9) == 0

    # --------------------
    # CDF
    # --------------------

    def test_cdf_below_range(self):
        distribution = Uniform(2, 8)

        assert distribution.cdf(1) == 0

    def test_cdf_above_range(self):
        distribution = Uniform(2, 8)

        assert distribution.cdf(9) == 1

    def test_cdf_boundaries(self):
        distribution = Uniform(2, 8)

        assert distribution.cdf(2) == 0
        assert distribution.cdf(8) == 1

    def test_cdf_inside_range(self):
        distribution = Uniform(2, 8)

        assert distribution.cdf(5) == 0.5
        assert distribution.cdf(3) == 1 / 6
        assert distribution.cdf(7) == pytest.approx(5 / 6)

    # --------------------
    # Distribution properties
    # --------------------

    def test_mean(self):
        distribution = Uniform(2, 8)

        assert distribution.mean == 5

    def test_standard_deviation(self):
        distribution = Uniform(2, 8)

        expected = (8 - 2) / np.sqrt(12)

        assert np.isclose(distribution.std, expected)

    def test_probability_density_integrates_to_one(self):
        distribution = Uniform(2, 8)

        width = distribution.b - distribution.a
        area = width * distribution.p

        assert np.isclose(area, 1)

    # --------------------
    # Sampling errors
    # --------------------

    def test_negative_sample_size(self):
        distribution = Uniform(2, 8)

        with pytest.raises(ValueError):
            distribution.sample(-1)

    def test_non_integer_sample_size(self):
        distribution = Uniform(2, 8)

        with pytest.raises(TypeError):
            distribution.sample(10.5)

        with pytest.raises(TypeError):
            distribution.sample("10")

    # --------------------
    # Sampling
    # --------------------

    def test_sample_size(self):
        distribution = Uniform(2, 8)

        samples = distribution.sample(1000)

        assert len(samples) == 1000

    def test_zero_samples(self):
        distribution = Uniform(2, 8)

        samples = distribution.sample(0)

        assert len(samples) == 0

    def test_samples_within_range(self):
        distribution = Uniform(2, 8)

        samples = distribution.sample(10000)

        assert np.all(samples >= 2)
        assert np.all(samples <= 8)

    def test_sample_mean(self):
        distribution = Uniform(2, 8)

        samples = distribution.sample(100000)

        assert np.isclose(
            np.mean(samples),
            distribution.mean,
            atol=0.02
        )

    def test_sample_standard_deviation(self):
        distribution = Uniform(2, 8)

        samples = distribution.sample(100000)

        assert np.isclose(
            np.std(samples),
            distribution.std,
            atol=0.02
        )
