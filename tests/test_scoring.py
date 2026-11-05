import numpy as np
from src.evaluation.scoring import FeatureErrorNormalizer, aggregate_scores


def _fake_errors():
    rng = np.random.default_rng(0)
    val = rng.gamma(2.0, 0.01, size=(500, 38))
    test = rng.gamma(2.0, 0.01, size=(500, 38))
    test[250:260, 7] += 0.5  # strong anomaly in ONE feature out of 38
    return val, test


def test_normalizer_centers_validation_errors():
    val, _ = _fake_errors()
    z = FeatureErrorNormalizer().fit(val).transform(val)
    assert z.shape == val.shape
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)


def test_max_beats_mean_on_single_feature_anomaly():
    val, test = _fake_errors()
    norm = FeatureErrorNormalizer().fit(val)

    def separation(scores):
        # how far the weakest anomalous window is above the normal ones
        return scores[250:260].min() / np.percentile(scores[:250], 99)

    s_mean = aggregate_scores(test, norm, "mean")
    s_max = aggregate_scores(test, norm, "max")
    assert separation(s_max) > 1.0
    assert separation(s_max) > separation(s_mean)


def test_topk_output_shape():
    val, test = _fake_errors()
    norm = FeatureErrorNormalizer().fit(val)
    assert aggregate_scores(test, norm, "topk", k=5).shape == (500,)