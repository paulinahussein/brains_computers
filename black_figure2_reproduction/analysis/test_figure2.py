"""Scientific invariants for the Figure 2 pipeline (unittest, CPU by default)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from reproduce_figure2 import (compute_isc, encode, inspect_npy, roi_scores,
                               select_embeddings, DISPLAY_IDS, summarize, MODELS,
                               ROI_NAMES, validate_mapping, ROOT)
from encoding_backend import configure_backend, standardize_train_test


class Figure2Tests(unittest.TestCase):
    def test_scaling_uses_unmodified_training_statistics(self):
        train = np.array([[1., 8.], [3., 8.], [5., 8.]])
        test = np.array([[7., 8.]])
        a, b = standardize_train_test(train, test)
        np.testing.assert_allclose(a[:, 0], [-1, 0, 1])
        np.testing.assert_allclose(b, [[2, 0]])

    def test_truncated_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.npy'
            np.save(path, np.ones((10, 20)))
            self.assertTrue(inspect_npy(path)['valid'])
            path.write_bytes(path.read_bytes()[:-8])
            self.assertFalse(inspect_npy(path)['valid'])

    def test_saved_mapping_has_expected_hemispheres(self):
        mapping = validate_mapping(ROOT / 'Parcel2ROI_new.npy')
        self.assertEqual(mapping.shape, (1000,))

    def test_embedding_selection_preserves_each_delay(self):
        x = np.broadcast_to(np.arange(52).reshape(1, 4, 13, 1), (2, 4, 13, 768)).copy()
        selected = select_embeddings(x.reshape(2, -1), 'paper').reshape(2, 4, 12, 768)
        np.testing.assert_array_equal(selected, x[:, :, 1:, :])

    def test_isc_self_inclusion_and_fold_alignment(self):
        rng = np.random.default_rng(2)
        ys = rng.normal(size=(3, 30, 2))
        actual = compute_isc(ys)
        mean = ys.mean(0)
        for s in range(3):
            for fold, (_, test) in enumerate(KFold(3).split(mean)):
                for p in range(2):
                    self.assertAlmostEqual(actual[s, p, fold],
                                           pearsonr(ys[s, test, p], mean[test, p]).statistic)

    def test_roi_aggregation_precedes_normalization(self):
        mapping = np.repeat(DISPLAY_IDS, 2)
        r = np.tile([[.1]*3, [.3]*3], (10, 1))
        c = np.tile([[.2]*3, [.6]*3], (10, 1))
        expected = 100*np.arctanh([.1, .3]).mean()/np.arctanh([.2, .6]).mean()
        np.testing.assert_allclose(roi_scores(r, c, mapping), expected)
        self.assertNotAlmostEqual(expected, 50)

    def test_kernel_primal_and_subject_batching_equivalence(self):
        rng = np.random.default_rng(3)
        xs = [rng.normal(size=(45, k)).astype('float32') for k in (2, 3, 4, 7)]
        y = (xs[-1][:, :3] + rng.normal(size=(45, 3))*.1).astype('float32')
        backend = configure_backend('numpy')
        primal, _ = encode(xs, y, backend, n_iter=4, solver='primal')
        kernel, _ = encode(xs, y, backend, n_iter=4, solver='kernel')
        separate, _ = encode(xs, y[:, :1], backend, n_iter=4, solver='kernel')
        np.testing.assert_allclose(primal, kernel, atol=1e-4)
        np.testing.assert_allclose(separate, kernel[:1], atol=1e-4)

    def test_nuisance_only_signal_not_used_in_prediction(self):
        rng = np.random.default_rng(9)
        nuisance = rng.normal(size=(45, 2)).astype('float32')
        xs = [nuisance, nuisance[:, :1], nuisance[:, :1], np.zeros((45, 2), dtype='float32')]
        r, _ = encode(xs, nuisance, configure_backend('numpy'), n_iter=3)
        self.assertTrue(np.isnan(r).all())  # Zero main-band prediction has undefined r.

    def test_plot_writes_all_formats_and_summary(self):
        rows = [dict(story=story, subject=f'sub-{s}', model=model, roi=ROI_NAMES[i-1],
                     percent_noise_ceiling=float(s+5))
                for model in MODELS for i in DISPLAY_IDS
                for story in ('black',) for s in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            summarize(rows, Path(folder), n_boot=20)
            for name in ['figure2.png', 'figure2.pdf', 'figure2.svg', 'figure2_summary.csv']:
                self.assertTrue((Path(folder)/name).stat().st_size > 0)


if __name__ == '__main__':
    unittest.main()
