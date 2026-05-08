import unittest

import networkx as nx
import numpy as np

from src.datasets.bipartite_graph import BipartiteGraphDataset


class BipartiteGraphDatasetTest(unittest.TestCase):
    def _dataset(self, **kwargs):
        defaults = {
            "nodes": 6,
            "edge_prob": (0.2, 0.8),
            "n_graphs": 16,
            "train_split_ratio": 0.75,
            "seed": 11,
            "store_graphs": True,
            "small_partition_threshold": 4,
            "max_attempts_per_target": 500,
        }
        defaults.update(kwargs)
        return BipartiteGraphDataset(**defaults)

    def test_reproducibility_and_shape(self):
        first = self._dataset()
        second = self._dataset()

        self.assertEqual(first.all_data.shape, (16, 15))
        self.assertEqual(first.train_data.shape, (12, 15))
        self.assertEqual(first.test_data.shape, (4, 15))
        self.assertEqual(first.data.dtype, np.int8)
        np.testing.assert_array_equal(first.all_data, second.all_data)
        np.testing.assert_array_equal(first.train_data, second.train_data)
        np.testing.assert_array_equal(first.test_data, second.test_data)

    def test_support_graphs_are_bipartite_and_isomorphism_unique(self):
        dataset = self._dataset()

        self.assertEqual(len(dataset._all_graphs), 16)
        self.assertTrue(all(nx.is_bipartite(graph) for graph in dataset._all_graphs))
        for i, graph in enumerate(dataset._all_graphs):
            for other in dataset._all_graphs[i + 1:]:
                self.assertFalse(nx.is_isomorphic(graph, other))

    def test_split_has_no_exact_vector_overlap(self):
        dataset = self._dataset()

        train_patterns = set(map(tuple, dataset.train_data))
        test_patterns = set(map(tuple, dataset.test_data))
        self.assertFalse(train_patterns & test_patterns)

    def test_generate_full_and_sampled_splits(self):
        dataset = self._dataset()

        np.testing.assert_array_equal(dataset.generate(split="test"), dataset.test_data)
        sampled_first = dataset.generate(n_samples=5, seed=123, split="test")
        sampled_second = dataset.generate(n_samples=5, seed=123, split="test")
        np.testing.assert_array_equal(sampled_first, sampled_second)
        self.assertEqual(sampled_first.shape, (5, 15))

    def test_evaluate_generation_tracks_memorization_and_generalization(self):
        dataset = self._dataset()

        train_metrics = dataset.evaluate_generation(dataset.train_data[:3])
        self.assertEqual(train_metrics["bipartite_rate"], 1.0)
        self.assertEqual(train_metrics["density_valid_rate"], 1.0)
        self.assertEqual(train_metrics["memorization_rate"], 1.0)
        self.assertEqual(train_metrics["novel_generalization_rate"], 0.0)

        test_metrics = dataset.evaluate_generation(dataset.test_data[:3])
        self.assertEqual(test_metrics["bipartite_rate"], 1.0)
        self.assertEqual(test_metrics["density_valid_rate"], 1.0)
        self.assertEqual(test_metrics["memorization_rate"], 0.0)
        self.assertEqual(test_metrics["novel_generalization_rate"], 1.0)

    def test_invalid_width_is_counted_without_crashing(self):
        dataset = self._dataset()

        metrics = dataset.evaluate_generation([[1, 0, 1]])

        self.assertEqual(metrics["bipartite_rate"], 0.0)
        self.assertEqual(metrics["invalid_shape_count"], 1)

    def test_raises_clear_error_when_support_cannot_be_reached(self):
        with self.assertRaisesRegex(RuntimeError, "Failed to generate requested"):
            BipartiteGraphDataset(
                nodes=3,
                edge_prob=0.0,
                n_graphs=10,
                max_attempts_per_target=2,
                seed=5,
            )


if __name__ == "__main__":
    unittest.main()
