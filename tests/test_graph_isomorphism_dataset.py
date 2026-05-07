import unittest
from math import factorial

import numpy as np

from src.datasets.graph_isomorphism import GraphIsomorphismDataset


class GraphIsomorphismDatasetTest(unittest.TestCase):
    def test_default_dataset_has_expected_support_and_split(self):
        dataset = GraphIsomorphismDataset()

        self.assertEqual(dataset.all_data.shape, (factorial(8), 28))
        self.assertEqual(dataset.train_data.shape, (5040, 28))
        self.assertEqual(dataset.test_data.shape, (35280, 28))
        self.assertEqual(len(dataset), 5040)
        self.assertEqual(dataset.data.dtype, np.int8)

        train_patterns = set(map(tuple, dataset.train_data))
        test_patterns = set(map(tuple, dataset.test_data))
        self.assertEqual(len(train_patterns), 5040)
        self.assertEqual(len(test_patterns), 35280)
        self.assertFalse(train_patterns & test_patterns)

    def test_generation_is_reproducible(self):
        first = GraphIsomorphismDataset(seed=42)
        second = GraphIsomorphismDataset(seed=42)

        np.testing.assert_array_equal(first.seed_adjacency, second.seed_adjacency)
        np.testing.assert_array_equal(first.train_data, second.train_data)
        np.testing.assert_array_equal(first.test_data, second.test_data)

    def test_evaluate_generation_tracks_memorization_and_generalization(self):
        dataset = GraphIsomorphismDataset()

        train_metrics = dataset.evaluate_generation(dataset.train_data[:3])
        self.assertEqual(train_metrics["valid_graph_rate"], 1.0)
        self.assertEqual(train_metrics["isomorphic_rate"], 1.0)
        self.assertEqual(train_metrics["memorization_rate"], 1.0)
        self.assertEqual(train_metrics["novel_generalization_rate"], 0.0)

        test_metrics = dataset.evaluate_generation(dataset.test_data[:3])
        self.assertEqual(test_metrics["valid_graph_rate"], 1.0)
        self.assertEqual(test_metrics["isomorphic_rate"], 1.0)
        self.assertEqual(test_metrics["memorization_rate"], 0.0)
        self.assertEqual(test_metrics["novel_generalization_rate"], 1.0)

    def test_invalid_width_is_counted_without_crashing(self):
        dataset = GraphIsomorphismDataset(num_nodes=6)

        metrics = dataset.evaluate_generation([[1, 0, 1]])

        self.assertEqual(metrics["valid_graph_rate"], 0.0)
        self.assertEqual(metrics["isomorphic_rate"], 0.0)
        self.assertEqual(metrics["invalid_shape_count"], 1)


if __name__ == "__main__":
    unittest.main()
