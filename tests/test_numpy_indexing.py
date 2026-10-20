"""src.numpy_indexing 的 unittest 测试。"""

import unittest

import numpy as np

from src.numpy_indexing import (
    broadcast_add,
    clip_array,
    get_center_crop,
    normalize_array,
    pairwise_distances,
    remove_outliers,
    select_by_threshold,
    standardize_array,
)


class NormalizeArrayTests(unittest.TestCase):
    def test_maps_range_to_zero_and_one(self):
        result = normalize_array([2, 4, 6])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_constant_array_becomes_zeros_without_mutation(self):
        original = np.array([5, 5, 5])
        result = normalize_array(original)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(original, [5, 5, 5])

    def test_empty_array_returns_empty_float_array(self):
        result = normalize_array([])
        self.assertEqual(result.size, 0)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            normalize_array([1, np.nan])


class StandardizeArrayTests(unittest.TestCase):
    def test_result_has_population_mean_zero_and_std_one(self):
        result = standardize_array([1, 2, 3])
        self.assertAlmostEqual(float(result.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(result.std(ddof=0)), 1.0, places=12)

    def test_constant_array_becomes_zeros(self):
        np.testing.assert_allclose(standardize_array([7, 7]), [0.0, 0.0])

    def test_empty_array_returns_empty_array(self):
        self.assertEqual(standardize_array([]).size, 0)

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            standardize_array([1, np.inf])


class RemoveOutliersTests(unittest.TestCase):
    def test_removes_large_z_score(self):
        result = remove_outliers([0, 0, 0, 10], threshold=1.5)
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_constant_array_keeps_all_values(self):
        np.testing.assert_allclose(remove_outliers([3, 3, 3]), [3, 3, 3])

    def test_empty_array_returns_empty_array(self):
        self.assertEqual(remove_outliers([]).size, 0)

    def test_invalid_threshold_raises_value_error(self):
        with self.assertRaises(ValueError):
            remove_outliers([1, 2, 3], threshold=-1)


class PairwiseDistancesTests(unittest.TestCase):
    def test_known_distances_are_symmetric_with_zero_diagonal(self):
        result = pairwise_distances([[0, 0], [3, 4], [6, 8]])
        expected = np.array([[0, 5, 10], [5, 0, 5], [10, 5, 0]], dtype=float)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, result.T)
        np.testing.assert_allclose(np.diag(result), 0.0)

    def test_empty_sample_matrix_returns_empty_distance_matrix(self):
        result = pairwise_distances(np.empty((0, 2)))
        self.assertEqual(result.shape, (0, 0))

    def test_non_2d_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            pairwise_distances([1, 2, 3])

    def test_non_finite_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            pairwise_distances([[0, np.nan]])


class SelectByThresholdTests(unittest.TestCase):
    def test_selects_only_values_strictly_greater_than_threshold(self):
        np.testing.assert_allclose(select_by_threshold([1, 2, 3, 4], 2), [3, 4])

    def test_multidimensional_input_is_flattened_before_selection(self):
        result = select_by_threshold([[1, 5], [7, 2]], 4)
        np.testing.assert_allclose(result, [5, 7])

    def test_empty_input_returns_empty_array(self):
        self.assertEqual(select_by_threshold([], 1).size, 0)

    def test_non_finite_threshold_raises_value_error(self):
        with self.assertRaises(ValueError):
            select_by_threshold([1, 2], np.nan)


class ClipArrayTests(unittest.TestCase):
    def test_clips_values_and_does_not_modify_original(self):
        original = np.array([-2, 0, 3, 8])
        result = clip_array(original, 0, 5)
        np.testing.assert_allclose(result, [0, 0, 3, 5])
        np.testing.assert_array_equal(original, [-2, 0, 3, 8])

    def test_keeps_shape_for_2d_input(self):
        result = clip_array([[-1, 2], [7, 4]], 0, 5)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, [[0, 2], [5, 4]])

    def test_reversed_bounds_raise_value_error(self):
        with self.assertRaises(ValueError):
            clip_array([1, 2], 5, 1)


class CenterCropTests(unittest.TestCase):
    def test_crops_2d_grayscale_image(self):
        image = np.arange(25).reshape(5, 5)
        result = get_center_crop(image, 3, 3)
        np.testing.assert_array_equal(result, image[1:4, 1:4])

    def test_crops_3d_color_image_and_preserves_channels(self):
        image = np.arange(4 * 6 * 3).reshape(4, 6, 3)
        result = get_center_crop(image, 2, 4)
        self.assertEqual(result.shape, (2, 4, 3))
        np.testing.assert_array_equal(result, image[1:3, 1:5, :])

    def test_crop_is_copy_not_view(self):
        image = np.arange(16).reshape(4, 4)
        result = get_center_crop(image, 2, 2)
        result[0, 0] = -1
        self.assertNotEqual(image[1, 1], -1)
        self.assertFalse(np.shares_memory(image, result))

    def test_invalid_image_or_crop_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_center_crop(np.zeros((2,)), 1, 1)
        with self.assertRaises(ValueError):
            get_center_crop(np.zeros((3, 3)), 4, 1)


class BroadcastAddTests(unittest.TestCase):
    def test_adds_channel_bias_to_every_pixel(self):
        image = np.zeros((2, 2, 3), dtype=np.float32)
        result = broadcast_add(image, np.array([10, 20, 30]))
        expected = np.array([[[10, 20, 30]]], dtype=np.float32)
        np.testing.assert_allclose(result, np.broadcast_to(expected, (2, 2, 3)))

    def test_adds_row_bias_and_does_not_modify_input(self):
        original = np.array([[1, 2, 3], [4, 5, 6]])
        result = broadcast_add(original, np.array([100, 200, 300]))
        np.testing.assert_array_equal(result, [[101, 202, 303], [104, 205, 306]])
        np.testing.assert_array_equal(original, [[1, 2, 3], [4, 5, 6]])

    def test_incompatible_shapes_raise_clear_value_error(self):
        with self.assertRaisesRegex(ValueError, "无法广播相加"):
            broadcast_add(np.zeros((2, 3, 3)), np.zeros((2,)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
