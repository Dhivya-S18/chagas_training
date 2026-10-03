import unittest

import numpy as np

from inference import InferenceService, validate_sequence
from preprocessing import load_dataset


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = InferenceService.from_artifacts()
        features, _ = load_dataset("signals_features.csv")
        cls.sequence = features.iloc[:10].values.tolist()

    def test_valid_dataset_sequence_predicts(self):
        result = self.service.predict(self.sequence)
        self.assertIsNone(result["prediction"]["class"])
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_invalid_row_counts_are_rejected(self):
        for sequence in (self.sequence[:9], self.sequence + [self.sequence[-1]]):
            with self.subTest(row_count=len(sequence)):
                with self.assertRaisesRegex(ValueError, "exactly 10 rows"):
                    validate_sequence(sequence, self.service.feature_names, 10)

    def test_invalid_feature_counts_are_rejected(self):
        for width in (48, 50):
            sequence = [row[:width] for row in self.sequence]
            if width == 50:
                sequence = [row + [0.0] for row in self.sequence]
            with self.subTest(feature_count=width):
                with self.assertRaisesRegex(ValueError, "exactly 49 features"):
                    validate_sequence(sequence, self.service.feature_names, 10)

    def test_named_features_require_schema_order(self):
        row = dict(zip(self.service.feature_names, self.sequence[0]))
        reversed_row = dict(reversed(list(row.items())))
        with self.assertRaisesRegex(ValueError, "feature order"):
            validate_sequence(
                [reversed_row] * 10, self.service.feature_names, 10
            )

        missing_row = dict(list(row.items())[1:])
        with self.assertRaisesRegex(ValueError, "missing required features"):
            validate_sequence(
                [missing_row] * 10, self.service.feature_names, 10
            )

        extra_row = dict(row)
        extra_row["unexpected"] = 0.0
        with self.assertRaisesRegex(ValueError, "unexpected features"):
            validate_sequence([extra_row] * 10, self.service.feature_names, 10)

    def test_non_numeric_and_infinite_values_are_rejected(self):
        non_numeric = [list(row) for row in self.sequence]
        non_numeric[0][0] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_sequence(non_numeric, self.service.feature_names, 10)

        infinite = [list(row) for row in self.sequence]
        infinite[0][0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            validate_sequence(infinite, self.service.feature_names, 10)


if __name__ == "__main__":
    unittest.main()