import unittest
import numpy as np
from run import clean_data, sample_data, make_preprocessor, run
from tempfile import TemporaryDirectory


class PipelineTests(unittest.TestCase):
    def test_cleaning_and_idempotency(self):
        cleaned, report = clean_data(sample_data())
        self.assertEqual(report['exact_duplicates_removed'], 5)
        self.assertEqual(len(cleaned), 200)
        self.assertTrue(np.isnan(cleaned.loc[0, 'age']))
        self.assertTrue(np.isnan(cleaned.loc[1, 'sleep_hours']))
        self.assertEqual(cleaned.loc[4, 'activity'], sample_data().loc[4, 'activity'].strip().lower() if sample_data().loc[4, 'activity'] != 'unknown' else np.nan)
        again, _ = clean_data(cleaned)
        self.assertTrue(cleaned.equals(again))

    def test_conflicting_ids_rejected(self):
        raw = sample_data().iloc[:10].copy()
        raw.loc[1, 'record_id'] = raw.loc[0, 'record_id']
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            clean_data(raw)

    def test_no_test_statistics_leak_and_unseen_category(self):
        cleaned, _ = clean_data(sample_data())
        train = cleaned.iloc[10:50].copy()
        test = cleaned.iloc[50:55].copy()
        test['activity'] = 'unseen'
        test['age'] = 100
        processor = make_preprocessor()
        processor.fit(train)
        median = processor.named_transformers_['numeric'].named_steps['impute'].statistics_[0]
        self.assertEqual(median, train.age.median())
        values = processor.transform(test)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(median, processor.named_transformers_['numeric'].named_steps['impute'].statistics_[0])

    def test_end_to_end(self):
        with TemporaryDirectory() as directory:
            report = run(output_dir=directory)
            self.assertEqual(report['train_rows'] + report['test_rows'], 200)


if __name__ == '__main__':
    unittest.main()
