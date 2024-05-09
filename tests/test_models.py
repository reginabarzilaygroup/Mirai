import unittest
import sys
import os
import pickle
import torch
import torch.nn as nn

# append module root directory to sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import onconet.models.factory as mf
import onconet.models.default_models


class Args():
    pass


class TestModels(unittest.TestCase):
    def setUp(self):
        self.args = Args()
        self.args.weight_decay = 5e-5
        self.args.lr = 0.001
        self.model = nn.Linear(10, 2)

    def tearDown(self):
        self.args = None
        self.model = None

    def test_get_existing_optimizers(self):
        args = self.args
        optimizers = [
            ('adam', torch.optim.Adam),
        ]
        for optimizer, optim_type in optimizers:
            args.optimizer = optimizer
            optim = mf.get_optimizer(self.model, args)
            self.assertIsInstance(optim, optim_type)

    def test_non_existing_optimizers(self):
        args = self.args
        optimizers = [
            None,
            'yala',
            5,
        ]
        for optimizer in optimizers:
            args.optimizer = optimizer
            with self.assertRaises(Exception) as context:
                mf.get_optimizer(self.model, args)

            self.assertTrue(
                'Optimizer {} not supported!'.format(optimizer) in str(
                    context.exception))

    def test_calibrator_regression(self):
        """
        Check that our calibrator is the same as the old one.
        """
        new_cal_path = os.path.join(project_dir, "snapshots/calibrators/Mirai_calibrator_mar12_2022.p")
        cal_table_path = os.path.join(project_dir, "tests/Mirai_pred_rf_callibrator_mar12_2022_values.csv")
        new_calibrators = pickle.load(open(new_cal_path, "rb"))

        import numpy as np

        data = np.genfromtxt(cal_table_path, delimiter=',', skip_header=1)
        test_vals = data[:, 0].flatten()
        test_ys = data[:, 1:]
        num_checked = 0
        for key, calibrator in new_calibrators.items():
            myx = test_vals
            new_vals = calibrator.predict_proba(myx)[1, :]
            exp_vals = test_ys[:, key].flatten()

            atol = 1e-12
            assert np.allclose(new_vals, exp_vals, atol=atol), f"key: {key}, old: {exp_vals}, new: {new_vals}"
            num_checked += len(exp_vals)

        assert num_checked == 505, f"Checked {num_checked} values, expected 505."


def _generate_calibration_data():
    # Here for posterity: The calibrators take float as input and output float
    # I'm upgrading the calibrator, want to make sure the outputs are the same.
    old_cal_path = "snapshots_v0.7.0/callibrators/Mirai_pred_rf_callibrator_mar12_2022.p"
    import pickle

    with open(old_cal_path, "rb") as fi:
        old_calibrators = pickle.load(fi)

    import numpy as np
    import pandas as pd

    test_vals = 0.01 * np.arange(0, 101)
    output_df = pd.DataFrame(data={"x": test_vals})
    for key in old_calibrators.keys():
        output_df[key] = 0.
    for key, calibrator in old_calibrators.items():
        myx = test_vals.reshape(-1, 1)
        output_df[key] = calibrator.predict_proba(myx)[:, 1]

    output_df.to_csv("tests/Mirai_pred_rf_callibrator_mar12_2022_values.csv", index=False, header=True)


if __name__ == '__main__':
    unittest.main()
