import json
import datetime
import logging
import math
import pprint
import traceback
import unittest
import sys
import os
import warnings
import zipfile

import pydicom


from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning, append=True)
warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", message=".*Manufacturer not GE or C-View/VOI LUT doesn't exist.*", append=True)

# append module root directory to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

import onconet

__doc__ = """
End-to-end test. 
Run the model on sample data.
"""


def download_file(url, destination):
    import urllib.request

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        logging.getLogger("mirai_full").error(f"An error occurred while downloading from {url} to {destination}: {e}")
        raise e


class TestPredictionRegression(unittest.TestCase):
    """
    Test that the model predictions are the same as the expected predictions.
    Running this test will be very time consuming, since we need to process so many scans.
    """
    def setUp(self):
        pass

    def test_inference_inbreast(self):
        if not os.environ.get("MIRAI_TEST_RUN_REGRESSION", "false").lower() == "true":
            import pytest
            pytest.skip(f"Skipping long-running test in {type(self)}.")

        import onconet.predict as predict
        import pandas as pd
        import requests

        allow_resume = True
        save_view_tags = True
        max_to_process = os.environ.get("MIRAI_TEST_MAX_TO_PROCESS", "5")
        max_to_process = math.inf if max_to_process == "inf" else int(max_to_process)

        # True ->  send web requests to the ARK server (must be launched separately).
        # False -> to run inference directly.
        use_ark = os.environ.get("MIRAI_TEST_USE_ARK", "false").lower() == "true"

        group_col = "Patient ID"
        filename_col = "Full File Name"
        temp_dir = os.path.join(PROJECT_DIR, "tests/.cache/temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Must download the INBreast dataset first
        # https://www.academicradiology.org/article/S1076-6332(11)00451-X/abstract
        # https://pubmed.ncbi.nlm.nih.gov/22078258/
        # https://www.kaggle.com/datasets/martholi/inbreast
        test_data_dir = os.path.join(PROJECT_DIR, "tests/test_data")
        image_data_dir = os.path.join(test_data_dir, "inbreast", "ALL-IMGS")
        input_table = os.path.join(PROJECT_DIR, "tests/test_data/inbreast_table_v01.tsv")

        version = onconet.__version__
        out_fi_name = f"inbreast_predictions_v{version}.json"
        if use_ark:
            # Query the ARK server to get the version
            resp = requests.get("http://localhost:5000/info")
            version = resp.json()["data"]["apiVersion"]
            out_fi_name = f"inbreast_predictions_ark_v{version}.json"

        cur_pred_results = os.path.join("tests", out_fi_name)

        all_results = {
            "__metadata__": {
                "version": version,
                "start_time": datetime.datetime.now().isoformat(),
                "input_table": input_table,
            }
        }
        if os.path.exists(cur_pred_results):
            if allow_resume:
                with open(cur_pred_results, 'r') as f:
                    all_results = json.load(f)
            else:
                os.remove(cur_pred_results)

        input_df = pd.read_csv(input_table, sep="\t")
        num_patients = input_df[group_col].nunique()
        num_to_process = min(num_patients, max_to_process)

        print(f"About to process {num_to_process} patients with version {version}.\n"
              f"Results will be saved to {cur_pred_results}")

        idx = 0
        for patient_id, group_df in input_df.groupby(group_col):
            if idx >= max_to_process:
                print(f"Reached max_to_process ({max_to_process}), stopping.")
                break

            idx += 1
            print(f"{datetime.datetime.now()} Processing {patient_id} ({idx}/{num_to_process})")
            if patient_id in all_results:
                print(f"Already processed {patient_id}, skipping")
                continue

            dicom_file_names = group_df[filename_col].tolist()
            dicom_file_paths = []
            if save_view_tags:
                for rn, row in group_df.iterrows():
                    dicom_file_name = row[filename_col]
                    dicom_file = os.path.join(image_data_dir, row[filename_col])
                    dicom = pydicom.dcmread(dicom_file)
                    view_str = row['View']
                    side_str = row['Laterality']
                    # view = 0 if view_str == 'CC' else 1
                    # side = 0 if side_str == 'R' else 1

                    dicom.Manufacturer = "GE"  # ???
                    dicom.ViewPosition = view_str
                    dicom.ImageLaterality = side_str
                    new_dicom_file = os.path.join(temp_dir, dicom_file_name.replace(".dcm", f"_resaved.dcm"))
                    dicom.save_as(new_dicom_file)
                    assert os.path.exists(new_dicom_file)
                    dicom_file_paths.append(new_dicom_file)
            else:
                dicom_file_paths = [os.path.join(image_data_dir, f) for f in dicom_file_names]

            prediction = {}

            if use_ark:
                import requests
                # Submit prediction to ARK server.
                files = [('dicom', open(file_path, 'rb')) for file_path in dicom_file_paths]
                r = requests.post("http://localhost:5000/dicom/files", data={"dcmtk": False}, files=files)
                _ = [f[1].close() for f in files]
                if r.status_code != 200:
                    print(f"An error occurred while processing {patient_id}: {r.text}")
                    prediction["error"] = r.text
                    continue
                else:
                    prediction = r.json()["data"]
            else:
                try:
                    prediction = predict.predict(dicom_file_paths, predict.DEFAULT_CONFIG_PATH, use_pydicom=False)
                except Exception as e:
                    print(f"An error occurred while processing {patient_id}: {e}")
                    prediction["error"] = traceback.format_exc()

            cur_dict = {"files": dicom_file_names,
                        group_col: patient_id}
            if prediction:
                cur_dict.update(prediction)

            all_results[patient_id] = cur_dict

            with open(cur_pred_results, 'w') as f:
                json.dump(all_results, f, indent=2)

    def test_compare_inference_scores(self):
        if not os.environ.get("MIRAI_TEST_RUN_REGRESSION", "false").lower() == "true":
            import pytest
            pytest.skip(f"Skipping long-running test in {type(self)}.")

        baseline_preds_path = os.path.join(PROJECT_DIR, "tests", "inbreast_predictions_v0.7.0.json")
        new_preds_path = os.environ.get("MIRAI_TEST_COMPARE_PATH")
        pred_key = "predictions"
        num_compared = 0

        with open(baseline_preds_path, 'r') as f:
            baseline_preds = json.load(f)
        with open(new_preds_path, 'r') as f:
            new_preds = json.load(f)

        ignore_keys = {"__metadata__"}
        overlap_keys = set(baseline_preds.keys()).intersection(new_preds.keys()) - ignore_keys
        union_keys = set(baseline_preds.keys()).union(new_preds.keys()) - ignore_keys
        print(f"{len(overlap_keys)} / {len(union_keys)} patients in common between the two prediction files.")

        for key in overlap_keys:
            if key in ignore_keys:
                continue

            if pred_key not in baseline_preds[key]:
                print(f"{pred_key} not found in baseline predictions for {key}")
                assert pred_key not in new_preds[key]
                # pprint.pprint(baseline_preds[key])
                continue

            cur_baseline_preds = baseline_preds[key]["predictions"]
            cur_new_preds = new_preds[key]["predictions"]
            for year in cur_baseline_preds:
                baseline_score = cur_baseline_preds[year]
                new_score = cur_new_preds[year]
                self.assertAlmostEqual(baseline_score, new_score, delta=0.0001, msg=f"Scores for {key} differ for year {year}. Baseline: {baseline_score}, New: {new_score}")

            num_compared += 1

        assert num_compared > 0
        print(f"Compared {num_compared} patients.")


class TestPredict(unittest.TestCase):
    def setUp(self):
        # Download demo data if it doesn't exist
        self.data_dir = os.path.join(PROJECT_DIR, "mirai_demo_data")
        latest_url = "https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip"
        pegged_url = "https://github.com/reginabarzilaygroup/Mirai/releases/download/v0.8.0/mirai_demo_data.zip"
        if not os.path.exists(self.data_dir):
            if not os.path.exists("mirai_demo_data.zip"):
                download_file(pegged_url, "mirai_demo_data.zip")
            # Unzip file
            with zipfile.ZipFile("mirai_demo_data.zip", 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def test_demo_data_v070(self):
        # Can only unpickle the old calibration file with sklearn 0.23.2
        import sklearn
        data_dir = self.data_dir
        dicom_files = [f"{data_dir}/ccl1.dcm",
                       f"{data_dir}/ccr1.dcm",
                       f"{data_dir}/mlol2.dcm",
                       f"{data_dir}/mlor2.dcm"]

        import onconet.predict as predict

        v07_config_path = os.path.join(predict.config_dir, "mirai_trained_v0.7.0.json")
        prediction = predict.predict(dicom_files, v07_config_path)
        expected_result = {'predictions': {'Year 1': 0.0298, 'Year 2': 0.0483, 'Year 3': 0.0684, 'Year 4': 0.09, 'Year 5': 0.1016}}

        self.assertEqual(prediction, expected_result, "Prediction does not match expected result.")

    def test_demo_data(self):
        data_dir = self.data_dir
        dicom_files = [f"{data_dir}/ccl1.dcm",
                       f"{data_dir}/ccr1.dcm",
                       f"{data_dir}/mlol2.dcm",
                       f"{data_dir}/mlor2.dcm"]

        import scripts.inference as inference

        prediction = inference.predict(dicom_files, inference.DEFAULT_CONFIG_PATH)
        expected_result = {'predictions': {'Year 1': 0.0298, 'Year 2': 0.0483, 'Year 3': 0.0684, 'Year 4': 0.09, 'Year 5': 0.1016}}

        self.assertEqual(prediction, expected_result, "Prediction does not match expected result.")

        # Try again with dicom files in a different order
        dicom_files = [dicom_files[2], dicom_files[3], dicom_files[0], dicom_files[1]]
        prediction = inference.predict(dicom_files, inference.DEFAULT_CONFIG_PATH)
        self.assertEqual(prediction, expected_result, "Prediction does not match expected result in new order.")


if __name__ == '__main__':
    unittest.main()
