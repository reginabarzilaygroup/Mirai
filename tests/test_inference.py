import logging
import unittest
import sys
import os
import zipfile

# append module root directory to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

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


class TestInference(unittest.TestCase):
    def setUp(self):
        # Download demo data if it doesn't exist
        self.data_dir = os.path.join(PROJECT_DIR, "mirai_demo_data")
        if not os.path.exists(self.data_dir):
            if not os.path.exists("mirai_demo_data.zip"):
                download_file("https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip",
                              "mirai_demo_data.zip")
            # Unzip file
            with zipfile.ZipFile("mirai_demo_data.zip", 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def test_demo_data(self):
        data_dir = self.data_dir
        dicom_files = [f"{data_dir}/ccl1.dcm",
                       f"{data_dir}/ccr1.dcm",
                       f"{data_dir}/mlol2.dcm",
                       f"{data_dir}/mlor2.dcm"]

        import scripts.inference as inference

        prediction = inference.inference(dicom_files, inference.DEFAULT_CONFIG_PATH)
        expected_result = {'predictions': {'Year 1': 0.0298, 'Year 2': 0.0483, 'Year 3': 0.0684, 'Year 4': 0.09, 'Year 5': 0.1016}}

        self.assertEqual(prediction, expected_result, "Prediction does not match expected result.")


if __name__ == '__main__':
    unittest.main()
