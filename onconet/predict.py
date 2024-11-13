#!/usr/bin/env python
import argparse
import csv
import io
import json
import os
import pprint
from typing import List

import tqdm

import onconet.utils.dicom
from onconet.models.mirai_full import MiraiModel
from onconet.utils import logging_utils
from onconet import __version__ as onconet_version

script_path = os.path.abspath(__file__)
package_dir = os.path.dirname(script_path)
config_dir = os.path.join(package_dir, "configs")
DEFAULT_CONFIG_PATH = os.path.join(config_dir, "mirai_trained.json")


__doc__ = """
Use Mirai to run inference on a single exam.
"""


def _get_parser():
    desc = __doc__ + f"\n\nVersion: {onconet_version}\n"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help="Path to model configuration file.")

    parser.add_argument('--output-path', default=None, dest="output_path",
                        help="Path to save prediction JSON. Prediction will be printed to stdout as well.")

    parser.add_argument('--table', default=None, type=str,
                        help="Process multiple scans at once. Input text table of paths here.")

    parser.add_argument('--use-dcmtk', default=False, action="store_true",
                        help="Use dcmtk to read DICOM files. Default is to use pydicom.")

    parser.add_argument('--window-method', default="minmax", choices=["minmax", "auto"],
                        help="Windowing method to use for preprocessing with pydicom.")

    parser.add_argument('--dry-run', default=False, action="store_true",
                        help="Load model and configuration, but don't actually do any predictions. "
                             "Useful for checking environment and downloading models.")

    parser.add_argument('--threads', type=int, default=0,
                        help="Number of threads to use for PyTorch inference. "
                             "Default is 0 (use all available cores). "
                             "Set to a negative number to use Pytorch default. ")

    parser.add_argument('-l', '--log', '--loglevel', '--log-level',
                        default="INFO", dest="loglevel")

    parser.add_argument('--version', action='version', version=onconet_version)
    parser.add_argument('dicoms', nargs="*", help="Path to DICOM files (from a single exam) to run inference on.")

    return parser


def _load_config(config_path, **kwargs):
    with open(config_path, 'r') as f:
        config = json.load(f)
        config.update(kwargs)
    args = argparse.Namespace(**config)
    args = MiraiModel.sanitize_paths(args)
    return args

# Load DICOM files into memory
def _load_binary(_dicom_file) -> io.BytesIO:
    _dicom_file = os.path.expanduser(_dicom_file)
    with open(_dicom_file, 'rb') as _fi:
        return io.BytesIO(_fi.read())

def _predict_single(model, dicom_files, use_dcmtk, window_method):
    dicom_files = list(map(os.path.expanduser, dicom_files))
    logger = logging_utils.get_logger()
    assert len(dicom_files) == 4, "Expected 4 DICOM files, got {}".format(len(dicom_files))
    for dicom_file in dicom_files:
        # assert dicom_file.endswith('.dcm'), f"DICOM files must have extension 'dcm'"
        assert os.path.exists(dicom_file), f"File not found: {dicom_file}"

    if use_dcmtk:
        if not onconet.utils.dicom.is_dcmtk_installed():
            logger.warning("DCMTK not found. Using pydicom.")
            use_dcmtk = False

    dicom_data_list = [_load_binary(dicom_file) for dicom_file in dicom_files]
    payload = {"dcmtk": use_dcmtk, "window_method": window_method}
    model_output_dict = model.run_model(dicom_data_list, payload=payload)
    model_output_dict["modelVersion"] = model.__version__
    return model_output_dict


def predict(dicom_files: List[str], config_path: str, output_path=None,
            table=None, use_dcmtk=False,
            threads=0, dry_run=False, window_method='minmax') -> dict:
    logger = logging_utils.get_logger()

    config = _load_config(config_path, threads=threads)
    MiraiModel.download_if_needed(config)

    model = MiraiModel(config)
    if dry_run:
        logger.info(f"Model version: {model.__version__}. Dry run complete.")
        return

    logger.info(f"Beginning prediction with model {model.__version__}")
    if table is not None:
        # Predict on multiple exams
        all_outputs = []
        # Write to output path in CSV format
        if output_path is None:
            output_path = f"{table}.predictions.csv"
        path_columns = ["path1", "path2", "path3", "path4"]
        logger.debug(f"Reading table from {table}. Path columns are {path_columns}")
        for row in tqdm.tqdm(csv.DictReader(open(table))):
            dicom_files = [row[col] for col in path_columns]
            output_row = {**row}

            try:
                cur_model_output_dict = _predict_single(model, dicom_files, use_dcmtk, window_method)
                for key, val in cur_model_output_dict["predictions"].items():
                    output_row[key] = val
            except Exception as e:
                logger.error(f"Error processing row {row}: {e}")

            all_outputs.append(output_row)

        if len(all_outputs) == 0:
            logger.warning("Table empty; No rows processed.")
            return

        if output_path is not None:
            logger.info(f"Saving prediction to {output_path}")
            fieldnames = all_outputs[0].keys()
            with open(output_path, 'w') as f:
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(all_outputs)
    else:
        # Predict on a single exam
        logger.debug(f"Input files: {', '.join(dicom_files)}")
        all_outputs = _predict_single(model, dicom_files, use_dcmtk, window_method)

        logger.info(f"Finished prediction version {model.__version__}")
        if output_path is not None:
            logger.info(f"Saving prediction to {output_path}")
            with open(output_path, 'w') as f:
                json.dump(all_outputs, f, indent=2)

    return all_outputs


def main():
    args = _get_parser().parse_args()
    logging_utils.configure_logger(args.loglevel)

    model_outputs = predict(args.dicoms, args.config, args.output_path,
                                args.table, args.use_dcmtk,
                                threads=args.threads, dry_run=args.dry_run)

    # Print if it's a single prediction
    if model_outputs and isinstance(model_outputs, dict):
        pprint.pprint(model_outputs)


if __name__ == "__main__":
    main()
