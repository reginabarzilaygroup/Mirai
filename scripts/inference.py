#!/usr/bin/env python
import argparse
import io
import json
import logging
import os
from typing import List

from onconet.models.mirai_full import MiraiModel

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_dir = os.path.dirname(script_dir)
config_dir = os.path.join(project_dir, "configs")
DEFAULT_CONFIG_PATH = os.path.join(config_dir, "mirai_trained.json")


__doc__ = """
Script for running inference on a single exam using a trained Mirai model.
"""


def _get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help="Path to model configuration file.")
    parser.add_argument('--output-path', default=None, dest="output_path",
                        help="Path to save prediction JSON. Prediction will be printed to stdout as well.")
    parser.add_argument('-l', '--log', '--loglevel', default="INFO", dest="loglevel")
    parser.add_argument('--use-pydicom', default=False, action="store_true",
                        help="Use pydicom instead of dcmtk to read DICOM files.")
    parser.add_argument('dicoms', nargs="+", help="Path to DICOM files (from a single exam) to run inference on.")
    return parser


def inference(dicom_files: List[str], config_path: str, output_path=None, use_pydicom=False):
    logger = logging.getLogger('inference')

    assert len(dicom_files) == 4, "Expected 4 DICOM files, got {}".format(len(dicom_files))
    for dicom_file in dicom_files:
        assert dicom_file.endswith('.dcm'), f"DICOM files must have extension 'dcm'"
        assert os.path.exists(dicom_file), f"File not found: {dicom_file}"

    with open(config_path, 'r') as f:
        config = json.load(f)
        # Convert from JSON dict to argparse.Namespace
        config = argparse.Namespace(**config)

    model = MiraiModel(config)

    logger.info(f"Beginning inference version {model.__version__}")

    # Load DICOM files into memory
    def load_binary(_dicom_file) -> io.BytesIO:
        with open(_dicom_file, 'rb') as _fi:
            return io.BytesIO(_fi.read())

    dicom_data_list = [load_binary(dicom_file) for dicom_file in dicom_files]
    payload = {"dcmtk": not use_pydicom}
    prediction = model.run_model(dicom_data_list, payload=payload)

    print(f"{prediction}")
    if output_path is not None:
        logger.info(f"Saving prediction to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)


def logging_basic_config(args):
    info_fmt = "[%(asctime)s] - %(message)s"
    debug_fmt = "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
    fmt = debug_fmt if args.loglevel.upper() == "DEBUG" else info_fmt

    logging.basicConfig(format=fmt,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=args.loglevel.upper())


def main():
    args = _get_parser().parse_args()
    logging_basic_config(args)

    inference(args.dicoms, args.config, args.output_path, args.use_pydicom)


if __name__ == "__main__":
    main()
