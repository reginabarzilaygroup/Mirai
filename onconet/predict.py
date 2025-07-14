#!/usr/bin/env python
import argparse
import io
import json
import logging
import os
import pprint
import sys
from typing import List

try:
    import onconet.utils.dicom
    from onconet.models.mirai_full import MiraiModel
    from onconet.utils import logging_utils
    from onconet import __version__ as onconet_version
except ImportError as e:
    print(f"Error importing onconet modules. Only predictions with ark will be possible.")
    onconet_version = "Onconet not installed"

script_path = os.path.abspath(__file__)
package_dir = os.path.dirname(script_path)
config_dir = os.path.join(package_dir, "configs")
DEFAULT_CONFIG_PATH = os.path.join(config_dir, "mirai_trained.json")


__doc__ = """
Use Mirai to run inference on a single exam.
"""

def _configure_logger(loglevel):
    """
    Configure the logger based on the provided log level.
    """
    if "logging_utils" in sys.modules:
        logging_utils.configure_logger(loglevel)
    else:
        # Fallback to basic logging if logging_utils fails
        logging.basicConfig(level=loglevel)

def _get_logger():
    if "logging_utils" in sys.modules:
        return logging_utils.get_logger()
    else:
        # Fallback to basic logging if logging_utils fails
        return logging.getLogger(__name__)


def _get_parser():
    desc = __doc__ + f"\n\nVersion: {onconet_version}\n"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help="Path to model configuration file.")

    parser.add_argument('--output-path', default=None, dest="output_path",
                        help="Path to save prediction JSON. Prediction will be printed to stdout as well.")

    parser.add_argument('--use-pydicom', default=False, action="store_true",
                        help="Use pydicom instead of dcmtk to read DICOM files.")

    parser.add_argument('--window-method', default="minmax", choices=["minmax", "auto"],
                        help="Windowing method to use for preprocessing with pydicom.")

    parser.add_argument('--dry-run', default=False, action="store_true",
                        help="Load model and configuration, but don't actually do any predictions. "
                             "Useful for checking environment and downloading models.")

    parser.add_argument('--use-ark', default=False, action="store_true",
                        help="Use ARK server for prediction.")

    parser.add_argument('--ark-host', default="localhost", help="Hostname of the ARK server. Default is 'localhost'.")
    parser.add_argument('--ark-port', type=int, default=5000, help="Port of the ARK server. Default is 5000.")

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

def predict_ark(dicom_files: List[str], output_path=None, use_pydicom=False,
                ark_host="localhost", ark_port=5000, window_method='minmax'):
    """
    Run prediction using an ARK server.

    :param dicom_files: List of DICOM file paths.
    :param output_path: Path to save prediction JSON.
    :param use_pydicom: Whether to use pydicom for reading DICOM files.
    :param ark_host: Hostname of the ARK server. Default is "localhost".
    :param ark_port: Port of the ARK server. Default is 5000.
    """

    logger = _get_logger()

    try:
        import requests
    except ImportError:
        logger.error("requests module is not installed. Please install it to use ARK server.")
        return {"error": "requests module is not installed."}

    assert len(dicom_files) == 4, "Expected 4 DICOM files, got {}".format(len(dicom_files))

    try:
        resp = requests.get(f"http://{ark_host}:{ark_port}/info")
        model_info = resp.json()["data"]
    except requests.RequestException as e:
        logger.error(f"Failed to get model info from ARK server: {e}")
        return {"error": f"Failed to get model info from ARK server: {e}"}

    # Submit prediction to ARK server.
    files = [('dicom', open(file_path, 'rb')) for file_path in dicom_files]
    payload = {"dcmtk": not use_pydicom, "window_method": window_method}
    r = requests.post(f"http://{ark_host}:{ark_port}/dicom/files", data=payload, files=files)
    _ = [f[1].close() for f in files]
    model_output_dict = {"info": model_info}
    if r.status_code != 200:
        logger.error(f"An error occurred while processing {dicom_files}: {r.text}")
        model_output_dict["error"] = r.text
    else:
        model_output_dict.update(r.json())

    if output_path is not None:
        logger.info(f"Saving prediction to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(model_output_dict, f, indent=2)

    return model_output_dict

def predict(dicom_files: List[str], config_path: str, output_path=None, use_pydicom=False,
            threads=0, dry_run=False, window_method='minmax') -> dict:
    logger = logging_utils.get_logger()

    config = _load_config(config_path, threads=threads)
    MiraiModel.download_if_needed(config)

    model = MiraiModel(config)
    if dry_run:
        logger.info(f"Model version: {model.__version__}. Dry run complete.")
        return

    assert len(dicom_files) == 4, "Expected 4 DICOM files, got {}".format(len(dicom_files))
    for dicom_file in dicom_files:
        # assert dicom_file.endswith('.dcm'), f"DICOM files must have extension 'dcm'"
        assert os.path.exists(dicom_file), f"File not found: {dicom_file}"

    logger.info(f"Beginning prediction with model {model.__version__}")
    logger.debug(f"Input files: {', '.join(dicom_files)}")

    if not use_pydicom:
        if not onconet.utils.dicom.is_dcmtk_installed():
            logger.warning("DCMTK not found. Using pydicom.")
            use_pydicom = True

    # Load DICOM files into memory
    def load_binary(_dicom_file) -> io.BytesIO:
        with open(_dicom_file, 'rb') as _fi:
            return io.BytesIO(_fi.read())

    dicom_data_list = [load_binary(dicom_file) for dicom_file in dicom_files]
    payload = {"dcmtk": not use_pydicom, "window_method": window_method}
    model_output_dict = model.run_model(dicom_data_list, payload=payload)
    model_output_dict["modelVersion"] = model.__version__

    logger.info(f"Finished prediction version {model.__version__}")
    if output_path is not None:
        logger.info(f"Saving prediction to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(model_output_dict, f, indent=2)

    return model_output_dict


def main():
    args = _get_parser().parse_args()
    _configure_logger(args.loglevel)

    if args.use_ark:
        model_output_dict = predict_ark(args.dicoms, args.output_path, args.use_pydicom,
                                ark_host=args.ark_host, ark_port=args.ark_port,
                                window_method=args.window_method)
    else:
        model_output_dict = predict(args.dicoms, args.config, args.output_path, args.use_pydicom,
                                threads=args.threads, dry_run=args.dry_run)
    if model_output_dict:
        pprint.pprint(model_output_dict)


if __name__ == "__main__":
    main()
