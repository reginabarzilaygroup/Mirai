#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
import subprocess
from typing import List

script_path = os.path.abspath(__file__)
package_dir = os.path.dirname(script_path)
config_dir = os.path.join(package_dir, "configs")
DEFAULT_CONFIG_PATH = os.path.join(config_dir, "mirai_trained.json")

def find_mammogram_dirs(base_dir, ext="dcm"):
    mammogram_dirs = []
    for root, dirs, files in os.walk(base_dir):
        dicom_files = [f for f in files if f.lower().endswith(f'.{ext}') and not f.startswith('.')]
        if len(dicom_files) == 4:
            full_paths = [os.path.join(root, f) for f in dicom_files]
            mammogram_dirs.append(full_paths)
    return mammogram_dirs

def _configure_logger(loglevel):
    try:
        from onconet.utils import logging_utils
        logging_utils.configure_logger(loglevel)
    except ImportError:
        import logging
        logging.basicConfig(level=loglevel)

def _get_logger():
    try:
        from onconet.utils import logging_utils
        return logging_utils.get_logger()
    except ImportError:
        import logging
        return logging.getLogger("predict_bulk")

def build_dicom_table(base_dir, output_csv, ext):
    logger = _get_logger()
    mammogram_dirs = find_mammogram_dirs(base_dir, ext)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file1', 'file2', 'file3', 'file4'])
        for files in mammogram_dirs:
            writer.writerow(files)
    logger.info(f"Wrote {len(mammogram_dirs)} rows to {output_csv}")

def run_predict_py(dicom_files: List[str], args, output_path):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'predict.py')]
    cmd += dicom_files
    cmd += ['--threads', str(args.threads), '--loglevel', args.loglevel]
    if args.config:
        cmd += ['--config', args.config]
    if args.use_ark:
        cmd += ['--use-ark', '--ark-host', args.ark_host, '--ark-port', str(args.ark_port)]
    if args.use_pydicom:
        cmd += ['--use-pydicom']
    if args.window_method:
        cmd += ['--window-method', args.window_method]
    cmd += ['--output-path', output_path]
    subprocess.run(cmd, check=True)

def predict_bulk(table_path, args):
    logger = _get_logger()
    results = []
    with open(table_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dicom_files = [row[f'file{i+1}'] for i in range(4)]
            mammogram_dir = os.path.dirname(dicom_files[0])
            output_path = os.path.join(mammogram_dir, 'predictions.json')
            if not os.path.exists(output_path):
                logger.info(f"Running prediction for {mammogram_dir}...")
                run_predict_py(dicom_files, args, output_path)
            else:
                logger.info(f"Prediction already exists for {mammogram_dir}, loading from file.")
            with open(output_path, 'r') as f:
                prediction = json.load(f)

            # Format is slightly different depending on whether it's from predict.py or Ark server
            if "data" in prediction and "predictions" in prediction["data"]:
                prediction = prediction["data"]
            # Collect scores for CSV
            row_result = {f'file{i+1}': dicom_files[i] for i in range(4)}
            if 'predictions' in prediction:
                for year, score in prediction['predictions'].items():
                    row_result[year] = score
            results.append(row_result)

    logger.info(f"Processed {len(results)} predictions from {table_path}. Saving to {args.score_table}")
    write_results_to_csv(results, args.score_table)

def write_results_to_csv(results, output_path):
    # Write results to score-table CSV
    if results:
        fieldnames = [f'file{i+1}' for i in range(4)]
        # Find all year columns
        year_cols = []
        for r in results:
            for k in r.keys():
                if k.lower().startswith('year') and k not in year_cols:
                    year_cols.append(k)
        fieldnames += sorted(year_cols)
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Bulk prediction and DICOM table builder.")
    parser.add_argument('-l', '--log-level', '--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.', dest="loglevel")
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_parser = subparsers.add_parser('build-file-table', help='Build CSV table of mammogram DICOMs.')
    build_parser.add_argument('--base-dir', required=True, help='Base directory to search for mammogram DICOMs.')
    build_parser.add_argument('--file-table', required=True, help='Output CSV table with DICOM file paths.')
    build_parser.add_argument('--ext', default="dcm", help='File extension to search for DICOM files (default: dcm).')

    predict_parser = subparsers.add_parser('predict-bulk', help='Run bulk predictions from CSV table.')
    predict_parser.add_argument('--file-table', required=True, help='CSV table with DICOM file paths.')
    predict_parser.add_argument('--score-table', required=False, default="predict_bulk_scores.csv", help='CSV table with returned model scores.')
    predict_parser.add_argument('--config', required=False, default=None, help='Path to model configuration file. If not provided, uses default config from predict.py (or Ark server).')
    predict_parser.add_argument('--threads', type=int, default=0)
    predict_parser.add_argument('--use-ark', action='store_true')
    predict_parser.add_argument('--ark-host', default='localhost')
    predict_parser.add_argument('--ark-port', type=int, default=5000)
    predict_parser.add_argument('--use-pydicom', action='store_true')
    predict_parser.add_argument('--window-method', default='minmax', choices=['minmax', 'auto'])

    args = parser.parse_args()

    _configure_logger(args.loglevel)

    if args.command == 'build-file-table':
        build_dicom_table(args.base_dir, args.file_table, args.ext)
    elif args.command == 'predict-bulk':
        predict_bulk(args.file_table, args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
