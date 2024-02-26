#!/bin/bash

# Run inference on the demo data
# The output will be printed to the console
# PyDicom is easier to install, (it's included in the requirements.txt),
# you can use it by adding "--use-pydicom" to the command below

# If you don't have dcmtk installed, you can install it with the following commands:
# Linux: sudo apt-get install dcmtk
# Mac with Homebrew: brew install dcmtk

demo_scan_dir=mirai_demo_data

# Download the demo data if it doesn't exist
if [ ! -d "$demo_scan_dir" ]; then
  wget -L wget -L https://github.com/reginabarzilaygroup/Mirai/releases/latest/download/mirai_demo_data.zip
  unzip mirai_demo_data.zip -d "$demo_scan_dir"
fi

python3 scripts/inference.py --config configs/mirai_trained.json \
--loglevel INFO \
--output-path demo_prediction.json \
--use-pydicom \
${demo_scan_dir}/ccl1.dcm ${demo_scan_dir}/ccr1.dcm ${demo_scan_dir}/mlol2.dcm ${demo_scan_dir}/mlor2.dcm