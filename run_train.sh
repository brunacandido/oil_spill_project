#!/bin/bash

# Optional: create virtual environment
# python3 -m venv .venv
# source .venv/bin/activate

# Go to YOLOv5 folder (assumes it is cloned in sibling directory)
cd yolov5 || exit

# Install requirements (if not done)
pip3 install -r requirements.txt

# Start training
python3 train.py \
  --img 640 \
  --batch 8 \
  --epochs 50 \
  --data ../data.yaml \
  --weights yolov5s.pt \
  --project ../yolov5_results \
  --name dartis_oil_detection

# Return to project root
cd ..