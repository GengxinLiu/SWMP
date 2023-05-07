#!/bin/bash
for dataset in "partnet_mobility" "partnet"; do
  for category in 'laptop' 'table' 'refrigerator' 'display' 'chair' 'Trashcan' 'Knife' 'microwave' 'bottle' 'scissors' 'faucet' 'dishwasher' 'storage_furniture' 'door_set'; do
    echo "--------- $dataset/$category ----------"
    python 1_PredictMotionType.py --category $category --dataset $dataset
    python 2_Save2Graphics.py --category $category --dataset $dataset
    python 3_Output2Mobility.py --category $category --dataset $dataset
    python 4_MergeObj.py --category $category --dataset $dataset
    cd OBBcalculation && python 5_ComputeMergeOBB.py && cd ..
    python 6_ProcessOBB.py --dataset $dataset
    python 7_GenCandObbMotion.py --category $category --dataset $dataset
    python 8_AxisSelectMotion.py --category $category --dataset $dataset
  done
done
