for dataset in "partnet_mobility" "partnet"; do
  echo "---- $dataset ----"
  python 1_ObjPreprocess.py --dataset $dataset
  cd OBBcalculation && python 2_ComputeOBB.py && cd ..
  python 3_GraphicsDataExtract.py --dataset $dataset
  python 4_AdjustObbOrder.py --dataset $dataset
  python 5_PcsegDataExtract.py --dataset $dataset
  python 6_GraphDataExtract.py --dataset $dataset
  python 7_TrainTestSplit.py --dataset $dataset
done
python 8_GetSemantic.py
