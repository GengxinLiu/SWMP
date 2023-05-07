epoch=400
for category in 'laptop' 'table' 'refrigerator' 'display' 'chair' 'mug' 'Trashcan' 'Knife' 'clock' 'microwave' 'bottle' 'scissors' 'faucet' 'dishwasher' 'storage_furniture' 'door_set'; do
  python train_struct.py --category $category --epoch $epoch
done