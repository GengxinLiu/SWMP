epoch=4000
for category in 'laptop' 'table' 'refrigerator' 'display' 'chair' 'Trashcan' 'Knife' 'clock' 'microwave' 'bottle' 'scissors' 'faucet' 'dishwasher' 'storage_furniture' 'door_set'; do
  python train.py --category $category --epoch $epoch
done