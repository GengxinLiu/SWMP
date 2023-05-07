obj_count="3"

dataset="partnet"
obj_cls="Refrigerator"
python mobility2shapemotion.py --dataset $dataset --obj_count $obj_count --obj_cls $obj_cls

dataset="partnet_mobility"
obj_cls="refrigerator"
python mobility2shapemotion.py --dataset $dataset --obj_count $obj_count --obj_cls $obj_cls
