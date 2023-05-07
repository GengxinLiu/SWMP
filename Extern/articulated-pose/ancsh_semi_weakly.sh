category="refrigerator"
name_data="partnet"
# >>>>>>>>>>>>>>>>>>>>>>>> train
python main.py --name_data=$name_data --item=$category --nocs_type='ancsh' --gpu='0'
python main.py --name_data=$name_data --item=$category --nocs_type='npcs' --gpu='1'

# >>>>>>>>>>>>>>>>>>>>>>>> predict
python main.py --name_data=$name_data --item=$category --nocs_type='ancsh' --test --gpu='2'
python main.py --name_data=$name_data --item=$category --nocs_type='npcs' --test --gpu='3'

# >>>>>>>>>>>>>>>>>>>>>>>> evaluation
cd evaluation && python compute_gt_pose.py --item=$category --domain='unseen' --nocs='ANCSH' --save
cd evaluation && python compute_gt_pose.py --item=$category --domain='unseen' --nocs='NAOCS' --save

# run baseline
python baseline_npcs.py --item=$category --domain='unseen' --nocs='ANCSH'

# run our processing over test group
python pose_multi_process.py --item=$category --domain='unseen'

# pose & relative joint rotation
python eval_pose_err.py --item=$category --domain='unseen' --nocs='ANCSH'

# 3d miou estimation
python compute_miou.py --item=$category --domain='unseen' --nocs='ANCSH'

# performance on joint estimations
python eval_joint_params.py --item=$category --domain='unseen' --nocs='ANCSH' > log.txt 2>&1 &
