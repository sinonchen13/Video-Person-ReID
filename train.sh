# train
# model : me, multiloss, coordatt, coordatt_me_multiloss, baseline 

# MARS
python train.py  --arch coordatt_me_multiloss  --gpu 0  --save_dir ./Mars_integrate

# Duke
# python train.py  -d duke  --arch coordatt_me_multiloss  --gpu 0  --save_dir ./DukeV_integrate

# LSVID
# python train.py  -d LSVID  --arch coordatt_me_multiloss --gpu 0   --save_dir ./LSVID_integrate

# ilidsvid
# python train.py  -d ilidsvid  --arch coordatt_me_multiloss --gpu 0,1  --save_dir ./iLIDS_integrate

# prid
# python train.py  -d prid  --arch coordatt_me_multiloss --gpu 2,3     --save_dir ./prid_integrate
