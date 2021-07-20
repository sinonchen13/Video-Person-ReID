# test
# model : me, multiloss, coordatt, coordatt_me_multiloss, baseline
# Mars
python test-all.py   --arch coordatt_me_multiloss --gpu 0 --resume Mars_integrate   --test_epochs  240  --test_sample_mode test_all_sampled

# Duke 
# python test-all.py  -d duke  --arch coordatt_me_multiloss --gpu 0 --resume DukeV_integrate   --test_epochs  240 --test_sample_mode test_all_sampled

# LSVID
# python test-all.py  -d LSVID  --arch coordatt_me_multiloss --gpu 0 --resume LSVID_integrate   --test_epochs  240 --test_sample_mode test_all_sampled

# ilidsvid
# python test-all.py  -d ilidsvid  --arch coordatt_me_multiloss --gpu 0 --resume iLIDS_integrate   --test_epochs 240 --test_sample_mode test_all_sampled

# prid
# python test-all.py  -d prid  --arch coordatt_me_multiloss --gpu 0 --resume prid_integrate       --test_epochs 240 --test_sample_mode test_all_sampled

# test-vis
# Mars
python test-vis.py   --arch coordatt_me_multiloss --gpu 0 --resume Mars_integrate   --test_epochs  240  --test_sample_mode rrs0