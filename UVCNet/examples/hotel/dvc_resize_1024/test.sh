ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/512.model --config config.json
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/2048.model --config config.json
# napshot/512.model  EWAP_eth dataset : average bpp : 0.116026, average psnr : 36.754345, average msssim: 0.991936
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter2168.model --config config.json
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter1122.model --config config.json
CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter45254.model --config config.json
# EWAP_eth dataset : average bpp : 0.078758, average psnr : 36.732913, average msssim: 0.991767
# EWAP_eth dataset : average bpp : 0.073200, average psnr : 36.966868, average msssim: 0.991892 gop = 10, ref chen
# EWAP_eth dataset : average bpp : 0.040038, average psnr : 35.987063, average msssim: 0.991892 gop = 25, ref chen
# EWAP_eth dataset : average bpp : 0.038608, average psnr : 37.904256, average msssim: 0.993642 gop = 25, ref h265

# 4096
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log log.txt --testeth --pretrain ../dvc_resize_4096/snapshot/iter11220.model --config config.json
# EWAP_eth dataset : average bpp : 0.135950, average psnr : 37.324983, average msssim: 0.993548
# EWAP_eth dataset : average bpp : 0.183806, average psnr : 37.925755, average msssim: 0.995103




# ../fg_bg_compression/snapshot/
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log log.txt --testeth --pretrain ../fg_bg_compression/snapshot/iter30294.model --config config.json
# EWAP_eth dataset : average bpp : 0.135950, average psnr : 37.324983, average msssim: 0.993548
# EWAP_eth dataset : average bpp : 0.183806, average psnr : 37.925755, average msssim: 0.995103
