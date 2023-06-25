ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot

# ../fg_bg_compression/snapshot/
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain ../fg_bg_compression/snapshot/iter30294.model --config config.json
# EWAP_eth dataset : average bpp : 0.135950, average psnr : 37.324983, average msssim: 0.993548
# EWAP_eth dataset : average bpp : 0.183806, average psnr : 37.925755, average msssim: 0.995103



CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain ../fg_bg_compression/snapshot/iter30294.model --config config.json