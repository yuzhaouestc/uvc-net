ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter20272.model --config config.json
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log log.txt --testeth --pretrain /ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/examples/fg_bg_compression/snapshot/iter16456.model --config config.json
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log log.txt --testhotel --pretrain snapshot/2048.model --config config.json
# CUDA_VISIBLE_DEVICES=2  python -u $ROOT/main.py --log log.txt --testcuhk --pretrain snapshot/2048.model --config config.json
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/dvc_pretrain2048.model --config config.json
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/512.model --config config.json
# snapshot/512.model  EWAP_eth dataset : average bpp : 0.081392, average psnr : 34.041590, average msssim: 0.980230

# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter2093.model --config config.json
# masknet : 2022-09-21 21:41:24,073 - INFO] EWAP_eth dataset : average bpp : 0.077599, average psnr : 35.079448, average msssim: 0.981171 
# masknet + recnet: 垃圾


# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter186277.model --config config.json
# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/dvc_iter2093.model --config config.json


# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter98371.model --config config.json
# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/iter52325.model --config config.json
# CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testhotel --pretrain snapshot/512.model --config config.json
