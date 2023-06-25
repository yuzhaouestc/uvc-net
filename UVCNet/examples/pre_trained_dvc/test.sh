ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/dvc_pretrain2048.model --config config.json


# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/2048.model --config config.json
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/1024.model --config config.json
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/512.model --config config.json
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain snapshot/256.model --config config.json



# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain ../dvc_resize_256/snapshot/iter22814.model --config config.json
# CUDA_VISIBLE_DEVICES=2  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain ../dvc_resize_512/snapshot/iter22814.model --config config.json
# CUDA_VISIBLE_DEVICES=3  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain ../dvc_resize_1024/snapshot/iter22814.model --config config.json
# CUDA_VISIBLE_DEVICES=1  python -u $ROOT/main.py --log loguvg.txt --testuvg --pretrain ../dvc_resize_2048/snapshot/iter67694.model --config config.json



# ========= 生成光流文件 ========
CUDA_VISIBLE_DEVICES=1  python -u $ROOT/ge_optical_flow.py --log loguvg.txt --testuvg --pretrain ../dvc_resize_2048/snapshot/iter67694.model --config config.json