ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=0,1,3,4 python -u $ROOT/main.py --log log.txt --config config_2048.json

