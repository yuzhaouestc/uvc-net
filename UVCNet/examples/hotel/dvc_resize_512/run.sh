ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=2 python -u $ROOT/main.py --log log.txt --config config.json

