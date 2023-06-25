ROOT=savecode/
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir snapshot
CUDA_VISIBLE_DEVICES=2,3  python -u $ROOT/main.py --log log.txt --testeth --pretrain snapshot/1024_iter2093.model --config config_1024.json
