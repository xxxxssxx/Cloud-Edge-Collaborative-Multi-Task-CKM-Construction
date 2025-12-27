# windows:
accelerate launch `
  --num_processes=1 `
  --num_machines=1 `
  --machine_rank=0 `
  --mixed_precision=no `
  main.py `
  --config "./configs/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" `
  --eval_folder "./eval_folder/" `
  --mode "train" `
  --workdir "./"

# linux:
# cd /opt/dpcvol/models/xsx/baseline_multi_task
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# pip install certifi==2024.7.4
# pip install ml-collections
# accelerate launch \
#   --num_processes=8 \
#   --num_machines=1 \
#   --machine_rank=0 \
#   --mixed_precision=no \
#   main.py \
#   --config "./configs/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" \
#   --eval_folder "./eval_folder/" \
#   --mode "train" \
#   --workdir "./"
