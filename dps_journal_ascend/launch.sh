accelerate launch `
  --num_processes=1 `
  --num_machines=1 `
  --machine_rank=0 `
  --mixed_precision=no `
  main.py `
  --config "./configs/vp/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" `
  --eval_folder "./eval_folder/" `
  --mode "train" `
  --workdir "./"

# cd /opt/dpcvol/models/xsx/score_sde_pytorch_ascend
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# pip install certifi==2024.7.4
# pip install ml-collections
# accelerate launch \
#   --num_processes=8 \
#   --num_machines=1 \
#   --machine_rank=0 \
#   --mixed_precision=no \
#   main.py \
#   --config "./configs//vp/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" \
#   --eval_folder "./eval_folder/" \
#   --mode "train" \
#   --workdir "./"
