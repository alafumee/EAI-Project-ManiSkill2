CUDA_VISIBLE_DEVICES=1 python sb3_ppo_liftcube_state.py -e PickSingleEGAD-v0 --log-dir logs_ppo_egad_sde
CUDA_VISIBLE_DEVICES=3 python sb3_ppo_liftcube_state.py -e PickSingleYCB-v0 --log-dir logs_ppo_ycb_sde
CUDA_VISIBLE_DEVICES=4 python sb3_ppo_liftcube_state.py -e StackCube-v0 --log-dir logs_ppo_stack_sde
CUDA_VISIBLE_DEVICES=5 python sb3_ppo_liftcube_state.py -e TurnFaucet-v0 --log-dir logs_ppo_faucet_sde
CUDA_VISIBLE_DEVICES=6 python sb3_ppo_liftcube_state_cabinet.py -e OpenCabinetDoor-v1 --log-dir logs_ppo_cabinet_sde