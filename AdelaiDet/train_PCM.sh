OMP_NUM_THREADS=1 python tools/train_PCM_solo.py \
    --config-file configs/SOLOv2/R50_3x_PCM.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/SOLOv2_R50_3x_PCM
