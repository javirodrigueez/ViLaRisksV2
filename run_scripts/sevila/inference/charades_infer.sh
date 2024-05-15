# parameters/data path
result_dir=""

exp_name='charades_infer_ft_age_2'
#ckpt='sevila_checkpoints/sevila_pretrained.pth'
ckpt='lavis/charades_agescene/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--cfg-path lavis/projects/sevila/eval/charades_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
model.total_frames=32 \
model.use_vit=True \
datasets.charades.load_videos=True \
datasets.charades.vis_processor.eval.n_frms=32 \
run.batch_size_eval=4 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa' \