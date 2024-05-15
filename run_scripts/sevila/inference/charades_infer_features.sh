# parameters/data path
result_dir=""

exp_name='charades_inferx'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--cfg-path lavis/projects/sevila/eval/charades_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=16 \
datasets.charades.vis_processor.eval.n_frms=16 \
run.batch_size_eval=4 \
model.task='qvh_freeze_loc_freeze_qa_vid_uni_eval' \
model.finetuned=${ckpt} \
model.use_vit=False \
run.task='videoqa'