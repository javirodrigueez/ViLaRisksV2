# parameters
result_dir=""

exp_name='charades_a-gg_uni_keys'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py \
python train.py \
--cfg-path lavis/projects/sevila/train/charades.yaml \
--options run.output_dir=${result_dir}${exp_name} \
    model.frame_num=16 \
    model.total_frames=144 \
    datasets.charades.vis_processor.train.n_frms=144 \
    datasets.charades.vis_processor.eval.n_frms=144 \
    run.batch_size_train=2 \
    run.batch_size_eval=2 \
    run.init_lr=3e-5 \
    run.max_epoch=50 \
    run.warmup_steps=1000 \
    run.accum_grad_iters=2 \
    model.task='qvh_freeze_loc_train_qa_wo_loc_train_qa_vid_uni_eval' \
    model.finetuned=${ckpt} \
    run.task='videoqa'