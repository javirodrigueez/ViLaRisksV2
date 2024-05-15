# parameters
result_dir=""

exp_name='charades_scene'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 train.py \
CUDA_VISIBLE_DEVICES=1 python train.py \
--cfg-path lavis/projects/sevila/train/charades.yaml \
--options run.output_dir=${result_dir}${exp_name} \
    model.frame_num=4 \
    model.total_frames=32 \
    model.use_vit=True \
    datasets.charades.load_videos=True \
    datasets.charades.vis_processor.train.n_frms=32 \
    datasets.charades.vis_processor.eval.n_frms=32 \
    run.batch_size_train=4 \
    run.batch_size_eval=4 \
    run.num_workers=4 \
    run.init_lr=3e-5 \
    run.max_epoch=15 \
    run.warmup_steps=3000 \
    run.accum_grad_iters=4 \
    model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
    model.finetuned=${ckpt} \
    run.task='videoqa'