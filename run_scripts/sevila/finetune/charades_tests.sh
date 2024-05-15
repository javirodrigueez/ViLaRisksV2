# parameters
result_dir=""

exp_name='charades_tests'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
#python train.py \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 train.py \
--cfg-path lavis/projects/sevila/train/charades_tests.yaml \
--options run.output_dir=${result_dir}${exp_name} \
    model.frame_num=4 \
    model.total_frames=32 \
    model.use_optical_flow=False \
    model.use_qformer_clf=True \
    datasets.charades_test.vis_processor.train.n_frms=32 \
    datasets.charades_test.vis_processor.eval.n_frms=32 \
    run.batch_size_train=4 \
    run.batch_size_eval=4 \
    run.init_lr=3e-5 \
    run.max_epoch=15 \
    run.warmup_steps=3000 \
    run.accum_grad_iters=4 \
    model.task='qvh_freeze_loc_train_qa_wo_loc_train_qa_vid' \
    model.finetuned=${ckpt} \
    run.task='videoqa'