python tools/inference_on_a_image.py \
  -c tools/GroundingDINO_SwinT_OGC.py \
  -p logs_train3/checkpoint_best_regular.pth \
  -i ./figs/kitchen-burn.jpg \
  -t "refrigerator . stove . towel_s . microwave . casserole" \
  -o output