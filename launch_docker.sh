#!/bin/bash
docker run --gpus '"device=0,1"' --rm -it \
	--name jrodriguez_sevila \
	--volume="/home/jrodriguez/tfm/:/workspace/models/:rw" \
	--volume="/mnt/md1/datasets/Charades_v1/:/charades/:rw" \
	--volume="/mnt/md1/datasets/ETRI-Activity/:/etri/:rw" \
	--shm-size=16gb \
	--memory=48gb \
	javiro01/vlm-risks-assessment:v1 bash
