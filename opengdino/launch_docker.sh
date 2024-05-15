#!/bin/bash
docker run --gpus '"device=0"' --rm -it \
	--name jrodriguez_opengdino \
	--volume="/home/jrodriguez/tfm/:/workspace/tfm/:rw" \
	--volume="/mnt/md1/datasets/Charades_v1/:/charades/:ro" \
	--volume="/mnt/md1/datasets/ETRI-Activity/P001-P010/:/etri/:ro" \
	--shm-size=16gb \
	--memory=24gb \
	jrodriguez/opengdino bash
