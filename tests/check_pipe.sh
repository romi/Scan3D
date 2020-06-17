#!/bin/bash

##### Check geometric pipeline
# 1. clean
romi_run_task --config ../config/original_pipe_0.toml Clean testdata/real_plant/ --local-scheduler
# 2. run pipeline
romi_run_task --config ../config/original_pipe_0.toml AnglesAndInternodes testdata/real_plant/ --local-scheduler

##### Check machine learning pipeline
# 1. download models
MODEL_DIRECTORY="testdata/models/models"

if [ ! -d ${MODEL_DIRECTORY} ]; then
	mkdir ${MODEL_DIRECTORY}
fi

MODEL_EPOCH_896_896_50="${MODEL_DIRECTORY}/Resnetdataset_gl_png_896_896_epoch50.pt"

if [ ! -f ${MODEL_EPOCH_896_896_50} ]; then
	wget https://media.romi-project.eu/data/Resnetdataset_gl_png_896_896_epoch50.pt
	mv Resnetdataset_gl_png_896_896_epoch50.pt ${MODEL_DIRECTORY}
fi

MODEL_TMP_40="${MODEL_DIRECTORY}/tmp_epoch40.pt"
if [ ! -f ${MODEL_TMP_40} ]; then
	wget https://media.romi-project.eu/data/tmp_epoch40.pt
	mv tmp_epoch40.pt ${MODEL_DIRECTORY}
fi

# 2. clean
romi_run_task --config ../config/ml_pipe_vplants_3.toml Clean testdata/virtual_plant/ --local-scheduler

# 3. run pipeline
romi_run_task --config ../config/ml_pipe_vplants_3.toml PointCloud testdata/virtual_plant/ --local-scheduler