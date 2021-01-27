#!/bin/bash

# Default configuration file used for CNN based pipeline is:
cfg='../config/ml_pipe_test.toml'
# Default database location:
db='testdata'
# Default test dataset for CNN based pipeline is the "real_plant":
dataset="$db/real_plant/"
# Virtual test dataset:
v_dataset="$db/virtual_plant/"
# Default tested ROMI task for CNN based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'
# Directory with the trained organ segmentation models
MODEL_DIRECTORY="models/models"

usage(){
  echo "USAGE:"
  echo "  ./check_test_ml_pipe.sh [OPTIONS]
  "
  echo "DESCRIPTION:"
  echo "  Run the CNN reconstruction pipeline with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'.

  It may download a trained organ segmentation model if missing from the database.
  "
  echo "OPTIONS:"
  echo "  -c, --config
    Pipeline configuration file to use, default to '$cfg'."
  echo "  -d, --dataset
    Data set to use, default to '$dataset'."
  echo "  -t, --task
    Task to test, default to '$task'."
  echo "  --virtual
    Use the virtual plant dataset."
  echo "  --tmp
    Clone the dataset to the temporary folder '/tmp' first.
    Use this to avoid messing up your repository 'testdata' directory as the Clean task is not reliable."
  echo "  -h, --help
    Output a usage message and exit.
  "
  echo "EXAMPLES:"
  echo "  #1 - Run the CNN reconstruction pipeline on default 'real plant' test dataset (safe mode):
  $ ./check_ml_pipe.sh --tmp
  "
  echo "  #2 - Run the CNN reconstruction pipeline up to the 'SegmentedPointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./check_ml_pipe.sh -t SegmentedPointCloud --virtual --tmp
  "
  echo "  #3 - Run the CNN reconstruction pipeline with another config & test dataset (safe mode):
  $ ./check_ml_pipe.sh --config ../config/ml_pipe_test.toml --dataset /data/ROMI/DB/arabido_test2/ --tmp
  "
}
while [ "$1" != "" ]; do
  case $1 in
  -c | --config)
    shift
    cfg=$1
    ;;
  -d | --dataset)
    shift
    dataset=$1
    ;;
  -t | --task)
    shift
    task=$1
    ;;
  --virtual)
    shift
    dataset=$v_dataset
    ;;
  --tmp)
    shift
    tmp=1
    ;;
  -h | --help)
    usage
    exit
    ;;
  *)
    usage
    exit 1
    ;;
  esac
  shift
done

# If not defined, set 'PYOPENCL_CTX' to O
if [ -z "$PYOPENCL_CTX" ]; then
  export PYOPENCL_CTX='0'
  echo "Missing 'PYOPENCL_CTX' environment variable, set it to '0'."
fi

# Create the copy to temporary folder
if [ "$tmp" = 1 ]; then
  tmp_db="/tmp/romidb"
  # Create `romidb` folder as root to temporary database...
  mkdir -p $tmp_db
  # Add the romidb marker file
  touch "$tmp_db/romidb"
  # Get the directory name (last in hierarchy):
  data_dir=`basename "$dataset"`
  # Add a date prefix to make folder unique and a 'ML' tag to further explicit test folder name
  data_dir="$(date +%F_%H-%M-%S)_ML_$data_dir"
  # Creates the temporary directory path variable with it
  tmp_dataset="$tmp_db/$data_dir"
  # Make sure it does not exist or remove it:
  if [[ -d $tmp_dataset ]]; then
    rm -rf $tmp_dataset
  fi
  # Copy the dataset to new temporary folder
  echo "Copying '$dataset' to 'tmp_dataset'..."
  cp -R $dataset "$tmp_dataset"
  # Copy the models fileset to new temporary folder
  echo "Copying '$db/models' to '$tmp_db/models'..."
  cp -R "$db/models" "$tmp_db/."
  # Finally replace the database & dataset locations by the temporary ones
  dataset="$tmp_dataset"
  db="$tmp_db"
fi

##### Check machine learning pipeline
MODEL_DIRECTORY="$db/$MODEL_DIRECTORY"
# 1. download models
if [ ! -d ${MODEL_DIRECTORY} ]; then
  mkdir -p ${MODEL_DIRECTORY}
  echo "Created missing models directory: ${MODEL_DIRECTORY}."
fi

MODEL_EPOCH_896_896_50="${MODEL_DIRECTORY}/Resnetdataset_gl_png_896_896_epoch50.pt"
if [ ! -f ${MODEL_EPOCH_896_896_50} ]; then
  echo 'Download missing trained CNN models...'
  wget -P ${MODEL_DIRECTORY} https://media.romi-project.eu/data/Resnetdataset_gl_png_896_896_epoch50.pt
fi
#MODEL_TMP_40="${MODEL_DIRECTORY}/tmp_epoch40.pt"
#if [ ! -f ${MODEL_TMP_40} ]; then
#	wget https://media.romi-project.eu/data/tmp_epoch40.pt
#	mv tmp_epoch40.pt ${MODEL_DIRECTORY}
#fi

# 2. clean
romi_run_task Clean $dataset --config $cfg

# 3. run pipeline
echo "romi_run_task $task $dataset --config $cfg"
romi_run_task $task $dataset --config $cfg

# 4.print informations about tested task
if [ "$task" = "AnglesAndInternodes" ]; then
  # Also inform about Colmap and PointCloud tasks if tested task is "AnglesAndInternodes"
  echo ""
  print_task_info Colmap $dataset
  echo "
  "
  print_task_info PointCloud $dataset
  echo "
  "
fi
print_task_info $task $dataset