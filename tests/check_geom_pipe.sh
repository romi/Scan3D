#!/bin/bash

# Default configuration file used for GEOMETRIC based pipeline is:
cfg='../config/geom_pipe_test.toml'
# Default database location:
db='testdata'
# Default test dataset for GEOMETRIC based pipeline is the "real_plant":
dataset="$db/real_plant/"
# Virtual test dataset:
v_dataset="$db/virtual_plant/"
# Default tested ROMI task for GEOMETRIC based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'

usage(){
  echo "USAGE:"
  echo "  ./check_test_geom_pipe.sh [OPTIONS]
  "
  echo "DESCRIPTION:"
  echo "  Run the geometric reconstruction pipeline with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'.
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
  echo "  #1 - Run the geometric reconstruction pipeline on default 'real plant' test dataset (safe mode):
  $ ./check_geom_pipe.sh --tmp
  "
  echo "  #2 - Run the geometric pipeline up to the 'PointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./check_geom_pipe.sh -t PointCloud --virtual --tmp
  "
  echo "  #3 - Run a geometric reconstruction pipeline with another config & test dataset (safe mode):
  $ ./check_geom_pipe.sh --config ../config/geom_pipe_test.toml --dataset /data/ROMI/DB/arabido_test2/ --tmp
  "
}
tmp=0
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
  # Creates the temporary directory path variable with it
  tmp_dataset="$tmp_db/$data_dir"
  # Make sure it does not exist or remove it:
  if [[ -d $tmp_dataset ]]; then
    rm -rf $tmp_dataset
  fi
  # Copy the data
  echo "Copying '$dataset' to '$tmp_db'..."
  cp -R $dataset "$tmp_db"
  # Finally replace the dataset location by the temporary one
  dataset="$tmp_dataset"
fi

##### Check geometric pipeline
# 1. clean
romi_run_task Clean $dataset --config $cfg

# 2. run pipeline
echo "romi_run_task $task $dataset --config $cfg"
romi_run_task $task $dataset --config $cfg

# 3. print informations about tested task
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
