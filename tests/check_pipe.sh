#!/bin/bash

# Default database location:
db='testdata'
# Default test dataset for CNN based pipeline is the "real_plant":
dataset="$db/real_plant/"
# Virtual test dataset:
v_dataset="$db/virtual_plant/"
# Default tested ROMI task for CNN based pipeline is "AnglesAndInternodes":
task='AnglesAndInternodes'

usage(){
  echo "USAGE:"
  echo "  ./check_pipe.sh [OPTIONS]
  "
  echo "DESCRIPTION:"
  echo "  Run both reconstruction pipelines with predefined datasets and configurations.

  Clean the dataset then run 'romi_run_task <task> <dataset> --config <config>'.
  Some information are printed for required task.
  By default report for tasks 'Colmap', 'PointCloud' & 'AnglesAndInternodes'.

  Fore the ML pipeline, it may download a trained organ segmentation model if missing from the database.
  "
  echo "OPTIONS:"
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
  echo "  #1 - Run both geometric & CNN reconstruction pipelines on default 'real plant' test dataset (safe mode):
  $ ./check_pipe.sh --tmp
  "
  echo "  #2 - Run both geometric & CNN reconstruction pipelines up to the 'PointCloud' task on 'virtual plant' test dataset (safe mode):
  $ ./check_pipe.sh -t PointCloud --virtual --tmp
  "
  echo "  #3 - Run both geometric & CNN reconstruction pipelines with another dataset (safe mode):
  $ ./check_pipe.sh --dataset /data/ROMI/DB/arabido_test2/ --tmp
  "
}
opts=""
while [ "$1" != "" ]; do
  case $1 in
  -d | --dataset)
    shift
    opts="$opts -d $1"
    ;;
  -t | --task)
    shift
    opts="$opts -t $1"
    ;;
  --virtual)
    shift
    opts="$opts --virtual"
    ;;
  --tmp)
    shift
    opts="$opts --tmp"
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

##### Check geometric pipeline
echo "

Testing the GEOMETRICAL pipeline...
"
./check_geom_pipe.sh $opts

##### Check machine learning pipeline
echo "

Testing the CNN based pipeline...
"
./check_ml_pipe.sh $opts
