# /usr/bin/sh

ROOT_DIR=$1
WORK_DIR="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )" )"

if [ ! -d WORK_DIR/data ]
then
    mkdir -p $WORK_DIR/data
    printf $ROOT_DIR >> $WORK_DIR/root.txt
fi

