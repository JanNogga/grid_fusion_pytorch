#!/usr/bin/env bash

# Check args
if [ "$#" -ne 0 ]; then
	  echo "usage: ./run.sh"
	    return 1
    fi

    # Get this script's path
    pushd `dirname $0` > /dev/null
    SCRIPTPATH=`pwd`
    popd > /dev/null

    set -e

	# for more info see: https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
	#xhost +local:root #local:root
	#--net=host	
    # Run the container with shared X11
    docker run\
	    --shm-size 12G\
	    --gpus all\
		  --net host\
		  -e SHELL\
      -e DISPLAY=$DISPLAY\
		  -e DOCKER=1\
		  --name module_name\
		  -v $(dirname `pwd`):/repos/module_name\
		  -v /home/nfs/inf6/data/datasets:/home/nfs/inf6/data/datasets\
		  -it module_name
