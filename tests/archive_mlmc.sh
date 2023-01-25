#!/bin/bash
set -x

function compress {
    for d in $@
    do
        (cd $d && tar -czvf output.tar.gz output && rm -rf output)
    done
}

function extract {
    for d in $@
    do
        (cd $d && tar -xvf output.tar.gz)
    done
}


if [ "$1" == "-x" ]
then
    shift
    extract $@
else    
    compress $@
fi
