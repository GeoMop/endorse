#!/bin/bash
set -x
cp ../requirements.txt .

git_commit=`git rev-parse --short=6 HEAD`

tag=flow123d/endorse_ci:${1:-${git_commit}}
docker build --tag ${tag} .

docker push ${tag}
