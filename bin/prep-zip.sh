#!/bin/bash

mods=`git status --porcelain | tail -n 1`
if [ "${#mods}" -gt "0" ]
then
    echo "Git not clean!"
fi

mkdir ~/tmp 
mkdir ~/tmp/build
cd ~/tmp/build
yes | rm -r gibbs-race-mev
git clone -b anon https://github.com/UBC-Stat-ML/gibbs-race-mev.git 

yes | rm -r gibbs-race-mev/.git 
rm gibbs-race-mev.zip
zip -r gibbs-race-mev gibbs-race-mev