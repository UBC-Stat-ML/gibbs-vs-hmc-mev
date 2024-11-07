#!/bin/bash

for i in colon ALLAML leukemia Prostate_GE BASEHOCK madelon RELATHE PCMAC arcene GLI_85 SMK_CAN_187;
do
    curl --remote-name https://jundongl.github.io/scikit-feature/files/datasets/$i.mat
done