#!/usr/bin/env bash

rsync -av --exclude='*.pty' prince:/scratch/jb6504/classifier-reweighted-flow/experiments/data/runs/ /Users/johannbrehmer/work/projects/classifier_reweighted_flows/classifier-reweighted-flow/experiments/data/runs/
rsync -av prince:/scratch/jb6504/classifier-reweighted-flow/experiments/hpc/ /Users/johannbrehmer/work/projects/classifier_reweighted_flows/classifier-reweighted-flow/experiments/hpc/
