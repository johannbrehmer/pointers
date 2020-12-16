#!/usr/bin/env bash

rsync -av --exclude='*.pty' prince:/path/to/experiment/folder/data/runs/ /local/experiment/folder/data/runs/
