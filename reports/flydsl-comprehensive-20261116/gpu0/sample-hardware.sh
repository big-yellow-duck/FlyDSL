#!/usr/bin/env bash
set -u
while :; do
  date -Is
  amd-smi metric -g 0 -p -c -t -l 2>&1
  sleep 300
done
