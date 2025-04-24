#!/bin/bash

TARGET="nkale"   # change me to the username whose GPU jobs you want gone

# 1) pull every GPU compute PID
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do

  # 2) get the owning user of that PID
  owner=$(ps -o user= -p "$pid")

  # 3) if it matches TARGET, kill it  
  if [[ "$owner" == "$TARGET" ]]; then
    echo "Killing GPU process $pid (owned by $owner)"
    sudo kill -9 "$pid"
  fi
done