#!/bin/bash

source pw.sh
source exclude.sh
echo "You are $USER"
echo "RX Sync in Progress"
if [ "$USER" == "wwzwart" ]; then
source paths_wz.sh
echo "$local_pw" | sudo -S rsync -azv  "${exclude[@]}" -e"sshpass -p "$remote_pw" ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh"  "$remote_path" "$local_path"
fi
if [ "$USER" == "jangomezroberts" ]; then
source paths_jan.sh
echo "$local_pw" | sudo -S rsync -azv -n "${exclude[@]}" -e"sshpass -p "$remote_pw" ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh"  "$remote_path" "$local_path"
fi
if [ "$USER" == "andreu" ]; then
rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh" s2000253@mlp1:mlpractical/ /mnt/c/Users/Andreu/Documents/GitHub/MLP_Project/Cluster
fi
echo "RX Sync Finished"

