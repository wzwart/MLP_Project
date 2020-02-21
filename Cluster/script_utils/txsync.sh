#!/bin/bash

#TXSYNC:
source pw.sh
source exclude.sh
echo "You are $USER"
echo "Local Path $local_path"
cd "$local_path"
if [ "$USER" == "wwzwart" ]; then
source paths_wz.sh
echo "TX Sync in Progress"
echo "$local_pw" | sudo -S rsync -azv  "${exclude[@]}" -e"sshpass -p "$remote_pw" ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh" "$local_path" "$remote_path"
echo "TX Sync Finished"
fi
if [ "$USER" == "andreu" ]; then
echo "TX Sync in Progress"
rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh"  /mnt/c/Users/Andreu/Documents/GitHub/MLP_Project/Cluster s2000253@mlp1:mlpractical/
echo "TX Sync Finished"
fi