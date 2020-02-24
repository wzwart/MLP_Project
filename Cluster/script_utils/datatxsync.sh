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
echo "$local_pw" | sudo -S rsync -azv   -e"sshpass -p "$remote_pw" ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh" "$local_path"../data "$remote_path"
echo "TX Sync Finished"
fi
if [ "$USER" == "jangomezroberts" ]; then
echo "TX Sync in Progress"
rsync -azv  -e"ssh -o StrictHostKeyChecking=no -A s1993340@student.ssh.inf.ed.ac.uk ssh"  /Users/jangomezroberts/MLP_Project/data s1993340@mlp1:mlpractical/MLP_Project
echo "TX Sync Finished"
fi
if [ "$USER" == "andreu" ]; then
echo "TX Sync in Progress"
rsync -azv  -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh"  /mnt/c/Users/Andreu/Documents/GitHub/MLP_Project/data s2000253@mlp1:mlpractical/MLP_Project
echo "TX Sync Finished"
fi