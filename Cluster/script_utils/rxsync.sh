#!/bin/bash


echo "You are $USER"
if [ "$USER" == "wwzwart" ]; then
echo "RX Sync in Progress"
echo K-58%U6m | sudo -S rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"sshpass -p "w*wZ47Ws" ssh -o StrictHostKeyChecking=no -A s1999534@student.ssh.inf.ed.ac.uk ssh" s1999534@mlp1:mlpractical/ /mnt/c/Data\ Science/Machine\ Learning\ Practical,\ MLP/MLP_Project/Cluster/
echo "RX Sync Finished"
fi
if [ "$USER" == "andreu" ]; then
echo "RX Sync in Progress"
rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh" s2000253@mlp1:mlpractical/ /mnt/c/Users/Andreu/Documents/GitHub/MLP_Project/Cluster
echo "RX Sync Finished"
fi

