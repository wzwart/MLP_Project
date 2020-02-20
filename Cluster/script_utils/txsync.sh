#!/bin/bash

#TXSYNC:

echo "You are $USER"
if [ "$USER" == "wwzwart" ]; then
echo "TX Sync in Progress"
echo K-58%U6m | sudo -S rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"sshpass -p "w*wZ47Ws" ssh -o StrictHostKeyChecking=no -A s1999534@student.ssh.inf.ed.ac.uk ssh"  /mnt/c/Data\ Science/Machine\ Learning\ Practical,\ MLP/MLP_Project/Cluster/ s1999534@mlp1:mlpractical/
echo "TX Sync Finished"
fi
if [ "$USER" == "andreu" ]; then
echo "TX Sync in Progress"
rsync -azv  --exclude 'data'  --exclude 'exp_basic/save_models/'  --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh"  /mnt/c/Users/Andreu/Documents/GitHub/MLP_Project/Cluster s2000253@mlp1:mlpractical/
echo "TX Sync Finished"
fi