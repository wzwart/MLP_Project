#!/bin/bash

#TXSYNC:
source pw.sh
source exclude.sh
echo "You are $USER"
echo "Local Path $local_path"
cd "$local_path"
if [ "$USER" == "wwzwart" ]; then
source paths_WZ.sh
echo "TX Sync in Progress"

echo $local_path
echo "$local_pw" | sudo -S rsync -azv  "${tx_exclude[@]}" -e"sshpass -p "$remote_pw" ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh" "$local_path" "$remote_path"


#rsync -azv  "${exclude[@]}" -e"ssh -o StrictHostKeyChecking=no -A "$remote_user" ssh" "$local_path" "$remote_path"

echo "TX Sync Finished"
fi
if [ "$USER" == "jangomezroberts" ]; then
source paths_jan.sh
echo "TX Sync in Progress"
rsync -azv  "${tx_exclude[@]}" -e"ssh -o StrictHostKeyChecking=no -A s1993340@student.ssh.inf.ed.ac.uk ssh"  "$local_path" "$remote_path"
echo "TX Sync Finished"
fi
if [ "$USER" == "andreu" ]; then
source paths_pablo.sh
echo "TX Sync in Progress"
rsync -azv  "${tx_exclude[@]}" -e"ssh -o StrictHostKeyChecking=no -A s2000253@student.ssh.inf.ed.ac.uk ssh"  "$local_path" "$remote_path"
echo "TX Sync Finished"
fi