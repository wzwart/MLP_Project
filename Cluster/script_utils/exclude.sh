echo "Setting sync exlusions"
tx_exclude=( --exclude 'exp_basic/saved_models/' --exclude 'exp_*/'   --exclude 'notes/'    --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015')
rx_exclude=(--exclude 'data' --exclude '*.py'  --exclude 'exp_*/saved_models/' --exclude '*.json' --exclude '*.sh' --exclude 'notes/'  --exclude 'script_utils/'   --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015')
