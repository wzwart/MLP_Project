echo "Setting user credentials"
tx_exclude=( --exclude 'exp_basic/saved_models/' --exclude 'exp_unet/saved_models/' --exclude 'notes/'  --exclude 'script_utils/'   --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015' --exclude 'exp_300W_5/' --exclude 'exp_300W_6/' --exclude 'exp_300W_7/' --exclude 'exp_300W_8/' --exclude 'exp_300W_9/' --exclude 'exp_Both/' --exclude 'exp_null/')
rx_exclude=(--exclude 'data' --exclude '*.py' --exclude '*.json' --exclude '*.sh' --exclude 'exp_basic/saved_models/' --exclude 'exp_unet/saved_models/' --exclude 'notes/'  --exclude 'script_utils/'   --exclude '.git' --exclude '__pycache__' --exclude '.idea' --exclude '\#015')
