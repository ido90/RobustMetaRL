#### Edit the soft link, and add other configuration commands here

# add soft link to your data. e.g.
ln -s /tmp/mnist_data/ mnist_data


###### Do not edit below this line
#### Install useful linux apps
apt-get -y update
yes | apt install tmux
yes | apt install mc
yes | apt-get install tmux
yes | apt-get install htop
yes | apt-get install bc
yes | apt install rsync
yes | apt install git
cp docker_cfg/inputrc /root/.inputrc
cp docker_cfg/ts /bin/

#### Install requirements
pip install -r requirements.txt

#### Useful aliases and bash functions
# gpus status
alias gpust='nvidia-smi | grep -E "[0-9]+C.*[0-9]+W"'
alias cpr='rsync --progress'

# gpu queues
for i in {0..7}
do
   alias tsg$i="TS_SOCKET=/tmp/ts_gpu$i.$USER docker_cfg/ts"
done

function num_gpus() { echo $( nvidia-smi | grep -E "[0-9]+C.*[0-9]+W" | wc -l ); }
function gpus_list() { echo $(seq -s" " 0 $(echo $(num_gpus) - 1 | bc)); }
function ts_num() { echo $(TS_SOCKET=/tmp/ts_gpu$1.$USER docker_cfg/ts | grep -E "running|queued" | wc -l); }
function all_ts_num()  { echo $(ts_num 0) + $(ts_num 1) + $(ts_num 2) + $(ts_num 3) + $(ts_num 4) + $(ts_num 5) + $(ts_num 6) + $(ts_num 7)| bc ; }

