
RED='\033[0;31m'
NC='\033[0m' # No Color

gpu=0
lr=1e-4

nepochs=20
nepochs_decay=5

# model stuff
model='bert2vqsc_v4'
bert_cfg='configs/bert2vqsc.yaml'
cat='chair'


vq_model='pvqvae'
vq_cfg='configs/pvqvae_snet.yaml'
vq_ckpt='../raw_dataset/checkpoints/vqvae.pth'
vq_dset='snet'
vq_cat='chair'

# dataset stuff
batch_size=32
max_dataset_size=100000000000
trunc_thres=0.2
display_freq=3000
print_freq=25
dataset_mode="text2shape-seq"
today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)
gpu_ids=0
note='july-7-network-bert-shapeInput-shapeset-crossEntropy-3'

name="${model}-${dataset_mode}-LR${lr}-${note}"

logs_dir='logs'

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=6
	max_dataset_size=24
    nepochs=400
    nepochs_decay=0
	update_html_freq=1
	display_freq=100
	print_freq=1
    name="DEBUG-${name}"
fi

# CUDA_VISIBLE_DEVICES=${gpuid} python train.py --name ${name} --gpu_ids ${gpuid} --lr ${lr} --batch_size ${batch_size} \
python train.py --name ${name} --gpu ${gpu} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} \
                --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset} --vq_cat ${vq_cat} \
                --nepochs ${nepochs} --nepochs_decay ${nepochs_decay} \
                --dataset_mode ${dataset_mode} --cat ${cat} --max_dataset_size ${max_dataset_size} --trunc_thres ${trunc_thres} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                --debug ${debug}
