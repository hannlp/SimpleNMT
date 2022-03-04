#! /bin/bash

time=$(date "+%m-%d %H:%M")
device=0

PWD=$(
  cd "$(dirname "$0")"
  pwd
)
root_dir=/home/hanyuchen/mt
data_dir=${root_dir}/data/iwslt14.de-en

exp_name=iwslt14
src_lang=de
tgt_lang=en
exp_tag=

patience=5

model=Transformer
d_model=512
d_ff=1024
n_head=4
n_encoder_layers=6
n_decoder_layers=6
encoder_prenorm=1
decoder_prenorm=1
share_vocab=1

. ${PWD}/../utils/parse_options.sh || exit 1

lang=${src_lang}-${tgt_lang}
save_dir=${root_dir}/checkpoints/${exp_name}-${lang}

if [[ -n ${exp_tag} ]]; then
  save_dir=${save_dir}-${exp_tag}
fi

project_dir=$PWD/../../simplenmt

cmd="python -u ${project_dir}/train.py
        -src ${src_lang}
        -tgt ${tgt_lang}
        -data_path ${data_dir}
        -save_path ${save_dir}
        -model ${model}
        -d_model ${d_model}
        -d_ff ${d_ff}
        -n_head ${n_head}
        -n_encoder_layers ${n_encoder_layers}
        -n_decoder_layers ${n_decoder_layers}
        -patience ${patience}
        "
if [[ ${encoder_prenorm} -eq 1 ]]; then
  cmd="$cmd
        -encoder_prenorm"
fi
if [[ ${decoder_prenorm} -eq 1 ]]; then
  cmd="$cmd
        -decoder_prenorm"
fi
if [[ ${share_vocab} -eq 1 ]]; then
  cmd="$cmd
        -share_vocab"
fi

echo -e "\033[36mTime: ${time} | Device: ${device} | Save_dir: ${save_dir}\nRun command: \n${cmd} \033[0m"
export CUDA_VISIBLE_DEVICES=${device}
cmd="nohup ${cmd} >/dev/null 2>error.txt &" #2>&1 &
eval $cmd
sleep 4s
tail -n "$(wc -l ${save_dir}/log.txt | awk '{print $1+1}')" -f ${save_dir}/log.txt
