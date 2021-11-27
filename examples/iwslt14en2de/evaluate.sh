#! /bin/bash

time=$(date "+%m-%d %H:%M")
device=0

PWD=$(cd "$(dirname "$0")";pwd)
root_dir=/home/hanyuchen/mt
data_dir=${root_dir}/data/iwslt14.de-en

exp_name=iwslt14
src_lang=en
tgt_lang=de
exp_tag=

dec_model=checkpoint_best.pt
ckpt_suffix=best
n_average=1
checkpoint_upper_bound=0
beam_size=0
len_penalty=1.0
quiet=1

. ${PWD}/../utils/parse_options.sh || exit 1;

lang=${src_lang}-${tgt_lang}
save_dir=${root_dir}/checkpoints/${exp_name}-${lang}

if [[ -n ${exp_tag} ]]; then
    save_dir=${save_dir}_${exp_tag}
fi

project_dir=$PWD/../../simplenmt

if [[ ${n_average} -gt 1 ]]; then
		dec_model=checkpoint_avg_${n_average}.pt
		ckpt_suffix=avg_${n_average}

		cmd="python ${project_dir}/utils/average_checkpoints.py
        --inputs ${save_dir}
        --num-epoch-checkpoints ${n_average}
        --output ${save_dir}/${dec_model}"
        if [[ ${checkpoint_upper_bound} -gt 0 ]]; then
            cmd="${cmd}
        --checkpoint-upper-bound ${checkpoint_upper_bound}"
        fi
    echo -e "\033[36mRun command: \n${cmd} \033[0m"
    eval $cmd
fi

cmd="python -u ${project_dir}/translate.py
        -src ${src_lang}
        -tgt ${tgt_lang}
        -data_path ${data_dir}
        -save_path ${save_dir}
        -beam_size ${beam_size}
        -ckpt_suffix ${ckpt_suffix}
        -generate"

      if [[ ${quiet} -eq 1 ]]; then
        cmd="$cmd
        -quiet"
      fi

echo -e "\033[36mTime: ${time} | Device: ${device} | Save_dir: ${save_dir}\nRun command: \n${cmd} \033[0m"
export CUDA_VISIBLE_DEVICES=${device}
eval $cmd

score_file=${save_dir}/avg${n_average}beam${beam_size}score.txt
[[ -f ${score_file} ]] && rm ${score_file}
grep ^-T ${save_dir}/result.txt | cut -f2 | sed -r 's/(@@ )| (@@ ?$)//g' > ${save_dir}/ref.txt
grep ^-P ${save_dir}/result.txt | cut -f2 | sed -r 's/(@@ )| (@@ ?$)//g' > ${save_dir}/pred.txt
perl ${project_dir}/utils/multi-bleu.perl ${save_dir}/pred.txt < ${save_dir}/ref.txt > ${score_file}
cat ${score_file}