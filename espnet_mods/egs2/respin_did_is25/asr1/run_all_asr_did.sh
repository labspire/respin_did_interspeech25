#!/usr/bin/env bash
set -euo pipefail

source /home1/Saurabh/tools/virtual_envs/venv_espnet/bin/activate

# Default global configs (can be overridden using parse_options.sh)
suffix="indic_char"
nbpe=500
asr_script=asr_respin_lid_ce.sh
train_nj=2
inference_nj=32
debug=false
ngpu=1
infer_model="valid.acc.ave.pth"
infer_tag="decode_lid_asr_model_valid.acc.ave"
aux_tag="did_utt"
asr_task="asr_did"
token_type=char
use_lm=false
feats_type=raw
audio_format=wav
speed_perturb_factors="0.9 1.0 1.1"
feats_normalize=utt_mvn
bpe_nlsyms=""
lm_config=conf/train_lm.yaml

# Language sets
set="bh bn ch kn mg mr mt te"

. utils/parse_options.sh

for z in ${set2}; do
    # Assign GPU
    if [ ${z} == bh ]; then gpu=0
    elif [ ${z} == bn ]; then gpu=1
    elif [ ${z} == ch ]; then gpu=2
    elif [ ${z} == kn ]; then gpu=3
    elif [ ${z} == mg ]; then gpu=0
    elif [ ${z} == mr ]; then gpu=1
    elif [ ${z} == mt ]; then gpu=2
    elif [ ${z} == te ]; then gpu=3
    fi

    for loss_scale_ce in 5; do
        for y in s12345; do
            x=${z}_${y}
            for layer in ml7-11; do
                asr_tag="noaux_ssl_${layer}_con_e8_lin1024_bs6M_gacc1_ctc03_ls${loss_scale_ce}_conv1d_bneck32_rob_hs64_gelu_asr_hs1024_layer2_detach_v2"
                train_config="conf/tuning/train_asr_ssl_conformer_transformer_auxdid_e8_linear1024_bs6M_gacc1_${layer}.yaml"
                inference_config="conf/tuning/decode_transformer.yaml"

                tag="${x}"
                data_folder=data_${z}
                lang="${x}"
                tokenizer_path="${data_folder}/tokenizer_${nbpe}.model"
                train_dev="dev_${z}_nt"
                test_set="test_${z}_nt"
                train_set="train_${tag}"

                export CUDA_VISIBLE_DEVICES="${gpu}"

                expdir="exp_${tag}_${suffix}/asr_${asr_tag}"

                if [ ! -f ${expdir}/valid.acc.ave.pth ]; then
                    ./${asr_script} --stage 11 --stop_stage 11 \
                        --tag "${tag}" \
                        --data_folder "${data_folder}" \
                        --expdir "exp_${tag}_${suffix}" \
                        --dumpdir "dump_${tag}" \
                        --lang "${lang}" \
                        --auxiliary_data_tags "${aux_tag}" \
                        --local_data_opts "--stage 0 --lang ${lang}" \
                        --post_process_local_data_opts "--stage 2 --lang ${lang}" \
                        --audio_format "${audio_format}" \
                        --use_lm ${use_lm} \
                        --feats_normalize "${feats_normalize}" \
                        --lm_config "${lm_config}" \
                        --token_type "${token_type}" \
                        --nbpe "${nbpe}" \
                        --bpe_nlsyms "${bpe_nlsyms}" \
                        --feats_type "${feats_type}" \
                        --speed_perturb_factors "${speed_perturb_factors}" \
                        --asr_config "${train_config}" \
                        --inference_config "${inference_config}" \
                        --train_set "${train_set}" \
                        --valid_set "${train_dev}" \
                        --test_sets "${test_set}" \
                        --bpe_train_text "${data_folder}/${train_set}/text" \
                        --lm_train_text "${data_folder}/${train_set}/text" \
                        --local_score_opts "--score_lang_id true" \
                        --nj ${train_nj} \
                        --ngpu ${ngpu} \
                        --inference_asr_model ${infer_model} \
                        --inference_nj ${inference_nj} \
                        --asr_tag "${asr_tag}" \
                        --inference_tag "${infer_tag}" \
                        --use_dial_file true \
                        --skip_train false \
                        --asr_task "${asr_task}" \
                        --tokenizer_path "${tokenizer_path}" \
                        --loss_scale_ce ${loss_scale_ce} \
                        --gpu_inference false \
                        --debug ${debug}
                else
                    echo "${expdir}/valid.acc.ave.pth exists"
                fi

                # Inference + Scoring
                for data in nt; do
                    decode_dir="${expdir}/${infer_tag}/test_${z}_${data}"

                    if [ -f ${expdir}/valid.acc.ave.pth ] && [ ! -f ${decode_dir}/lid_score.txt ]; then
                        ./${asr_script} --stage 12 --stop_stage 13 \
                            --tag "${tag}" \
                            --data_folder "${data_folder}" \
                            --expdir "exp_${tag}_${suffix}" \
                            --dumpdir "dump_${tag}" \
                            --lang "${lang}" \
                            --auxiliary_data_tags "${aux_tag}" \
                            --local_data_opts "--stage 0 --lang ${lang}" \
                            --post_process_local_data_opts "--stage 2 --lang ${lang}" \
                            --audio_format "${audio_format}" \
                            --use_lm ${use_lm} \
                            --feats_normalize "${feats_normalize}" \
                            --lm_config "${lm_config}" \
                            --token_type "${token_type}" \
                            --nbpe "${nbpe}" \
                            --bpe_nlsyms "${bpe_nlsyms}" \
                            --feats_type "${feats_type}" \
                            --speed_perturb_factors "${speed_perturb_factors}" \
                            --asr_config "${train_config}" \
                            --inference_config "${inference_config}" \
                            --train_set "${train_set}" \
                            --valid_set "${train_dev}" \
                            --test_sets "test_${z}_${data}" \
                            --bpe_train_text "${data_folder}/${train_set}/text" \
                            --lm_train_text "${data_folder}/${train_set}/text" \
                            --local_score_opts "--score_lang_id true" \
                            --nj ${train_nj} \
                            --ngpu ${ngpu} \
                            --inference_asr_model ${infer_model} \
                            --inference_nj ${inference_nj} \
                            --asr_tag "${asr_tag}" \
                            --inference_tag "${infer_tag}" \
                            --use_dial_file true \
                            --skip_train true \
                            --asr_task "${asr_task}" \
                            --tokenizer_path "${tokenizer_path}" \
                            --loss_scale_ce ${loss_scale_ce} \
                            --gpu_inference false \
                            --debug ${debug}
                    else
                        echo "${expdir}/valid.acc.ave.pth missing or decode already done"
                    fi

                    if [ -f ${decode_dir}/lid_score.txt ]; then
                        cat ${decode_dir}/logdir/output.*/1best*/encoder* | sort -u -k1,1 \
                            | awk '{print $1 "\t" $2}' | awk '{if(NF==1){$2="6"}; print $1 "\t" $2}' > ${decode_dir}/text_did

                        python3 local/score_lang_id_jan21.py \
                            --ref_file ${data_folder}/test_${z}_nt_did/utt2dial \
                            --hyp_file ${decode_dir}/text_did \
                            --out ${decode_dir}/lid_score_did.txt \
                            --mismatch_file ${decode_dir}/lid_mismatch_did.tsv \
                            --lang_file ${decode_dir}/lang_id_refs_did.tsv || exit 1
                    fi
                done
            done
        done
    done
done
