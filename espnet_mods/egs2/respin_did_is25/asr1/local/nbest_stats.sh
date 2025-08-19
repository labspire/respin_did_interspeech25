set -euo pipefail

langs="bh bn ch kn mg mr mt te" #"bh bn ch kn mg mr mt te"
# langs="mr"

rdir="/home1/Saurabh/exp/espnet/RESPIN/DID_ASR/CTC_AUX_EXP"
stage=2

for lang in ${langs}; do
    if [ ${lang} == bh ]; then
        nbest=5
        ndial=3
    elif [ ${lang} == ch ] || [ ${lang} == mg ] || [ ${lang} == mr ] || [ ${lang} == mt ] || [ ${lang} == te ]; then
        nbest=6
        ndial=4
    elif [ ${lang} == bn ] || [ ${lang} == kn ]; then
        nbest=7
        ndial=5
    fi
    nbest=10

    layers="noaux_ssl_ml7-11_con_e8_lin1024_bs6M_gacc1_ctc03_did"

    for y in s12345; do
		x=${lang}_${y}
		suffix="did"
		test_suffix="${nbest}best_05ctc"
        for layer in ${layers}; do

            if [ ${stage} -eq 1 ]; then
            # (
                # if [ ${lang} == "bh" ] || [ ${lang} == "mr" ]; then
                # 	expdir=${rdir}/exp_${x}${suffix:+"_$suffix"}_indic_char/asr_auxctc_utt1_6enc${suffix:+"_$suffix"}_${layer}
                # else
                expdir=${rdir}/exp_${x}${suffix:+"_$suffix"}_indic_char/asr_${layer}
                # fi

                for data in nt; do

                    sdir=${expdir}/decode_did${test_suffix:+"_$test_suffix"}_asr_model_valid.acc.ave/org/dev_${lang}_${data}${suffix:+"_$suffix"}/nbest_stats
                    mkdir -p ${sdir}

                    for n in $(seq 1 ${nbest}); do
                        cat ${sdir}/../logdir/output.*/${n}best_recog/text |awk '{if($2==""){$2="<blank>"}; print $1 "\t" $2}' |sort -u -k1,1 >${sdir}/${n}best_recog.text
                        cat ${sdir}/../logdir/output.*/${n}best_recog/score |sed -e 's/tensor(//g' -e 's/)//g' |awk '{print $1 "\t" $2}' |sort -u -k1,1 >${sdir}/${n}best_recog.score
                    done

                    # Paste the nbest_recog.text and nbest_recog.score files to a single file
                    if [ ${nbest} -eq 5 ]; then
                        paste ${sdir}/{1,2,3,4,5}best_recog.text ${sdir}/{1,2,3,4,5}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 6 ]; then
                        paste ${sdir}/{1,2,3,4,5,6}best_recog.text ${sdir}/{1,2,3,4,5,6}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 7 ]; then
                        paste ${sdir}/{1,2,3,4,5,6,7}best_recog.text ${sdir}/{1,2,3,4,5,6,7}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24,26,28 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 10 ]; then
                        paste ${sdir}/{1,2,3,4,5,6,7,8,9,10}best_recog.text ${sdir}/{1,2,3,4,5,6,7,8,9,10}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40 >${sdir}/utt_label_post_all
                    fi

                    # Get the posterior.csv file
                    echo -e "Getting posterior.csv file for ${lang}"
                    python3 ${rdir}/local/sort_posterior_v2.py ${sdir}/utt_label_post_all ${sdir}/posterior.csv \
                        --nbest ${nbest} --lang ${lang}

                    head -n 1 ${sdir}/posterior.csv |tail -1
                done
                
                for data in public utss stus stss; do # nt utss stus stss

                    sdir=${expdir}/decode_did${test_suffix:+"_$test_suffix"}_asr_model_valid.acc.ave/test_${lang}_${data}${suffix:+"_$suffix"}/nbest_stats
                    mkdir -p ${sdir}

                    for n in $(seq 1 ${nbest}); do
                        cat ${sdir}/../logdir/output.*/${n}best_recog/text |awk '{if($2==""){$2="<blank>"}; print $1 "\t" $2}' |sort -u -k1,1 >${sdir}/${n}best_recog.text
                        cat ${sdir}/../logdir/output.*/${n}best_recog/score |sed -e 's/tensor(//g' -e 's/)//g' |awk '{print $1 "\t" $2}' |sort -u -k1,1 >${sdir}/${n}best_recog.score
                    done

                    # Paste the nbest_recog.text and nbest_recog.score files to a single file
                    if [ ${nbest} -eq 5 ]; then
                        paste ${sdir}/{1,2,3,4,5}best_recog.text ${sdir}/{1,2,3,4,5}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 6 ]; then
                        paste ${sdir}/{1,2,3,4,5,6}best_recog.text ${sdir}/{1,2,3,4,5,6}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 7 ]; then
                        paste ${sdir}/{1,2,3,4,5,6,7}best_recog.text ${sdir}/{1,2,3,4,5,6,7}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24,26,28 >${sdir}/utt_label_post_all
                    elif [ ${nbest} -eq 10 ]; then
                        paste ${sdir}/{1,2,3,4,5,6,7,8,9,10}best_recog.text ${sdir}/{1,2,3,4,5,6,7,8,9,10}best_recog.score |cut -f1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40 >${sdir}/utt_label_post_all
                    fi

                    # Get the posterior.csv file
                    echo -e "Getting posterior.csv file for ${lang}"
                    python3 ${rdir}/local/sort_posterior_v2.py ${sdir}/utt_label_post_all ${sdir}/posterior.csv \
                        --nbest ${nbest} --lang ${lang}

                    head -n 1 ${sdir}/posterior.csv |tail -1
                done
            # ) &
            fi

            if [ ${stage} -eq 2 ]; then
                expdir=${rdir}/exp_${x}${suffix:+"_$suffix"}_indic_char/asr_${layer}

                if [ ${layer} == "noaux_ssl_ml7-11_con_e8_lin1024_bs6M_gacc1_ctc03_did" ]; then
                    postdir="/home1/Saurabh/exp/fasttext/posteriors_all_jun12/aa_tt_at/asr-did"
                    s="asr-did"
                elif [ ${layer} == "ssl_ml7-11_con_scctc" ]; then
                    postdir="/home1/Saurabh/exp/fasttext/posteriors_all/aa_tt_at/ssl-asr-did-sc"
                    s="ssl-asr-did-sc"
                elif [ ${layer} == "con_scctc" ]; then
                    postdir="/home1/Saurabh/exp/fasttext/posteriors_all/aa_tt_at/asr-did-sc"
                    s="asr-did-sc"
                elif [ ${layer} == "con_auxctc" ]; then
                    postdir="/home1/Saurabh/exp/fasttext/posteriors_all/aa_tt_at/asr-did-aux"
                    s="asr-did-aux"
                fi
                # fi

                for data in nt; do

                    sdir=${expdir}/decode_did${test_suffix:+"_$test_suffix"}_asr_model_valid.acc.ave/org/dev_${lang}_${data}${suffix:+"_$suffix"}/nbest_stats

                    cp ${sdir}/posterior.csv ${postdir}/post_${s}_dev_${lang}_${data}.csv
                done
                
                for data in public utss stus stss; do # nt utss stus stss

                    sdir=${expdir}/decode_did${test_suffix:+"_$test_suffix"}_asr_model_valid.acc.ave/test_${lang}_${data}${suffix:+"_$suffix"}/nbest_stats
                    cp ${sdir}/posterior.csv ${postdir}/post_${s}_test_${lang}_${data}.csv
                done

            fi
        done
        # wait
    done
done
