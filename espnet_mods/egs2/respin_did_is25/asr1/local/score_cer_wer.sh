#!/usr/bin/env bash
ref="data_respin/test_com_all/text"
hyp="exp_respin/asr_train_asr_conformer_hier_lid_utt_raw_all_bpe6500_sp/decode_lid_asr_model_valid.acc.best/test_com_all/text"
cer=true
wer=true
wdir=RESULTS/test_com_all
nlsyms_txt=data_respin/list_lang
stage=1
use_dial_file=false
dial_file="data_respin/test_com_all/utt2dial"
. utils/parse_options.sh

mkdir -p $wdir

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail

if [ $stage -le 1 ]; then
    if [ $use_dial_file = true ]; then
        echo "Using utt2dial file for dialect-wise scoring"
        cat $ref |awk '{printf "%s", $1; for(i=2;i<=NF;i++) {printf " %s", $i}; printf "\n"}' >${wdir}/text_ref_word
        cat $hyp |awk '{printf "%s", $1; for(i=2;i<=NF;i++) {printf " %s", $i}; printf "\n"}' >${wdir}/text_hyp_word
        # cat $ref | awk '{printf "%s", $1; for(i=2;i<=NF;i++) {for(j=1;j<=length($i);j++) {printf " %s", substr($i,j,1)}}; printf "\n"}' > ${wdir}/text_ref_char
        # cat $hyp | awk '{printf "%s", $1; for(i=2;i<=NF;i++) {for(j=1;j<=length($i);j++) {printf " %s", substr($i,j,1)}}; printf "\n"}' > ${wdir}/text_hyp_char
        paste <(cat $ref |awk '{print $1}') <(cat $ref |awk '{printf "%s", $2; for(i=3;i<=NF;i++) {printf " %s", $i}; printf "\n"}' |sed -E 's/.{1}/& /g') >${wdir}/text_ref_char
        paste <(cat $hyp |awk '{print $1}') <(cat $hyp |awk '{printf "%s", $2; for(i=3;i<=NF;i++) {printf " %s", $i}; printf "\n"}' |sed -E 's/.{1}/& /g') >${wdir}/text_hyp_char

        for x in $(cat $dial_file |sort -u -k2,2 |awk '{print $2}'); do
            cat $dial_file |grep -Fw "$x" |awk '{print $1}' >${wdir}/uttids_ref_${x}
        done
    else
        echo "Using text file for dialect-wise scoring"
        cat $ref |awk '{printf "%s", $1; for(i=3;i<=NF;i++) {printf " %s", $i}; printf "\n"}' >${wdir}/text_ref_word
        cat $hyp |awk '{printf "%s", $1; for(i=3;i<=NF;i++) {printf " %s", $i}; printf "\n"}' >${wdir}/text_hyp_word
        # cat $ref | awk '{printf "%s", $1; for(i=3;i<=NF;i++) {for(j=1;j<=length($i);j++) {printf " %s", substr($i,j,1)}}; printf "\n"}' > ${wdir}/text_ref_char
        # cat $hyp | awk '{printf "%s", $1; for(i=3;i<=NF;i++) {for(j=1;j<=length($i);j++) {printf " %s", substr($i,j,1)}}; printf "\n"}' > ${wdir}/text_hyp_char
        paste <(cat $ref |awk '{print $1}') <(cat $ref |awk '{printf "%s", $3; for(i=4;i<=NF;i++) {printf " %s", $i}; printf "\n"}' |sed -E 's/.{1}/& /g') >${wdir}/text_ref_char
        paste <(cat $hyp |awk '{print $1}') <(cat $hyp |awk '{printf "%s", $3; for(i=4;i<=NF;i++) {printf " %s", $i}; printf "\n"}' |sed -E 's/.{1}/& /g') >${wdir}/text_hyp_char

        for x in $(cat $nlsyms_txt); do
            # cat $ref |grep -Fi "$x" |awk '{print $1}' >${wdir}/uttids_ref_${x}
            cat $ref |grep -Fw "$x" |awk '{print $1}' >${wdir}/uttids_ref_${x}
        done
    fi
fi

if [ $stage -le 2 ]; then
    echo "Scoring CER and WER"
    if [ "${cer}" = true ]; then
        echo "Scoring CER"

        python3 /home1/Saurabh/exp/kaldi/RESPIN/SEP1323/QC_ASR/cer_scripts/compute_cer.py --ref ${wdir}/text_ref_char --hyp ${wdir}/text_hyp_char --out $wdir/utt2csid_cer

        >${wdir}/cer_all

        for x in $(cat $nlsyms_txt); do
            cat $wdir/utt2csid_cer |grep -Fwf ${wdir}/uttids_ref_${x} |awk -v s="$x" '{x=x+$4+$5+$6; y=y+$3+$4+$6} END {print s "\t" x/y*100}' >>${wdir}/cer_all
        done

        cat $wdir/utt2csid_cer |awk '{x=x+$4+$5+$6; y=y+$3+$4+$6} END {print "all" "\t" x/y*100}' >>${wdir}/cer_all
    fi

    if [ "${wer}" = true ]; then
        echo "Scoring WER"

        python3 /home1/Saurabh/exp/kaldi/RESPIN/SEP1323/QC_ASR/cer_scripts/compute_cer.py --ref ${wdir}/text_ref_word --hyp ${wdir}/text_hyp_word --out $wdir/utt2csid_wer

        >${wdir}/wer_all

        for x in $(cat $nlsyms_txt); do
            cat $wdir/utt2csid_wer |grep -Fwf ${wdir}/uttids_ref_${x} |awk -v s="$x" '{x=x+$4+$5+$6; y=y+$3+$4+$6} END {print s "\t" x/y*100}' >>${wdir}/wer_all
        done

        cat $wdir/utt2csid_wer |awk '{x=x+$4+$5+$6; y=y+$3+$4+$6} END {print "all" "\t" x/y*100}' >>${wdir}/wer_all
    fi
fi