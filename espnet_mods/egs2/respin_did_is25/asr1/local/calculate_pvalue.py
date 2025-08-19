import argparse
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def compute_p_value(proposed_file, baseline_file, utt2dial_file):
    # Load the TSV files
    proposed_data = pd.read_csv(proposed_file, sep="\t", header=None, names=["uttid", "predicted_dialect"])
    baseline_data = pd.read_csv(baseline_file, sep="\t", header=None, names=["uttid", "predicted_dialect"])
    true_data = pd.read_csv(utt2dial_file, sep="\t", header=None, names=["uttid", "true_dialect"])

    # Merge the data on uttid to align predictions and true labels
    merged_data = pd.merge(true_data, proposed_data, on="uttid")
    merged_data = pd.merge(merged_data, baseline_data, on="uttid", suffixes=("_proposed", "_baseline"))

    # Create contingency table for McNemar's test
    # [0][0]: Both methods are incorrect
    # [0][1]: Proposed method correct, baseline incorrect
    # [1][0]: Proposed method incorrect, baseline correct
    # [1][1]: Both methods are correct
    contingency_table = [[0, 0], [0, 0]]

    for _, row in merged_data.iterrows():
        true_dialect = row["true_dialect"]
        proposed_correct = row["predicted_dialect_proposed"] == true_dialect
        baseline_correct = row["predicted_dialect_baseline"] == true_dialect

        if proposed_correct and baseline_correct:
            contingency_table[1][1] += 1  # Both correct
        elif proposed_correct and not baseline_correct:
            contingency_table[0][1] += 1  # Proposed correct, baseline incorrect
        elif not proposed_correct and baseline_correct:
            contingency_table[1][0] += 1  # Proposed incorrect, baseline correct
        else:
            contingency_table[0][0] += 1  # Both incorrect

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)
    p_value = result.pvalue

    print("Contingency Table:")
    print(pd.DataFrame(contingency_table, columns=["Baseline Incorrect", "Baseline Correct"],
                                       index=["Proposed Incorrect", "Proposed Correct"]))
    print(f"\nP-value: {p_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute p-value between proposed and baseline methods.")
    parser.add_argument("--proposed_file", default="/home1/Saurabh/exp/espnet/RESPIN/DID_ASR/CTC_AUX_EXP/exp_bn_s12345_indic_char/asr_noaux_ssl_ml7-11_con_e8_lin1024_bs6M_gacc1_ctc03_ls5_conv1d_bneck_rob_hs64_gelu_asr_hs1024_layer2/decode_lid_asr_model_valid.acc.ave/test_bn_nt/text_did", help="Path to the proposed method TSV file.")
    parser.add_argument("--baseline_file", default="/home1/Saurabh/exp/espnet/RESPIN/DID_ASR/CTC_AUX_EXP/exp_bn_s12345_did_indic_char/asr_ssl_ml7-11_con_e8_bs6M_gacc1_ctc03_ls5_rob_hs64_v2/decode_lid_asr_model_valid.acc.ave/test_bn_nt_did/text_did", help="Path to the baseline method TSV file.")
    parser.add_argument("--utt2dial_file", default="/home1/Saurabh/exp/espnet/RESPIN/DID_ASR/CTC_AUX_EXP/data_bn/test_bn_nt/utt2dial", help="Path to the TSV file with true dialect labels.")

    args = parser.parse_args()
    compute_p_value(args.proposed_file, args.baseline_file, args.utt2dial_file)
