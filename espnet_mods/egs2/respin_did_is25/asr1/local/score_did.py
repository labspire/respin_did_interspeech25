import argparse
import pandas as pd
from collections import defaultdict

def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['uttid', 'dialect'])

def calculate_accuracy(ref_df, pred_df):
    merged_df = pd.merge(ref_df, pred_df, on='uttid', suffixes=('_ref', '_pred'))
    overall_accuracy = (merged_df['dialect_ref'] == merged_df['dialect_pred']).mean()

    dialect_wise_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    for _, row in merged_df.iterrows():
        ref_dialect = row['dialect_ref']
        pred_dialect = row['dialect_pred']
        dialect_wise_counts[ref_dialect]['total'] += 1
        if ref_dialect == pred_dialect:
            dialect_wise_counts[ref_dialect]['correct'] += 1

    dialect_wise_accuracy = {
        dialect: counts['correct'] / counts['total'] 
        for dialect, counts in dialect_wise_counts.items()
    }
    
    return overall_accuracy, dialect_wise_accuracy

def save_accuracy_to_tsv(dialect_wise_accuracy, overall_accuracy, output_file):
    with open(output_file, 'w') as f:
        f.write("dialect\taccuracy\n")
        for dialect in sorted(dialect_wise_accuracy):
            accuracy = dialect_wise_accuracy[dialect]
            f.write(f"{dialect}\t{accuracy:.4f}\n")
        f.write(f"overall\t{overall_accuracy:.4f}\n")

def main(args):
    ref_df = load_tsv(args.ref_tsv)
    pred_df = load_tsv(args.hyp_tsv)
    
    overall_accuracy, dialect_wise_accuracy = calculate_accuracy(ref_df, pred_df)
    
    save_accuracy_to_tsv(dialect_wise_accuracy, overall_accuracy, args.out_tsv)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dialect-wise and overall accuracy from reference and predicted TSV files.")
    parser.add_argument('--ref-tsv', type=str, required=True, help="Path to the reference TSV file.")
    parser.add_argument('--hyp-tsv', type=str, required=True, help="Path to the predicted TSV file.")
    parser.add_argument('--out-tsv', type=str, required=True, help="Path to save the accuracy results TSV file.")
    
    args = parser.parse_args()
    main(args)
