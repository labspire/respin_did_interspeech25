import argparse
from scipy.stats import ttest_rel

def perform_t_test(baseline, proposed):
    # Perform paired t-test
    t_stat, p_value = ttest_rel(proposed, baseline)
    print("T-Test Results:")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Perform paired t-test between baseline and proposed accuracies.")
    parser.add_argument("--baseline", nargs='+', type=float, required=True, help="List of baseline accuracies.")
    parser.add_argument("--proposed", nargs='+', type=float, required=True, help="List of proposed accuracies.")

    args = parser.parse_args()

    # Ensure baseline and proposed have the same length
    if len(args.baseline) != len(args.proposed):
        print("Error: Baseline and proposed lists must have the same length.")
        return

    # Perform t-test
    perform_t_test(args.baseline, args.proposed)

if __name__ == "__main__":
    main()

