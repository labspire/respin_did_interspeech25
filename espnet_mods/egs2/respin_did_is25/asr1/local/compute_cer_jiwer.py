import argparse
import jiwer

def load_kaldi_text(file_path):
    """Load a Kaldi-format text file and return a dictionary of uttid to text."""
    text_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                uttid, text = parts
                text_data[uttid] = text
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return text_data

def convert_to_space_separated_chars(text):
    """Convert text to space-separated characters."""
    return " ".join(list(text.replace(" ", "")))

def compute_cer(reference, hypothesis):
    """Compute CER by treating it as WER on space-separated characters."""
    ref_chars = convert_to_space_separated_chars(reference)
    hyp_chars = convert_to_space_separated_chars(hypothesis)

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    ref_transformed = transformation(ref_chars)
    hyp_transformed = transformation(hyp_chars)

    measures = jiwer.compute_measures(ref_transformed, hyp_transformed)
    return measures['wer']

def main():
    parser = argparse.ArgumentParser(description="Compute CER between reference and hypothesis text files in Kaldi format.")
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference text file.")
    parser.add_argument("--hyp", type=str, required=True, help="Path to the hypothesis text file.")
    args = parser.parse_args()

    # Load reference and hypothesis texts
    ref_data = load_kaldi_text(args.ref)
    hyp_data = load_kaldi_text(args.hyp)

    # Ensure only common uttids are considered
    common_uttids = set(ref_data.keys()) & set(hyp_data.keys())

    if not common_uttids:
        print("No common utterance IDs found between reference and hypothesis.")
        return

    total_cer = 0
    total_utterances = 0

    for uttid in common_uttids:
        ref_text = ref_data[uttid]
        hyp_text = hyp_data[uttid]
        cer = compute_cer(ref_text, hyp_text)
        total_cer += cer
        total_utterances += 1

    overall_cer = total_cer / total_utterances if total_utterances > 0 else 0

    print(f"Overall CER: {overall_cer:.4f}")

if __name__ == "__main__":
    main()
