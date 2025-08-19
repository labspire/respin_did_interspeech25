import argparse

parser = argparse.ArgumentParser(description="Create lid_utt and lid_tok files for asr data")
parser.add_argument(
    "--data_dir",
    default="dump/raw/train_all",
    type=str,
    help="directory containing asr data",
)

args = parser.parse_args()

# Dictionary to store word-to-dialect mapping
word_dialect_dict = {}

with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file:
    lines = in_file.readlines()

    for line in lines:
        # Split line into components
        utt_id, lid, *words = line.split()

        # Iterate through words and update the dictionary
        for word in words:
            if word not in word_dialect_dict:
                word_dialect_dict[word] = set()

            word_dialect_dict[word].add(lid)

# Create lid_utt file
with open(f"{args.data_dir}/lid_utt", "w", encoding="utf-8") as utt_file:
    for line in lines:
        utt_id, lid, *_ = line.split()
        utt_file.write(f"{utt_id} {lid}\n")

# Create lid_tok file
with open(f"{args.data_dir}/lid_tok", "w", encoding="utf-8") as tok_file:
    for line in lines:
        utt_id, lid, *words = line.split()
        tok_file.write(f"{utt_id} {' '.join(str(next(iter(word_dialect_dict.get(word, {'0'})))) if len(word_dialect_dict.get(word, {})) <= 2 else '0' for word in words)}\n")
