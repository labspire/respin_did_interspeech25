import argparse

parser = argparse.ArgumentParser(description="Download and format FLEURS dataset")
parser.add_argument(
    "--data_dir",
    default="dump/raw/train_all",
    type=str,
    help="directory containing asr data",
)

args = parser.parse_args()

with open(f"{args.data_dir}/text", "r", encoding="utf-8") as in_file, open(f"{args.data_dir}/utt2dial", "r", encoding="utf-8") as in_file2, open(
    f"{args.data_dir}/did_utt", "w", encoding="utf-8"
) as utt_file, open(f"{args.data_dir}/did_tok", "w", encoding="utf-8") as tok_file:
    lines = in_file.readlines()
    lines2 = in_file2.readlines()
    for line,line2 in zip(lines,lines2):
        utt_id = line.split()[0]
        lid = line[line.index("[") : line.index("]") + 1]
        did = line2[line2.index("[") : line2.index("]") + 1]
        utt_file.write(f"{utt_id} {did} \n")

        words = line[line.index("]") + 1 :]
        dids = [did for word in words.split()]
        tok_file.write(f"{utt_id} {did} {' '.join(dids)}\n")
