import argparse
import codecs
import os
import sys
import traceback


def get_parser():
    parser = argparse.ArgumentParser(description="prep data for lang id scoring")
    parser.add_argument(
        "--ref_file", type=str, help="Path to the reference language id file", required=True
    )
    parser.add_argument(
        "--hyp_file", type=str, help="Path to the hypothesis language id file", required=True
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The scoring output filename. " "If omitted, then output to sys.stdout",
    )
    parser.add_argument(
        "--mismatch_file",
        type=argparse.FileType("w"),
        help="File to store utterance ids with language mismatches",
    )
    parser.add_argument(
        "--lang_file",
        type=argparse.FileType("w"),
        help="File to store utterance ids with original and predicted language",
    )
    return parser


def main(args):
    args = get_parser().parse_args(args)
    scoring(args.ref_file, args.hyp_file, args.out, args.mismatch_file, args.lang_file)


def scoring(ref_file, hyp_file, out, mismatch_file, lang_file):
    ref_data = None
    hyp_data = None

    try:
        ref_data = codecs.open(ref_file, "r", encoding="utf-8")
        hyp_data = codecs.open(hyp_file, "r", encoding="utf-8")

    except Exception:
        traceback.print_exc()
        print("\nUnable to open input files.")
        sys.exit(1)

    output_file = lang_file  # Just use the provided file object directly

    output_file.write(f"utt_id\tref_lid\thyp_lid\n")

    utt_num = 0
    correct = 0

    mismatch_data = []

    while True:
        ref_line = ref_data.readline()
        hyp_line = hyp_data.readline()

        if not ref_line or not hyp_line:
            break

        ref_line = ref_line.strip().split()
        hyp_line = hyp_line.strip().split()

        utt_id_ref = ref_line[0]
        ref_lid = ref_line[1]

        utt_id_hyp = hyp_line[0]
        hyp_lid = hyp_line[1]

        if utt_id_ref != utt_id_hyp:
            print("Mismatch in utt_ids between reference and hypothesis files.")
            sys.exit(1)

        if ref_lid == hyp_lid:
            correct += 1
        else:
            mismatch_data.append(f"{utt_id_ref}\t{ref_lid}\t{hyp_lid}")

        utt_num += 1

        output_file.write(f"{utt_id_ref}\t{ref_lid}\t{hyp_lid}\n")

    out.write("Language Identification Scoring: Accuracy {:.4f} ({}/{})\n".format(
        (correct / float(utt_num)), correct, utt_num
    ))

    if mismatch_file:
        mismatch_file.write("\n".join(mismatch_data))


if __name__ == "__main__":
    main(sys.argv[1:])
