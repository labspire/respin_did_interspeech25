import os
import argparse
from collections import defaultdict

def create_dialect_word_frequency_dict(file_path):
    dialect_word_frequency = defaultdict(lambda: defaultdict(int))

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            utterance_id, dialect_id, text = line.split(maxsplit=2)
            words = text.split()

            for word in words:
                dialect_word_frequency[dialect_id][word] += 1

    return dialect_word_frequency

def create_word_frequency_dict(file_path):
    # word_frequency = defaultdict(lambda: defaultdict(int))
    word_frequency = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            utterance_id, dialect_id, text = line.split(maxsplit=2)
            words = text.split()

            for word in words:
                word_frequency[word] += 1

    return word_frequency

def create_lid_utt_tok_files(file_path, word_dialect_dict, utt_file_path, tok_file_path):
    with open(file_path, "r", encoding="utf-8") as in_file, open(utt_file_path, "w", encoding="utf-8") as utt_file, open(tok_file_path, "w", encoding="utf-8") as tok_file:
        lines = in_file.readlines()
        for line in lines:
            utt_id = line.split()[0]
            # assign 2nd column as language id and words as 3rd column onwards
            lid = line.split()[1]
            utt_file.write(f"{utt_id} {lid} \n")
            
            lids = []
            for word in line.split()[2:]:
                if word not in word_dialect_dict:
                    lids.append('0')
                else:
                    lids.append(str(word_dialect_dict[word]))
            
            # lids = [word_dialect_dict[word] for word in words]
            tok_file.write(f"{utt_id} {lid} {' '.join(lids)}\n")

def main():
    parser = argparse.ArgumentParser(description='Create a dictionary with word frequencies for each dialect.')
    parser.add_argument("--data_dir", default="dump/raw/train_all", type=str, help="directory containing asr data")
    parser.add_argument("--text", default="text", type=str, help="text file name")
    parser.add_argument("--lid_utt", default="lid_utt", type=str, help="lid_utt file name")
    parser.add_argument("--lid_tok", default="lid_tok", type=str, help="lid_tok file name")

    args = parser.parse_args()
    
    file_path = os.path.join(args.data_dir, args.text)
    utt_file_path = os.path.join(args.data_dir, args.lid_utt)
    tok_file_path = os.path.join(args.data_dir, args.lid_tok)
    
    dial_word_freq = create_dialect_word_frequency_dict(file_path)
    word_freq = create_word_frequency_dict(file_path)
    
    word_prob = defaultdict(lambda: defaultdict(float))
    word_dial = defaultdict(str)
    for word,freq in word_freq.items():
        for dial in dial_word_freq.keys():
            if word in dial_word_freq[dial]:
                word_prob[word][dial] = dial_word_freq[dial][word]/freq
            else:
                word_prob[word][dial] = 0.0
                
        word_dial[word] = int(max(word_prob[word], key=word_prob[word].get))

    
    create_lid_utt_tok_files(file_path, word_dial, utt_file_path, tok_file_path)
    

if __name__ == "__main__":
    main()
