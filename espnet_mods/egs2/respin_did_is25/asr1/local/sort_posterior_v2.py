import argparse
import csv

def parse_tsv_to_csv(input_file, output_file, nbest, lang):
    if lang in ['bn', 'hi', 'kn']:
        valid_labels = ['1', '2', '3', '4', '5']
    elif lang in ['ch', 'mg', 'mr', 'mt', 'te']:
        valid_labels = ['1', '2', '3', '4']
    elif lang == 'bh':
        valid_labels = ['1', '2', '3']
    
    with open(input_file, 'r') as tsv_file, open(output_file, 'w', newline='') as csv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        csv_writer = csv.writer(csv_file)

        # Write CSV header
        csv_writer.writerow(['uttids'] + ['posterior_' + label for label in valid_labels])

        for row in tsv_reader:
            uttid = row[0]
            predicted_labels = row[1:nbest+1]
            posteriors = row[nbest+1:]

            posterior_dict = {label: min(posteriors) for label in valid_labels}

            for label, posterior in zip(predicted_labels, posteriors):
                if label in valid_labels:
                    posterior_dict[label] = posterior

            csv_writer.writerow([uttid] + [posterior_dict[label] for label in valid_labels])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TSV file to CSV file with specific posteriors.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file.')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file.')
    parser.add_argument('--nbest', type=int, default=7, help='Number of best hypotheses to consider. Default is 7.')
    parser.add_argument('--lang', type=str, default='bn', help='Language of the input file.')

    args = parser.parse_args()
    parse_tsv_to_csv(args.input_file, args.output_file, args.nbest, args.lang)
