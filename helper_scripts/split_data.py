def split(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            if i >= 50000:
                break
            outfile.write(line)

input_path = '../data/parsed_pmc_1.jsonl'
output_path = '../small_data/parsed_pmc_1_small.jsonl'
split(input_path, output_path)
