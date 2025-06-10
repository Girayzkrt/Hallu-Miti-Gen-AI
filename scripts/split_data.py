def split(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            if i >= 1000:
                break
            outfile.write(line)

input_path = '../data/parsed_pmc_2.jsonl'
output_path = '../data/parsed_pmc_2_1000.jsonl'
split(input_path, output_path)
