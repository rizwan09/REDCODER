import json
iclr_data_path='/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/c_functions_all_data.jsonl'


langs = ["go",   "javascript", "ruby", "php", "java",  "python"][-2:]
# lang_list = ['java', 'python']


# codes = {'python':{}, 'java':{}}

# for line in open(iclr_data_path):
#     data = json.loads(line)
#     for lang in lang_list:
#         codes[lang].append('')

for lang in langs:
    csnet_train_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/train.jsonl"
    csnet_valid_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/valid.jsonl"
    csnet_test_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/test.jsonl"

    codes={}
    for line in open(csnet_train_data_path):
        data = json.loads(line)
        codes.update({' '.join(data['docstring_tokens']): ' '.join(data['code_tokens'])})

    for line in open(csnet_test_data_path):
        data = json.loads(line)
        codes.update({' '.join(data['docstring_tokens']): ' '.join(data['code_tokens'])})

    for line in open(csnet_test_data_path):
        data = json.loads(line)
        codes.update({' '.join(data['docstring_tokens']): ' '.join(data['code_tokens'])})


    print(len(codes))

    with open(lang+'_summary_codes.json', 'w') as sf:
        json.dump(codes, sf)

    # cc = json.load(open(lang+'_summary_codes.json') )
