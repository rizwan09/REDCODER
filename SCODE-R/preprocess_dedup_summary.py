import json
iclr_data_path='/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/c_functions_all_data.jsonl'


langs = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

summaries = []

for line in open(iclr_data_path):
    data = json.loads(line)
    summaries.append(data['summary'])

# for lang in langs:
#     csnet_train_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/train.jsonl"
#     csnet_valid_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/valid.jsonl"
#     csnet_test_data_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/test.jsonl"
#
#     for line in open(csnet_train_data_path):
#         data = json.loads(line)
#         summaries.append(' '.join(data['docstring_tokens']))
#     for line in open(csnet_valid_data_path):
#         data = json.loads(line)
#         summaries.append(' '.join(data['docstring_tokens']))
#     for line in open(csnet_test_data_path):
#         data = json.loads(line)
#         summaries.append(' '.join(data['docstring_tokens']))


print(len(summaries))
print(len(set(summaries)))

with open('iclr_only.deduplicated.summaries.txt', 'w') as sf:
    for s in set(summaries):
        sf.write(s+"\n")
