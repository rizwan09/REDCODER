import json
concode_base_path='/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/'
splits = ['train', 'valid', 'test']
all_nl_length = []
all_code_length = []
for splt in splits:
    file = concode_base_path+splt+'.json'
    with open(file) as f:
        for line in f:
            json_data=json.loads(line)
            nl_len = len(json_data['nl'].split())
            all_nl_length.append(nl_len)
            code_len = len(json_data['code'].split())
            all_code_length.append(code_len)

print ("--concode--")
print ('Avg NL leng: ', sum(all_nl_length)/len(all_nl_length))
print ('Avg code leng: ', sum(all_code_length)/len(all_code_length))



csnet_base_path="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" # + lang +"/test.jsonl"'
langs=['java', 'python']
nl_lengths={l:[] for l in langs}
code_lengths={l:[] for l in langs}
for lang in langs:
    for splt in splits:
        file = csnet_base_path + lang+ '/'+splt + '.jsonl'
        with open(file) as f:
            for line in f:
                json_data = json.loads(line)
                nl_len = len(json_data['docstring_tokens'])
                nl_lengths[lang].append(nl_len)
                code_len = len(json_data['code_tokens'])
                code_lengths[lang].append(code_len)
        # for d in data:


print ("--csnet--")
for lang in langs:
    print("lang: ", lang)
    print ('Avg NL leng: ', sum(nl_lengths[lang])/len(nl_lengths[lang]))
    print ('Avg code leng: ', sum(code_lengths[lang])/len(code_lengths[lang]))


