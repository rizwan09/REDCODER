path='/home/rizwan/DPR_models/external-knowledge-codegen/apidocs/processed/distsmpl/snippet_15k/'
snippets = []
intents = []
import os, json
for file in os.listdir(path):
    if file.endswith(".jsonl"):
        with open(os.path.join(path, file)) as rf:
            for line in rf:
                data = json.loads(line)
                snippets.append(data["snippet"])
                intents.append(data["intent"])
with open(path+'unique_snippets.txt', 'w') as wf:
    for snip in set(snippets):
        wf.write(snip+"\n")
with open(path+'unique_intents.txt', 'w') as wf:
    for intnt in set(intents):
        wf.write(intnt+"\n")
print ("Total unique intents written: ", len(set(intents)))
print ("Total unique snippets written: ", len(set(snippets)))