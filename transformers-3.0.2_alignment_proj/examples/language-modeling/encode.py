from transformers import RobertaTokenizer, RobertaModel
import torch, numpy as np
import tqdm, os

tokenizer = RobertaTokenizer.from_pretrained('./alignement2')
model = RobertaModel.from_pretrained('./alignement2')

input_path="../../data/encoded-aln.txt"
encoddings=[]


with tqdm.tqdm(os.path.getsize(input_path)) as pbar:
   with open(input_path, "r") as f:
      for line in f:
          pbar.update(len(line))
          inputs = tokenizer(line, return_tensors="pt")
          outputs = model(**inputs)
          import ipdb

          ipdb.set_trace()
          class_rep = outputs[1][0].cpu().detach().numpy()
          encoddings.append(class_rep)



with open('alignment_encoded.npy', 'wb') as wf:
    np.save(wf, np.array(encoddings))


with open('alignment_encoded.npy', 'rb') as rf:
    encd=np.load(rf)