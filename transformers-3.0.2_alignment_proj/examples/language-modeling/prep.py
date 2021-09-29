import numpy as np
from sklearn.model_selection import train_test_split

l=19438
x, y = train_test_split(np.arange(l), test_size=0.1)
input_path="../../data/encoded-aln.txt"

train_data=[]
val_data=[]



with open(input_path) as f:
    for i, line in enumerate(f):
        if i in x:
            train_data.append(line)
        else:
            val_data.append(line)

print("total: ", i, "train: ", len(train_data), "val: ", len(val_data))

with open("../../data/train.txt", "w") as tf, open("../../data/eval.txt", "w") as vf:
    for line in train_data:
        tf.write(line)
    for line in val_data:
        vf.write(line)