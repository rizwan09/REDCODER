
# News
- 🎉 Our new paper [CodeSim](https://github.com/kagnlp/CodeGenerator) got accepted in NAACL 2025 Findings.
- 🎉 Our new paper [MapCoder](https://raw.githubusercontent.com/Md-Ashraful-Pramanik/MapCoder/) got accepted in [ACL 2024](https://2024.aclweb.org/).
- All our codebase in both MapCoder and Redcoder are open-sourced with MIT and Modified MIT License.
- See you at ACL, 2024, Bangkok, Thailand. 

# REDCODER (Retrieval augmentED CODe gEneration and summaRization)

This is repository for the SCODE-R retriever in the [Retrieval Augmented Code Generation and Summarization](https://arxiv.org/abs/2108.11601) paper.

If you find this paper or this code useful, please cite this paper:
```
@inproceedings{parvez2021retrieval,
  title = {Retrieval Augmented Code Generation and Summarization},
  author = {Parvez, Md Rizwan and Ahmad, Wasi Uddin and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
  booktitle = {EMNLP-Findings},
  year = {2021}
}
```

Our model has two parts. You can use them seperately as well.
- SCODE-R: Summary and Code Retriever. Please see instructions in ```./SCODE-R```.
- SCODE-G: Summary and Code Generator. Please see instructions in ```./SCODE-G```.


## All REDCODER data/models/outputs together:
- Exclude retrieval candidate embeddings(too large)
- Exclude tokenized input to SCODE-G (by sentencepiece, we provide code and docs in ```SCODE-G``` directory. Please use them instead.)
- Please go through issues specially this [issue](https://github.com/rizwan09/REDCODER/issues/1)
- Sample SCODE-R output: [code to text valid split top 30 k retrievals](https://drive.google.com/file/d/1ktOoJc0uRG7TqfYDI0OZlsLpMnRjEmLl/view?usp=sharing)
- Finetuned SCODE-R checkpoints:
  - Code2Text Python: [Link](https://drive.google.com/file/d/13-5wAHvNQwPifiODnpFYUFJpK-8NHtWt/view?usp=sharing)
  - Text2Code Python: [Link](https://drive.google.com/file/d/1-YWPicpjynkC2sa8Mo02MhFiSvkV3ThJ/view?usp=sharing)
  - Code2Text Java: [Link](https://drive.google.com/file/d/14nAonUhEKrE7Aufg6u2eNpchaiutNxIn/view?usp=sharing)
  - Text2Code Java: [Link](https://drive.google.com/file/d/1pvolKC7o8iyGKLDCy37HXt4yH9lqTjpr/view?usp=sharing)
- All the retrieval database: (a) one combined summary retrieval corpus for code2text for both python and Java (b) Java and Python code retrieval corpus for text2Code tasks: [LINK](https://drive.google.com/drive/folders/1njGXJuPsq5Eod9Ff5zAutRULk_G0TzQr?usp=sharing)
