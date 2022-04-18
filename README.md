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
- Sample SCODE-R output: [code to text valid split top 30 k retrievals](https://drive.google.com/file/d/1OZHQWajhq4F4TVs7Oiqmze_MK4XZoC_N/view?usp=drive_web)
- Finetuned SCODE-R checkpoints w/ all the above retrieval database and sample outputs together: [LINK](https://drive.google.com/drive/folders/1qrt6DkJKp43TNN_wCTVCP98BOtD-9o9J?usp=sharing)
