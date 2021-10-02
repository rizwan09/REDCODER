# RaProLanG
Retrieval-augmented Program and Language Generation

### Folder contents: 

output (model pred)

output.hyp (ref target)

finetune.log or train.log (has gold target file information and also path for the used retrved candidates)



## Ceckpoint locations:
### Summarization:

#### without ref java: 
##### NLP11: 
/local/rizwan/workspace/projects/RaProLanG/csnet_code_text_java/java_en_XX_no_30/

#### with ref java:
##### NLP11: 
/local/rizwan/workspace/projects/RaProLanG/csnet_code_text_java/java_en_XX_with_30/




#### without ref python:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/csnet_code_text_python/python_en_XX_no_10/

#### with ref python:
##### NLP11:  
/local/rizwan/workspace/projects/RaProLanG/csnet_code_text_python/python_en_XX_with_50/

#### PLBART: from PLBART



### Code Gen:


#### without ref java: 
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-5-retrived-from-no-ref-no-mask-java-ms100000-wu5000-bsz72/

#### with ref java:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-4-retrived-from-with-ref-no-mask-java-ms100000-wu5000-bsz72/

#### just PLBART java:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-0-retrived-from-with-ref-no-mask-java-ms100000-wu5000-bsz72/

## Main Redcoder: without ref but with adding summary of the retrieved code java:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-comments-top-5-without-java-ms100000-wu5000-bsz72

## Main Redcoder: with ref and with adding summary of the retrieved code java:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-comments-top-5-with-java-ms100000-wu5000-bsz72/



#### without ref python:
##### NLP10:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-with-top-5-retrived-from-no-ref-no-mask-python-ms100000-wu5000-bsz72/

#### with ref python:
##### NLP10:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-with-top-5-retrived-from-with-ref-no-mask-python-ms100000-wu5000-bsz72/


#### just PLBART python
NLP10:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-with-top-0-retrived-from-with-ref-no-mask-python-ms100000-wu5000-bsz72/


## Main Redcoder: without ref but with adding summary of the retrieved code python:
##### NLP10:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-comments-top-5-without-python-ms100000-wu5000-bsz64/

## Main Redcoder: with ref and with adding summary of the retrieved code python:
##### NLP11:
/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-comments-top-5-with-python-ms100000-wu5000-bsz72/







