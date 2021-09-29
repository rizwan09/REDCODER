
import numpy as np
import csv
import json
import jsonlines
import random
import bleu_code
import weighted_ngram_match
import syntax_match
import dataflow_match
#----------------------------------------------------------------------------------------

#Usage:
# python human_ann.py

#----------------------------------------------------------------------------------------



np.random.seed(1) #1 generates low bleu scores for hyps but high codebleu score
'''
np.random.seed(1)

DPR: 
Ngram match:    11.56
Weighted ngram: 11.90
Syntax match:   29.96
Dataflow match: 12.87
CodeBLEU score: 16.57
EM:  0.0

REDCODER: 
Ngram match:    20.18
Weighted ngram: 20.81
Syntax match:   35.89
Dataflow match: 25.41
CodeBLEU score: 25.57
EM:  0.13333333333333333

'''
max_code_length = 50
num_samples = 30
lang = 'python' #NLP10
# lang = 'java' #NLP11



redcoder_pred_path = '/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-'+lang+'-with-top-5-retrived-from-no-ref-no-mask-'+lang+'-ms100000-wu5000-bsz72'
# redcoder_pred_path = '/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-'+lang+'-comments-top-5-without-'+lang+'-ms100000-wu5000-bsz64'
# ref = redcoder_pred_path+'/'+'output'
hyp_file = redcoder_pred_path+'/'+'output.hyp'



ref_file='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/'+lang+'_retrievd_from_no_ref_top_5_mask_rate_0/test.target'
dpr_file='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/'+lang+'_retrievd_from_no_ref_top_5_mask_rate_0/test.source'


refs = []
dprs = []
hyps = []

max_len = 1000

with open(ref_file) as ref_f, open(dpr_file) as dpr_f, open(hyp_file) as hype_f:
    for ref, dpr, hyp in zip(ref_f, dpr_f, hype_f):
        dpr = dpr.split('_CODE_SEP_')[1]
        if len(ref.split()) < max_len and len(dpr.split()) < max_len and len(hyp.split()) < max_len and dpr.strip()!=ref.strip():
            refs.append(ref.strip())
            dprs.append(dpr.strip())
            hyps.append(hyp.strip())

print('len(refs): ', len(refs))
rand_nums = np.random.choice(len(refs), num_samples)

def compute_bleu(refss, hypss, lang='python'):
    # preprocess inputs
    pre_references = [[ref.strip() for ref in refss]]
    hypothesis = [hyp.strip() for hyp in hypss]

    for i in range(len(pre_references)):
        if len(hypothesis) != len(pre_references[i]):
            import ipdb
            ipdb.set_trace()

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references) * len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]



    ngram_match_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)
    # calculate weighted ngram match
    keywords = [x.strip() for x in open('keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    alpha, beta, gamma, theta = 0.25, 0.25, 0.25, 0.25

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'. \
    #      format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
    print('Ngram match:\t%.2f\nWeighted ngram:\t%.2f\nSyntax match:\t%.2f\nDataflow match:\t%.2f' % ( \
        ngram_match_score * 100, weighted_ngram_match_score * 100, syntax_match_score * 100,
        dataflow_match_score * 100))

    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    print('CodeBLEU score: %.2f' % (code_bleu_score * 100.0))


refss=[]
dprss=[]
hypss=[]
correct_dpr = 0
correct_redcoder = 0
# with open("comment_annotation.txt", 'w') as anf, open('comment_ann.csv', 'w') as csvfile:
with open("annotation.txt", 'w') as anf, open('ann.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i, rand in enumerate(rand_nums):
        ref = refs[rand]
        dpr = dprs[rand]
        hyp = hyps[rand]
        refss.append(ref)
        dprss.append(dpr)
        hypss.append(hyp)

        if dpr.strip()==ref.strip(): correct_dpr+=1
        if dpr.strip()==hyp.strip(): correct_redcoder+=1


        anf.write(("===" *30)+"\n")
        print("i: ", i)
        print(("===" *30)+"\n")
        anf.write("REF: "+ ref + "\n")
        print("REF: "+ ref + "\n")
        anf.write(("-" * 30) + "\n")
        print(("-" * 30) + "\n")
        anf.write("DPR Prediction: "+dpr + "\n")
        print("DPR Prediction: "+dpr + "\n")
        anf.write(("-" * 30) + "\n")
        print(("-" * 30) + "\n")
        print(("-" * 30) + "\n")
        anf.write(("-" * 30) + "\n")
        anf.write("REF: "+ ref + "\n")
        print("REF: "+ ref + "\n")
        anf.write(("-" * 30) + "\n")
        print(("-" * 30) + "\n")
        anf.write("REDCODER Prediction: "+hyp + "\n")
        print("REDCODER Prediction: "+hyp + "\n")

        csvwriter.writerow([rand, "Ref: " + ref + '\n\n' + "DPR Prediction: " +  dpr, 'a_'+str(i), 'b_'+str(i)])
        csvwriter.writerow([rand, "Ref: " + ref + '\n\n' + "REDCODER Prediction: " +hyp, 'a_'+str(i), 'b_'+str(i)])
        csvwriter.writerow([ '--'*20 ])

        #
        # import ipdb
        # ipdb.set_trace()

print("DPR: ")
compute_bleu(refss, dprss, lang=lang)
print('EM: ', correct_dpr/len(refss))
print("REDCODER: ")
compute_bleu(refss, hypss, lang=lang)
print('EM: ', correct_redcoder
      /len(refss))



# if __name__ == "__main__":
    # main()







