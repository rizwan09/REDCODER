import os, ipdb, csv
import json
import sys
import weighted_ngram_match
import syntax_match
import dataflow_match
import bleu_code
from bleu import _bleu

base_path='/home/rizwan/DPR_models/'
output_path=base_path+'prediction_ori_raw_top_1_bleu_70.30.txt'
output_path='/local/rizwan/workspace/projects/RaProLanG/plbart_ori_top_1_masked/-java-ms100000-wu5000-bsz64/output.hyp'
output_path='/local/rizwan/workspace/projects/RaProLanG/maksed_with_top_2-java-ms100000-wu5000-bsz64/output.hyp'
input_path=base_path+'source_ori_raw_top_1.txt'
target_path=base_path+'test.target'
annotation_path=base_path+'input_target_output_coder.csv'
plbart_pred_path=base_path+'plbart_ori_pred.txt'

ensamble_preds_path=base_path+'ensamble_pred.txt'
retrieved_code_path=base_path+'retrieved.txt'

def check_retrived_acc(retrieved_code_path, max_k=10, lang='python'):
    with open(retrieved_code_path) as f:
        retrieved_code = json.load(f)
        scores = [[0 for i in range(max_k)] for j in range(len(retrieved_code))]

        refss = []
        hypss = []

        dpr_scores= [ ]


        for idx, ex in enumerate(retrieved_code):
            try:
                target = ex['answers'].strip()
            except:
                ex = retrieved_code[ex]
                target = ex['answers'].strip()
            refss.append(target)

            for rank, ctx in enumerate(ex['ctxs']):
                try:
                    dpr_scores.append(ctx['score'])
                except:
                    dpr_scores.append(ctx['_score'])
                cand = ctx["text"].strip()
                if rank==0:
                    hypss.append(cand)
                if cand==target:
                    for j in range(rank, max_k): scores[idx][j] = 1
    for i in range(max_k):
        EM = sum([score[i] for score in scores])/len(retrieved_code)
        print("At top ", i, " EM/Recall: ", EM*100, 'dpr score: ', dpr_scores[i])

    if lang == 'js':
        lang = 'javascript'
    alpha, beta, gamma, theta = 0.25, 0.25, 0.25, 0.25

    # preprocess inputs
    pre_references = [[ref.strip() for ref in refss]]
    hypothesis = [hyp.strip() for hyp in hypss]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

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


def check_retrived_acc_other_than_without_ref(retrieved_code_path, max_k=10, lang="python"):
    with open(retrieved_code_path) as f:
        retrieved_code = json.load(f)
        scores = [[] for j in range(len(retrieved_code))]

        refss = []
        hypss = []
        dpr_scores=[]

        for idx, ex in enumerate(retrieved_code):
            try:
                target = ex['answers'].strip()
            except:
                ex = retrieved_code[ex]
                target = ex['answers'].strip()
            refss.append(target)
            inserted=False
            for rank, ctx in enumerate(ex['ctxs']):
                cand = ctx["text"].strip()
                if cand!=target:
                    if not inserted:
                        hypss.append(cand)
                        inserted=True
                    scores[idx].append(_bleu(target, cand))
                    try:
                        dpr_scores.append(ctx['score'])
                    except:
                        dpr_scores.append(ctx['_score'])
            if not inserted:
                hypss.append("")
            if len(scores[idx])!=max_k:
                for i in range(len(scores[idx]), max_k): scores[idx].append(0)

    min_len = min([len(score) for score in scores[:-1]])
    print (min_len)

    for i in range(min_len):
        Blue = sum([score[i] for score in scores])/len(retrieved_code)
        print("At top ", i, " Bleu: ", Blue, 'dpr score: ', dpr_scores[i])


    if lang == 'js':
        lang = 'javascript'
    alpha, beta, gamma, theta = 0.25, 0.25, 0.25, 0.25

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






def get_num_word_tokens(code):
    count = 0
    for w in code.split():
        if len(w)>1: count+=1
    return count


def error_analysis():
    with open(annotation_path, 'w') as csvfile, \
        open(ensamble_preds_path, 'w') as ensamble_pred_f, \
        open(retrieved_code_path, 'w') as retrieved_f:


        csvwriter = csv.writer(csvfile)
        tars = [x.strip() for x in open(target_path, 'r', encoding='utf-8').readlines()]
        pres = [x.strip() for x in open(output_path, 'r', encoding='utf-8').readlines()]
        plbart_preds = [x.strip() for x in open(plbart_pred_path, 'r', encoding='utf-8').readlines()]

        retrievd_codes = [x.split('_CODE_SEP_')[-1].strip() for x in open(input_path, 'r', encoding='utf-8').readlines()]
        NLs = [x.split('concode')[0].strip() for x in open(input_path, 'r', encoding='utf-8').readlines()]


        correct_pred_coount = 0
        correct_retrival_count = 0
        copy_cpount = 0
        not_copied_but_correct = 0
        copied_and_correct = 0



        for id, (nl, retrievd_code, output, target) in enumerate(zip(NLs, retrievd_codes, pres, tars)):
            retrieved_f.write(retrievd_code+"\n")
            blue_score=_bleu(target, output)
            # print(target, output, blue_score)
            csvwriter.writerow([nl, retrievd_code,  target, output, output==target, retrievd_code==target, retrievd_code==output, blue_score])
            if output==target:
                correct_pred_coount+=1
            if retrievd_code==target:
                correct_retrival_count+=1
            if retrievd_code==output:
                copy_cpount+=1
                ensamble_pred_f.write(output+"\n")
                if output==target:
                    copied_and_correct+=1
            else:
                # print("wring: ", output, "->",  plbart_preds[id])
                if get_num_word_tokens(retrievd_code)>15:
                    ensamble_pred_f.write(output + "\n")
                else:
                    ensamble_pred_f.write(plbart_preds[id] + "\n")
                if output==target:
                    not_copied_but_correct+=1

        print("Acc/Em: ", correct_pred_coount, "in percentage: ", correct_pred_coount/len(tars)*100)
        print("Retrieved/: ", correct_retrival_count, "in percentage: ", correct_retrival_count/len(tars)*100)
        print("Copied: ", copy_cpount, "in percentage: ", copy_cpount/len(tars)*100)
        print("Not Copied: ", len(tars)-copy_cpount, "in percentage: ", (len(tars)-copy_cpount)/len(tars)*100)
        print("Not Copied but correct: ", not_copied_but_correct, "in percentage of not copied: ", not_copied_but_correct/(len(tars)-copy_cpount)*100)
        print("Copied and correct: ", copied_and_correct, "in percentage of copied: ", copied_and_correct/(copy_cpount)*100)

# error_analysis()



def call_ret(lang, k=3):

    retrievd_test_file='/local/rizwan/DPR_models/csnet/'+lang+'_csnet_pos_only_retrieval_dedup_test_30.json'
    print(retrievd_test_file)
    print("---" * 50)
    print("With Ref:: ")
    print("---" * 50)
    check_retrived_acc(retrievd_test_file, max_k=k,  lang=lang)
    print("---"*50)
    print("Without Ref:: ")
    print("---" * 50)
    check_retrived_acc_other_than_without_ref(retrievd_test_file, max_k=k, lang=lang)



# lang='java'
# call_ret(lang, k=1)
# lang='python'
# call_ret(lang, k=1)


#
#
# retrievd_test_file='/local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/python_csnet_pos_only_retrieval_dedup_test_20.json'
# print(retrievd_test_file)
# check_retrived_acc(retrievd_test_file, lang="python")
# check_retrived_acc_other_than_without_ref(retrievd_test_file, lang="python")
# #
# # #
# retrievd_test_file='/local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/java_csnet_pos_only_retrieval_dedup_test_20.json'
# print(retrievd_test_file)
# check_retrived_acc(retrievd_test_file, lang="java")
# check_retrived_acc_other_than_without_ref(retrievd_test_file)



retrievd_test_file='/local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/test_100.json'
check_retrived_acc(retrievd_test_file, lang="java")
check_retrived_acc_other_than_without_ref(retrievd_test_file)


exit(0)

# RETDIR='/local/rizwan/DPR_models/csnet/'
# langs=[ 'python', 'java']
# test_ret_paths = { lang: RETDIR+lang+'_csnet_pos_only_retrieval_dedup_test_30.json' for lang in langs}
# for lang in langs:
#     print('lang: ', lang, 'with')
#     check_retrived_acc(test_ret_paths[lang], lang=lang)
#     print('lang: ', lang, 'without')
#     check_retrived_acc_other_than_without_ref(test_ret_paths[lang], lang=lang)


# RETDIR='/local/rizwan/workspace/projects/RaProLang/retrieval/bm25/'
# langs=[ 'python', 'java']
# test_ret_paths = { lang: RETDIR+'codexglue-csnet-'+lang+'.test_bm25.json' for lang in langs}
# for lang in langs:
#     print('lang: ', lang, 'with')
#     check_retrived_acc(test_ret_paths[lang], lang=lang)
#     print('lang: ', lang, 'without')
#     check_retrived_acc_other_than_without_ref(test_ret_paths[lang], lang=lang)
#

# print('Concode lang: ', lang, 'with')
# filepth='/local/rizwan/workspace/projects/RaProLang/retrieval/bm25/concode.test_bm25.json'
# check_retrived_acc(filepth, lang='java')
# print('Concode lang: ', lang, 'without')
# check_retrived_acc_other_than_without_ref(filepth, lang='java')



# RETDIR='/local/rizwan/workspace/projects/RaProLang/retrieval/bm25/code_to_text/'
# langs=[  'java', 'python',]
# test_ret_paths = { lang: RETDIR+'codexglue-csnet-'+lang+'.test_code_text_bm25.json' for lang in langs}
# for lang in langs:
#     print('lang: ', lang, 'with')
#     check_retrived_acc(test_ret_paths[lang], lang=lang)
#     print('lang: ', lang, 'without')
#     check_retrived_acc_other_than_without_ref(test_ret_paths[lang], lang=lang)



redcoder_ext_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-comments-top-5-without-python-ms100000-wu5000-bsz64/output.hyp'
redcoder_ext_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-comments-top-5-without-python-ms100000-wu5000-bsz64/output.hyp'
redcoder_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-with-top-5-retrived-from-no-ref-no-mask-python-ms100000-wu5000-bsz72/output.hyp'
plbart_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-python-with-top-0-retrived-from-with-ref-no-mask-python-ms100000-wu5000-bsz72/output.hyp'
retrived_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/python_retrievd_from_no_ref_top_5_mask_rate_0/test.source'
retrived_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet_with_comments/python_without_ref_top_5/test.source'
target_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/python_retrievd_from_no_ref_top_5_mask_rate_0/test.target'



# redcoder_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-5-retrived-from-no-ref-no-mask-java-ms100000-wu5000-bsz72/output.hyp'
# plbart_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-0-retrived-from-with-ref-no-mask-java-ms100000-wu5000-bsz72/output.hyp'
# redcoder_ext_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-comments-top-5-without-java-ms100000-wu5000-bsz72/output.hyp'
# target_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/java_retrievd_from_no_ref_top_5_mask_rate_0/test.target'
# retrived_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet_with_comments/java_without_ref_top_5/test.source'


quants=[20, 40, 60, 80, 100, 50]
num_examples = {x:0 for x in quants}
plbarts={x:[] for x in quants}
retriveds={x:[] for x in quants}
redcoders={x:[] for x in quants}
redcoders_exts={x:[] for x in quants}




# with open(redcoder_ext_f) as rd_ext, open(redcoder_f) as rd, open(plbart_f) as plbrt, open(retrived_f) as rtvd, open(target_f) as target:
#     for rdext, rd, rtr, plbt, tgt in zip(rd_ext, rd, rtvd, plbrt, target):
#
#         tgt = tgt.strip()
#         # calculate ngram match (BLEU)
#         tokenized_hyps = [plbt.split() ]
#         tokenized_refs = [[tgt.split() ] ]
#         plbart_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)
#
#
#         tokenized_hyps = [ rtr.split('_CODE_SEP_')[1].split('_NL_')[0].strip().split()]
#         rtr_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)
#
#         tokenized_hyps = [rd.strip().split()]
#         rd_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)
#
#         tokenized_hyps = [rdext.strip().split()]
#         rdext_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)
#
#         # if rdext_bleu_score > rd_bleu_score and rd_bleu_score>rtr_bleu_score and rtr_bleu_score>plbart_bleu_score and \
#         #     len(tgt.split()) >= len(rdext.strip().split()) and len(rdext.strip().split())>= len(rtr.split('_CODE_SEP_')[1].strip().split())\
#         #     and len(rtr.split('_CODE_SEP_')[1].strip().split())>= len(plbt.split() ) and rdext_bleu_score>0.3 and rd_bleu_score>0.2 \
#         #         and tgt not in rtr:
#         #
#         #     print('input: ', rtr.split('_CODE_SEP_')[0])
#         #     print("target:", tgt)
#         #
#         #
#         #     print('='*10)
#         #     print("plbart:", plbt)
#         #     print("bleu:", plbart_bleu_score)
#         #     print('=' * 10)
#         #     print("rtrvd:", rtr)
#         #     print("bleu:", rtr_bleu_score)
#         #
#         #     print('=' * 10)
#         #     print("rd:", rd)
#         #     print("bleu:", rd_bleu_score)
#         #     print('=' * 10)
#         #     print('rdext: ' ,rdext)
#         #     print("bleu:", rdext_bleu_score)
#
#
#         l=len(tgt.split())
#         for q in quants:
#             if l<q:
#                 num_examples[q]+=1
#                 plbarts[q].append(plbart_bleu_score)
#                 retriveds[q].append(rtr_bleu_score)
#                 redcoders[q].append(rd_bleu_score)
#                 redcoders_exts[q].append(rdext_bleu_score)
#                 break







# redcoder_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-5-retrived-from-no-ref-no-mask-java-ms100000-wu5000-bsz72/output.hyp'
# plbart_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-with-top-0-retrived-from-with-ref-no-mask-java-ms100000-wu5000-bsz72/output.hyp'
# redcoder_ext_f='/local/rizwan/workspace/projects/RaProLanG/plbart-codexglue-csnet-java-comments-top-5-without-java-ms100000-wu5000-bsz72/output.hyp'
# target_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/java_retrievd_from_no_ref_top_5_mask_rate_0/test.target'
# retrived_f='/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet_with_comments/java_without_ref_top_5/test.source'


quants=[20, 40, 60, 80, 100, 150, 500]
num_examples = {x:0 for x in quants}
plbarts={x:[] for x in quants}
retriveds={x:[] for x in quants}
redcoders={x:[] for x in quants}
redcoders_exts={x:[] for x in quants}


# print(num_examples)
# print("plbarts:", )
# for x, y in plbarts.items():
#     print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y)/len(y))
# print("retrievd:", )
# for x, y in retriveds.items():
#     print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y)/len(y))
# print("redcoder:", )
# for x, y in redcoders.items():
#     print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y)/len(y))
# print("redcoder-ext:", )
# for x, y in redcoders_exts.items():
#     print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y)/len(y))



with open(redcoder_ext_f) as rd_ext, open(redcoder_f) as rd, open(plbart_f) as plbrt, open(retrived_f) as rtvd, open(target_f) as target:
    for rdext, rd, rtr, plbt, tgt in zip(rd_ext, rd, rtvd, plbrt, target):

        tgt = tgt.strip()
        # calculate ngram match (BLEU)
        tokenized_hyps = [plbt.split() ]
        tokenized_refs = [[tgt.split() ] ]
        plbart_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)


        tokenized_hyps = [ rtr.split('_CODE_SEP_')[1].split('_NL_')[0].strip().split()]
        rtr_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)

        tokenized_hyps = [rd.strip().split()]
        rd_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)

        tokenized_hyps = [rdext.strip().split()]
        rdext_bleu_score = bleu_code.corpus_bleu(tokenized_refs, tokenized_hyps)

        # if rdext_bleu_score > rd_bleu_score and rd_bleu_score>rtr_bleu_score and rtr_bleu_score>plbart_bleu_score and \
        #     len(tgt.split()) >= len(rdext.strip().split()) and len(rdext.strip().split())>= len(rtr.split('_CODE_SEP_')[1].strip().split())\
        #     and len(rtr.split('_CODE_SEP_')[1].strip().split())>= len(plbt.split() ) and rdext_bleu_score>0.3 and rd_bleu_score>0.2 \
        #         and tgt not in rtr:
        #
        #     print('input: ', rtr.split('_CODE_SEP_')[0])
        #     print("target:", tgt)
        #
        #
        #     print('='*10)
        #     print("plbart:", plbt)
        #     print("bleu:", plbart_bleu_score)
        #     print('=' * 10)
        #     print("rtrvd:", rtr)
        #     print("bleu:", rtr_bleu_score)
        #
        #     print('=' * 10)
        #     print("rd:", rd)
        #     print("bleu:", rd_bleu_score)
        #     print('=' * 10)
        #     print('rdext: ' ,rdext)
        #     print("bleu:", rdext_bleu_score)


        l=len(tgt.split())
        for q in quants:
            if l<q:
                num_examples[q]+=1
                plbarts[q].append(plbart_bleu_score)
                retriveds[q].append(rtr_bleu_score)
                redcoders[q].append(rd_bleu_score)
                redcoders_exts[q].append(rdext_bleu_score)
                break



print(num_examples.keys(), num_examples.values())
print("plbarts:", )
xxxx=[]

for x, y in plbarts.items():
    if len(y)==0:
        print(x, 'num exmples ', num_examples[x])
    else:
        print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y)/len(y))
        xxxx.append(sum(y)/len(y))
print(xxxx)
xxxx=[]
print("retrievd:", )
for x, y in retriveds.items():
    if len(y) == 0:
        print(x, 'num exmples ', num_examples[x])
    else:
        print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y) / len(y))
        xxxx.append(sum(y) / len(y))
print(xxxx)
xxxx=[]
print("redcoder:", )

for x, y in redcoders.items():
    if len(y) == 0:
        print(x, 'num exmples ', num_examples[x])
    else:
        print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y) / len(y))
        xxxx.append(sum(y) / len(y))
print(xxxx)
xxxx = []
print("redcoder-ext:", )
for x, y in redcoders_exts.items():
    if len(y) == 0:
        print(x, 'num exmples ', num_examples[x])
    else:
        print("len: ", x, " number ", num_examples[x], "avg blue: ", sum(y) / len(y))
        xxxx.append(sum(y) / len(y))
print(xxxx)
xxxx = []




