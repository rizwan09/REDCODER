import os
lang='python'
top_k_ori=5
top_k = 2
dir='/local/wasiahmad/workspace/projects/RaProLanG/data/csnet/'+lang+'_with_ref_top_'+str(top_k_ori)
output_dir='/local/rizwan/workspace/projects/RaProLanG/data/csnet/'+lang+'_with_ref_top_'+str(top_k)
os.system('mkdir -p '+output_dir)


for split in ['train', 'valid', 'test']:
    input_file=dir+'/'+split
    output_file=output_dir+'/'+split
    os.system('cp '+input_file+'.target'+' '+output_dir)
    with open(input_file+'.source') as source, open(output_file+'.source', 'w') as target:
        for line in source:
            line = ' _CODE_SEP_ '.join(line.split('_CODE_SEP_')[:top_k+1])
            target.write(line+"\n")
