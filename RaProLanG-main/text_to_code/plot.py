import numpy as np
import matplotlib.pyplot as plt
import pdb
from matplotlib.legend import Legend


plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

LANGS = "ar	eu	bg	ca	zh	hr	cs	da	nl	fi	fr	de	iw	hi 	in	it	ja	ko	no	fa	pl	pt	ro	ru	sr	sk	sl	es	sv	tr".split()
YMIN = -0.035
YMAX = 0.09


X_SIZE = 15
Y_SIZE = 5

FONTSIZE = "medium"
BACKGROUND_COLOR = 'blue'
CODER = 'CodeR'
BM25='BM25'
REDCODER = 'REDCODER'
TICK_FONTSIZE = 30





def plot_two_curves():
    values = [
        (CODER+'-Java', [20.97672295755363 , 26.389776357827476 , 29.475125513464175 , 31.64764947512551,\
                    33.25422181652213 , 34.53217708808763 , 35.3445915107257 , 36.248288452761294, \
                 37.10634413509813, 37.845732542218165]),
        (CODER + '-Python', [19.178174017964874 , 24.896098672744333, 27.97291862179917 , 30.09116503552755, \
                              31.54578361710685,32.96018233007105,33.945569111140905 ,34.89073602359566, \
                              35.58117710148814 ,36.171068507842875]),
        (BM25 + '-Java', [5.7234139662254675, 8.251939753537197, 9.685075308078503, 10.725696029210406, \
                          11.419443176631674, 12.295755362848015, 12.816065723413967, 13.39114559561844,
                          14.267457781834777, 14.687357371063442]),
        (BM25 + '-Python',
         [8.868481029628636, 11.804531438530635, 13.31277651159673, 14.512669258613755, 15.350583188094918, \
          16.10805738034589, 16.73816865531573, 17.254323635876123, 17.797291862179918, 18.212897171202574]),

    ]

    print("plotting")
    plt.figure(figsize=(6, 6))


    ymin = 0
    ymax = 30
    num_x = 10
    plt.xticks(range(num_x), np.arange(num_x)+1, fontsize=25)
    FMTs = ["b", "r-", "c-" , "m-", "y-", "g-", "burlywood", '#1F6333', 'blue', 'magenta', '#FF5733', '#F78F1E', "r-", ]

    linestyles = ['-', '--', '-', '--', '--', '--', '--', '-']
    linewidths = [5, 5, 3.5, 3.5, 1.5, 1.75, 2.0, 2.25]
    markers = ["P", "X", 'o', 's', ">", 'D']

    values = values

    for i in range(len(values)):
        name, val = values[i]


        try:
            plt.plot(val, FMTs[i],label=name,linestyle=linestyles[i], \
            marker=markers[i],
                 markersize=linewidths[i]+4, linewidth=linewidths[i]
                     )
        except:
            import pdb
            pdb.set_trace()

    plt.legend(bbox_to_anchor=(0.35, 0.8), loc='upper left', ncol=1, fontsize=FONTSIZE)

    plt.xlabel("k: # of retrieved ", fontsize=TICK_FONTSIZE)
    plt.ylabel( "Recall@k", fontsize=TICK_FONTSIZE)

    plt.grid(linestyle='-')
    plt.gca().xaxis.grid(True)
    plt.margins(0.02, 0.04)
    # plt.ylim(ymin, ymax)
    # plt.axvspan(0, 12, alpha=0.025, color=BACKGROUND_COLOR)

    plt.tight_layout()
    # plt.show()
    print("saving!!")
    plt.savefig('bm25_vs_coder_value.png', bbox_inches='tight')




def plot_xy_python_summ_curves():
    values = [
        # ('Java', [19.3, 21.23, 22.94]),
        ('Python\nSum.\nw target ret.', [18.45, 45.98, 46.41, 46.76, 46.96]),
    ]

    print("plotting")
    plt.figure(figsize=(5, 3))


    ymin = 18
    ymax = 50
    num_x = 10
    plt.xticks(range(num_x), [0,5,10,30,50], fontsize=25)
    FMTs = ["#F78F1E", "r-", "c-" , "m-", "y-", "g-", "burlywood", '#1F6333', 'blue', 'magenta', '#FF5733', '#F78F1E', "r-", ]

    linestyles = ['-', '--', '-', '--', '--', '--', '--', '-']
    linewidths = [5, 5, 3.5, 3.5, 1.5, 1.75, 2.0, 2.25]
    markers = ["o", "X", 'o', 's', ">", 'D']

    values = values

    for i in range(len(values)):
        name, val = values[i]


        try:
            plt.plot(val, FMTs[i],label=name,linestyle=linestyles[i], \
            marker=markers[i],
                 markersize=linewidths[i]+4, linewidth=linewidths[i]
                     )
        except:
            import pdb
            pdb.set_trace()

    plt.legend(bbox_to_anchor=(0.17, 0.8), loc='upper left', ncol=1, fontsize=FONTSIZE)

    plt.xlabel("#Retrievals", fontsize=TICK_FONTSIZE)
    plt.ylabel( "BLEU-4", fontsize=TICK_FONTSIZE)

    plt.grid(linestyle='-')
    plt.gca().xaxis.grid(True)
    plt.margins(0.02, 0.04)
    # plt.ylim(ymin, ymax)
    # plt.axvspan(0, 12, alpha=0.025, color=BACKGROUND_COLOR)

    plt.tight_layout()
    # plt.show()
    print("saving!!")
    plt.savefig('sum_ret_python_bleu_four.png', bbox_inches='tight')

def plot_xy_java_summ_curves():
    values = [
        # ('Java', [19.3, 21.23, 22.94]),
        ('Java\nSum.\nw/o target ret.', [19.3, 21.23, 22.94, 22.55]),
    ]

    print("plotting")
    plt.figure(figsize=(5, 3))


    ymin = 18
    ymax = 23
    num_x = 10
    plt.xticks(range(num_x), [0,10,30, 50], fontsize=25)
    FMTs = ["#F78F1E", "r-", "c-" , "m-", "y-", "g-", "burlywood", '#1F6333', 'blue', 'magenta', '#FF5733', '#F78F1E', "r-", ]

    linestyles = ['-', '--', '-', '--', '--', '--', '--', '-']
    linewidths = [5, 5, 3.5, 3.5, 1.5, 1.75, 2.0, 2.25]
    markers = ["o", "X", 'o', 's', ">", 'D']

    values = values

    for i in range(len(values)):
        name, val = values[i]


        try:
            plt.plot(val, FMTs[i],label=name,linestyle=linestyles[i], \
            marker=markers[i],
                 markersize=linewidths[i]+4, linewidth=linewidths[i], fillstyle=None
                     )
        except:
            import pdb
            pdb.set_trace()

    plt.legend(bbox_to_anchor=(0.17, 0.8), loc='upper left', ncol=1, fontsize=FONTSIZE)

    plt.xlabel("#Retrievals", fontsize=TICK_FONTSIZE)
    plt.ylabel( "BLEU-4", fontsize=TICK_FONTSIZE)

    plt.grid(linestyle='-')
    plt.gca().xaxis.grid(True)
    plt.margins(0.02, 0.04)
    # plt.ylim(ymin, ymax)
    # plt.axvspan(0, 12, alpha=0.025, color=BACKGROUND_COLOR)

    plt.tight_layout()
    # plt.show()
    print("saving!!")
    plt.savefig('sum_ret_java_bleu_four.png', bbox_inches='tight')


def plot_xy_java_gen_curves():
    values = [
        # ('Java', [19.3, 21.23, 22.94]),
        ('Java\nGen.\nw target ret.', [0,23.34,27.42,27.16,29.41]),
    ]

    print("plotting")
    plt.figure(figsize=(5, 3))


    ymin = 18
    ymax = 23
    num_x = 10
    plt.xticks(range(num_x), [0,1,2,3,4], fontsize=25)
    FMTs = ["#F78F1E", "r-", "c-" , "m-", "y-", "g-", "burlywood", '#1F6333', 'blue', 'magenta', '#FF5733', '#F78F1E', "r-", ]

    linestyles = ['-', '--', '-', '--', '--', '--', '--', '-']
    linewidths = [5, 5, 3.5, 3.5, 1.5, 1.75, 2.0, 2.25]
    markers = ["o", "X", 'o', 's', ">", 'D']

    values = values

    for i in range(len(values)):
        name, val = values[i]


        try:
            plt.plot(val, FMTs[i],label=name,linestyle=linestyles[i], \
            marker=markers[i],
                 markersize=linewidths[i]+4, linewidth=linewidths[i], fillstyle=None
                     )
        except:
            import pdb
            pdb.set_trace()

    plt.legend(bbox_to_anchor=(0.19, 0.8), loc='upper left', ncol=1, fontsize=FONTSIZE)

    plt.xlabel("#Retrievals", fontsize=TICK_FONTSIZE)
    plt.ylabel( "EM", fontsize=TICK_FONTSIZE)

    plt.grid(linestyle='-')
    plt.gca().xaxis.grid(True)
    plt.margins(0.02, 0.04)
    # plt.ylim(ymin, ymax)
    # plt.axvspan(0, 12, alpha=0.025, color=BACKGROUND_COLOR)

    plt.tight_layout()
    # plt.show()
    print("saving!!")
    plt.savefig('gen_ret_java_EM_four.png', bbox_inches='tight')

def plot_xy_python_gen_curves():
    values = [
        # ('Java', [19.3, 21.23, 22.94]),
        ('Python\nGen.\nw/o target ret.', [4.89,7.89,8.68,8.88]),
    ]

    print("plotting")
    plt.figure(figsize=(5, 3))


    ymin = 18
    ymax = 23
    num_x = 10
    plt.xticks(range(num_x), [0,2,4,5], fontsize=25)
    FMTs = ["#F78F1E", "r-", "c-" , "m-", "y-", "g-", "burlywood", '#1F6333', 'blue', 'magenta', '#FF5733', '#F78F1E', "r-", ]

    linestyles = ['-', '--', '-', '--', '--', '--', '--', '-']
    linewidths = [5, 5, 3.5, 3.5, 1.5, 1.75, 2.0, 2.25]
    markers = ["o", "X", 'o', 's', ">", 'D']

    values = values

    for i in range(len(values)):
        name, val = values[i]


        try:
            plt.plot(val, FMTs[i],label=name,linestyle=linestyles[i], \
            marker=markers[i],
                 markersize=linewidths[i]+4, linewidth=linewidths[i], fillstyle=None
                     )
        except:
            import pdb
            pdb.set_trace()

    plt.legend(bbox_to_anchor=(0.19, 0.8), loc='upper left', ncol=1, fontsize=FONTSIZE)

    plt.xlabel("#Retrievals", fontsize=TICK_FONTSIZE)
    plt.ylabel( "EM", fontsize=TICK_FONTSIZE)

    plt.grid(linestyle='-')
    plt.gca().xaxis.grid(True)
    plt.margins(0.02, 0.04)
    # plt.ylim(ymin, ymax)
    # plt.axvspan(0, 12, alpha=0.025, color=BACKGROUND_COLOR)

    plt.tight_layout()
    # plt.show()
    print("saving!!")
    plt.savefig('gen_ret_python_EM_four.png', bbox_inches='tight')



num_examples_java=[ 1962, 2470, 1685, 1142, 1715, 1971]
num_examples_python=[3059, 3088, 2172, 1475, 2233, 2874]

sequence_len=[40, 60, 80, 100, 150, 500]

plbart_java=[0.08885573162969604, 0.1303200652468112, 0.12017483701220719, 0.0998331041121777, 0.07971244942750906, 0.03642097073393836]
plbart_python=[0.10104668334574766, 0.07649092289991484, 0.050263672284854684, 0.04054149436051039, 0.028167232206048148, 0.01251723990000324]

retrieved_java=[0.17832993212905313, 0.16982412501700647, 0.1791569096763891, 0.17488795242502558, 0.17725466777386273, 0.1928365955203575]
retrieved_python=[0.12786706114210097, 0.1407866940928393, 0.12980852804549858, 0.14490172457460532, 0.1771703769057607, 0.18214743053709637]

redcoder_java=[0.20608282323804197, 0.23727162562475484, 0.25728351301117597, 0.2583843018410592, 0.27145554266235933, 0.18460523877458423]
redcoder_python=[0.4026392965991861, 0.43559834383468404, 0.43061419682802876, 0.4026302488360376, 0.37553473913208085, 0.160594517616099]

redcoder_ext_java=[0.25049314299522396, 0.273769133008136, 0.2810408458939894, 0.2744861257870286, 0.28920982328081274, 0.2022020259825037]
redcoder_ext_python=[0.2001864853169571, 0.21974598032434753, 0.22315140706864717, 0.23516221388387143, 0.26948933451748436, 0.15413773558294208]


if __name__ == '__main__':
    plot_two_curves()
    # plot_xy_curves()
    plot_xy_python_summ_curves()
    plot_xy_java_summ_curves()
    plot_xy_java_gen_curves()
    plot_xy_python_gen_curves()