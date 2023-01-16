import os
import stanza # parsing to UD
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils

stanzapipeline = None

def get_stanza():
    global stanzapipeline
    if stanzapipeline is None:
        stanzapipeline = stanza.Pipeline(lang='en', processors = 'tokenize,pos,lemma,depparse')
    return stanzapipeline

def parsepmb(pmbdir, outdir):
    pmbfiles = []
    for path, _, files in os.walk(pmbdir):
        pmbfiles = pmbfiles + [path + '\\' + x for x in files]
    datapointprefixes = [".".join(x.split(".")[:-2]) for x in pmbfiles]
    datapointpathdict = dict((x,{'drs':x+'.drs.clf','tokens':x+'.tok.off','raw':x+'.raw'}) for x in datapointprefixes if x+'.drs.clf' in pmbfiles and x+'.tok.off' in pmbfiles and x+'.raw' in pmbfiles)
    stanzanlp = get_stanza()
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    returnlist = []
        
    i = 0
    # We'll process each datapoint in turn.
    for dpname in datapointpathdict.keys():
        # There are certain individual PMB files that give it trouble; these are currently tracked here.
        if dpname in ("..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p04\\d2024\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p14\\d3314\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p19\\d1455\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p23\\d2513\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p39\\d2847\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p49\\d1795\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p50\\d0704\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p59\\d3229\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p74\\d0808\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p75\\d3043\\en",
                        "..\\..\\..\\Downloads\\pmb-4.0.0\\data\\en\\gold\\p80\\d0554\\en"):
            continue
        # This bit of code lets you set certain indices to skip over
        if i > -1:
            i+=1
        else:
            i+=1
            continue
        try:
            print(dpname)
            dpdir = outdir+'\\'+dpname.replace('\\','-')
            if not os.path.exists(dpdir):
                os.mkdir(dpdir)
            # read the raw file
            with open(datapointpathdict[dpname]['raw']) as f:
                textraw = f.read()
            ud_parse = conlluutils.stanza_to_conllu(stanzanlp(textraw))
            # UD parse to an output file, for testing
            ud_out = ud_parse.serialize()
            with open(dpdir+'\\'+'udparse.conll',mode='w') as f:
                nlines = f.write(ud_out)
            preprocessed = preprocessing.preprocess(ud_parse)
            preproc_out = preprocessed.serialize()
            with open(dpdir+'\\'+'preproc.conll',mode='w',encoding='utf8') as f:
                nlines = f.write(preproc_out)
            # DRSs to an output file
            ud_drss = treetodrs.getalldens(preprocessed)
            clflines = clfutils.drses_to_clf([y for x,y in ud_drss])
            with open(dpdir+'\\'+'drsoutput.clf', 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(x) for x in clflines]))
            # change both of them to simplified files
            with open(dpdir+'\\'+'drsoutputsimple.clf', 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(clfutils.simplify_clf(x)) for x in clflines]))
            pmbclf = ''
            with open(datapointpathdict[dpname]['drs']) as f:
                pmbclf = f.read()
            pmbclflines = pmbclf.split('\n')
            simplifiedpmbname = dpdir+'\\'+"pmbsimple.clf"
            with open(simplifiedpmbname, 'w',encoding='utf8') as f:
                nlines = f.write('\n'.join(clfutils.simplify_clf(pmbclflines)))
            returnlist.append((simplifiedpmbname,dpdir+'\\'+'drsoutputsimple.clf'))
        except Exception as e:
            print(e)
            continue
    return returnlist
    # TODO multiprocessing goes here

if __name__ == "__main__":
    # TODO this command line tool
    # when called from the command line it needs the exact same inputs lol
    print("I promise I'll implement this :)")