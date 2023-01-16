import os
import stanza # parsing to UD
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils
import sys
from multiprocessing import Process, Manager, freeze_support # To allow things to sometimes time out.

stanzapipeline = None

def get_stanza():
    global stanzapipeline
    if stanzapipeline is None:
        stanzapipeline = stanza.Pipeline(lang='en', processors = 'tokenize,pos,lemma,depparse')
    return stanzapipeline

def getalldens_proc_wrapper(tree, queue):
    dens = treetodrs.getalldens(tree)
    queue.put(dens)
    return

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
        
    # We'll process each datapoint in turn.
    for dpname in datapointpathdict.keys():
        try:
            print(dpname)
            dpdir = outdir+'\\'+dpname.replace('\\','-')
            if not os.path.exists(dpdir):
                os.mkdir(dpdir)
            # read the raw file
            with open(datapointpathdict[dpname]['raw']) as f:
                textraw = f.read()
            # UD parse, and print to output file
            ud_parse = conlluutils.stanza_to_conllu(stanzanlp(textraw))
            ud_out = ud_parse.serialize()
            with open(dpdir+'\\'+'udparse.conll',mode='w') as f:
                nlines = f.write(ud_out)
            preprocessed = preprocessing.preprocess(ud_parse)
            preproc_out = preprocessed.serialize()
            with open(dpdir+'\\'+'preproc.conll',mode='w',encoding='utf8') as f:
                nlines = f.write(preproc_out)
            # DRSs to an output file
            #ud_drss = treetodrs.getalldens(preprocessed)
            man = Manager()
            q = man.Queue()
            treetodrsproc = Process(target=getalldens_proc_wrapper,args=(preprocessed,q))
            treetodrsproc.start()
            treetodrsproc.join(timeout=300) # seconds
            treetodrsproc.terminate()
            if treetodrsproc.exitcode is None or q.empty():
                print("Parsing this one took too long.")
                continue
            ud_drss = q.get()
            clflines = clfutils.drses_to_clf([y for x,y in ud_drss])
            with open(dpdir+'\\'+'drsoutput.clf', 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(x) for x in clflines]))
            # change both of them to simplified files
            simplifieddrsname = dpdir+'\\'+'drsoutputsimple.clf'
            with open(simplifieddrsname, 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(clfutils.simplify_clf(x)) for x in clflines]))
            pmbclf = ''
            with open(datapointpathdict[dpname]['drs']) as f:
                pmbclf = f.read()
            pmbclflines = pmbclf.split('\n')
            simplifiedpmbname = dpdir+'\\'+"pmbsimple.clf"
            with open(simplifiedpmbname, 'w',encoding='utf8') as f:
                nlines = f.write('\n'.join(clfutils.simplify_clf(pmbclflines)))
            returnlist.append((simplifiedpmbname,simplifieddrsname))
        except Exception as e:
            continue
    return returnlist
    # TODO multiprocessing goes here

if __name__ == "__main__":
    freeze_support()
    # TODO this command line tool
    # when called from the command line it needs the exact same inputs lol
    print("I promise I'll implement this :)")