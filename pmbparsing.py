import os
import stanza # parsing to UD
import logging
from multiprocessing import Process, Manager, freeze_support # To allow things to sometimes time out.
from queue import Empty as EmptyException
os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils

stanzapipeline = None

def get_stanza():
    global stanzapipeline
    if stanzapipeline is None:
        stanzapipeline = stanza.Pipeline(lang='en', processors = 'tokenize,pos,lemma,depparse',logging_level="CRITICAL")
    return stanzapipeline

def getalldens_proc_wrapper(tree, queue, logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    dens = treetodrs.getalldens(tree)
    queue.put(dens)
    return

def parsepmb_proc(datapointpathdict,outdir,queue,list,workingqueue,logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    toggle = 0
    while toggle==0:
        try:
            stanzanlp = get_stanza()
            toggle=1
        except PermissionError:
            continue
    while True:
        try:
            dpname = queue.get_nowait()
            logging.info(dpname)
            dpdir = outdir+'\\'+dpname.replace('\\','-')
            if not os.path.exists(dpdir):
                os.mkdir(dpdir)
            # read the raw file
            with open(datapointpathdict[dpname]['raw']) as f:
                textraw = f.read()
            logging.debug(f"Successfully read file for {dpname}.")
            # UD parse, and print to output file
            ud_parse = conlluutils.stanza_to_conllu(stanzanlp(textraw))
            ud_out = ud_parse.serialize()
            with open(dpdir+'\\'+'udparse.conll',mode='w') as f:
                nlines = f.write(ud_out)
            logging.debug(f"Successfully performed UD parsing for {dpname}.")
            preprocessed = preprocessing.preprocess(ud_parse)
            preproc_out = preprocessed.serialize()
            with open(dpdir+'\\'+'preproc.conll',mode='w',encoding='utf8') as f:
                nlines = f.write(preproc_out)
            logging.debug(f"Successfully performed UD preprocessing for {dpname}.")
            # DRSs to an output file
            #ud_drss = treetodrs.getalldens(preprocessed)
            treetodrsproc = Process(target=getalldens_proc_wrapper,args=(preprocessed,workingqueue,dpdir+'\\treetodrslog'),daemon=True)
            treetodrsproc.start()
            timeoutsecs = 300 # seconds
            treetodrsproc.join(timeout=timeoutsecs) #
            treetodrsproc.terminate()
            if treetodrsproc.exitcode is None or workingqueue.empty():
                logging.warning("Parsing datapoint:{dpname} took too long (over {timeoutsecs} seconds).")
                continue
            ud_drss = workingqueue.get_nowait()
            logging.debug(f"Found {len(ud_drss)} DRS parses for {dpname}.")
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
            logging.debug(f"Successfully performed DRS simplification for {dpname}.")
            list.append((simplifiedpmbname,simplifieddrsname))
        except EmptyException:
            return
        except Exception as e:
            logging.exception(str(Exception))

def parsepmb(pmbdir, outdir, nproc=8, logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+"-main.log", encoding='utf-8', level=logging.DEBUG,force=True)
    pmbfiles = []
    for path, _, files in os.walk(pmbdir):
        pmbfiles = pmbfiles + [path + '\\' + x for x in files]
    datapointprefixes = [".".join(x.split(".")[:-2]) for x in pmbfiles]
    datapointpathdict = dict((x,{'drs':x+'.drs.clf','tokens':x+'.tok.off','raw':x+'.raw'}) for x in datapointprefixes if x+'.drs.clf' in pmbfiles and x+'.tok.off' in pmbfiles and x+'.raw' in pmbfiles)
    logging.info(f"Found {len(datapointpathdict)} files to process.")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    man = Manager()
    datainputqueue = man.Queue()
    for datapoint in datapointpathdict.keys():
        datainputqueue.put(datapoint)
    dataoutputlist = man.list()
    workingqueues = [man.Queue()  for i in range(nproc)]
    procs = [Process(target=parsepmb_proc, args=(datapointpathdict,outdir,datainputqueue,dataoutputlist,workingqueues[i],f"{logfilepfx}-proc{i}"), daemon=False) for i in range(nproc)]
    logging.info(f"Successfully created the {nproc} parsing processes.")
    for proc in procs:
        proc.start()
    logging.info("Successfully started the parsing processes.")
    for proc in procs:
        proc.join()
        proc.terminate()
    logging.info("All parsing processes have finished.")
    if logfilepfx is not None:
        logging.basicConfig(force=True)
    return list(dataoutputlist)

if __name__ == "__main__":
    freeze_support()
    # TODO this command line tool
    # when called from the command line it needs the exact same inputs lol
    print("I promise I'll implement this :)")