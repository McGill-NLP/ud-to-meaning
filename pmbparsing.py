import os
import stanza # parsing to UD
import logging
from multiprocessing import Process, Manager, freeze_support # To allow things to sometimes time out.
from queue import Empty as EmptyException
import argparse # command line arguments
#os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils
import postprocessing
from tqdm import tqdm

def get_stanza(language='en'):
    procs = 'tokenize,pos,lemma,depparse' if language in ('en','nl') else 'tokenize,mwt,pos,lemma,depparse'
    return stanza.Pipeline(lang=language, processors = procs,logging_level="CRITICAL")

def getalldens_proc_wrapper(tree, queue, logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    try:
        dens = treetodrs.getalldens(tree)
        queue.put(dens)
    except Exception as e:
        logging.exception(str(e))
    return

def parsepmb_proc(datapointpathdict,outdir,queue,outlistsimple,outlistfull,workingqueue,logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    stanzanlp = {}
    for language in ('en','de','it','nl'):
        toggle = 0
        while toggle==0:
            try:
                stanzanlp[language] = get_stanza(language=language)
                toggle=1
            except PermissionError:
                continue
    queuetries = 0
    while True:
        try:
            dpname = queue.get_nowait()
            logging.info(dpname)
            rightlang = ""
            for language in ('en','de','it','nl'):
                if language in dpname:
                    rightlang = language
                    break
            if rightlang == "":
                logging.warning(f"No language found for datapoint {dpname}.")
                rightlang = "en"
            logging.info(f"Assuming language {rightlang} for datapoint {dpname}.")
            dpdir = os.path.join(outdir,dpname.replace('\\','-').replace('/','-').replace('.',''))
            if not os.path.exists(dpdir):
                os.mkdir(dpdir)
            # read the raw file
            with open(datapointpathdict[dpname]['raw']) as f:
                textraw = f.read()
            logging.debug(f"Successfully read file for {dpname}.")
            # UD parse, and print to output file
            ud_parse = conlluutils.stanza_to_conllu(stanzanlp[rightlang](textraw))
            ud_out = ud_parse.serialize()
            with open(os.path.join(dpdir,'udparse.conll'),mode='w') as f:
                nlines = f.write(ud_out)
            logging.debug(f"Successfully performed UD parsing for {dpname}.")
            preprocessed = preprocessing.preprocess(ud_parse)
            preproc_out = preprocessed.serialize()
            with open(os.path.join(dpdir,'preproc.conll'),mode='w',encoding='utf8') as f:
                nlines = f.write(preproc_out)
            logging.debug(f"Successfully performed UD preprocessing for {dpname}.")
            # DRSs to an output file
            #ud_drss = treetodrs.getalldens(preprocessed)
            treetodrsproc = Process(target=getalldens_proc_wrapper,args=(preprocessed,workingqueue,os.path.join(dpdir,'treetodrslog')),daemon=True)
            treetodrsproc.start()
            timeoutsecs = 300 # seconds
            treetodrsproc.join(timeout=timeoutsecs) #
            treetodrsproc.terminate()
            if treetodrsproc.exitcode is None or workingqueue.empty():
                logging.warning(f"Parsing datapoint:{dpname} took too long (over {timeoutsecs} seconds).")
                continue
            ud_drss = workingqueue.get_nowait()
            logging.debug(f"Found {len(ud_drss)} DRS parses for {dpname}.")
            clflines = clfutils.drses_to_clf([y for x,y in ud_drss])
            # add one always-there useless parse
            clflines = [postprocessing.postprocess_clf(x) for x in clflines] + [["b0 REF x1"]]
            fulldrsname = os.path.join(dpdir,'drsoutput.clf')
            with open(fulldrsname, 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(x) for x in clflines]))
            # change both of them to simplified files
            simplifieddrsname = os.path.join(dpdir,'drsoutputsimple.clf')
            with open(simplifieddrsname, 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(clfutils.simplify_clf(x)) for x in clflines]))
            pmbclf = ''
            fullpmbname = datapointpathdict[dpname]['drs']
            with open(fullpmbname) as f:
                pmbclf = f.read()
            pmbclflines = pmbclf.split('\n')
            simplifiedpmbname = os.path.join(dpdir, "pmbsimple.clf")
            with open(simplifiedpmbname, 'w',encoding='utf8') as f:
                nlines = f.write('\n'.join(clfutils.simplify_clf(pmbclflines)))
            logging.debug(f"Successfully performed DRS simplification for {dpname}.")
            outlistsimple.append((simplifiedpmbname,simplifieddrsname))
            outlistfull.append((fullpmbname,fulldrsname))
        except EmptyException:
            return
        except ConnectionError:
            queuetries = queuetries+1
            if queuetries>20:
                logging.exception(f"Too many ConnectionErrors (at least {queuetries}):\n"+str(Exception))
                return
        except Exception as e:
            logging.exception(str(Exception))

def parsepmb(pmbdir, outdir, nproc=8, logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+"-main.log", encoding='utf-8', level=logging.DEBUG,force=True)
    pmbfiles = []
    for path, _, files in tqdm(os.walk(pmbdir)):
        pmbfiles = pmbfiles + [os.path.join(path,x) for x in files]
    datapointprefixes = [".".join(x.split(".")[:-2]) for x in pmbfiles]
    datapointpathdict = dict((x,{'drs':x+'.drs.clf','tokens':x+'.tok.off','raw':x+'.raw'}) for x in datapointprefixes if x+'.drs.clf' in pmbfiles and x+'.tok.off' in pmbfiles and x+'.raw' in pmbfiles)
    logging.info(f"Found {len(datapointpathdict)} files to process.")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    man = Manager()
    datainputqueue = man.Queue()
    for datapoint in datapointpathdict.keys():
        datainputqueue.put(datapoint)
    dataoutputlistsimple = man.list()
    dataoutputlistfull = man.list()
    workingqueues = [man.Queue()  for i in range(nproc)]
    procs = [Process(target=parsepmb_proc, args=(datapointpathdict,outdir,datainputqueue,dataoutputlistsimple,dataoutputlistfull,workingqueues[i],(f"{logfilepfx}-proc{i}" if logfilepfx is not None else None)), daemon=False) for i in range(nproc)]
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
    return list(dataoutputlistsimple), list(dataoutputlistfull)

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run UD parsing and then conversion to PMB for a batch of files.")
    parser.add_argument("-p","--pmbdir",action="store",required=True,dest="pmbdir",help="the directory to (recursively) search for input PMB files")
    parser.add_argument("-o","--outdir",action="store",required=True,dest="outdir",help="the directory in which to write output files")
    parser.add_argument("-s","--simpleoutfile",action="store",required=False,dest="simpleoutfile",help="the file to write simplified file pairs for evaluation")
    parser.add_argument("-f","--fulloutfile",action="store",required=False,dest="fulloutfile",help="the file to write full file pairs for evaluation")
    parser.add_argument("-n","--nproc",action="store",default=8,dest="nproc",type=int, help="how many processes to use when running the parsing?")
    parser.add_argument("-l","--logfilepfx",action="store",dest="logfilepfx",help="where to write the log? just the prefix as different endings will be added")
    args = parser.parse_args()
    filepairssimple, filepairsfull = parsepmb(args.pmbdir,args.outdir,nproc=int(args.nproc),logfilepfx=args.logfilepfx)
    if args.simpleoutfile is not None:
        with open(args.simpleoutfile,"w") as f:
            f.write('\n'.join(['\t'.join(x) for x in filepairssimple]))
    if args.fulloutfile is not None:
        with open(args.fulloutfile,"w") as f:
            f.write('\n'.join(['\t'.join(x) for x in filepairsfull]))

