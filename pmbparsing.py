import os
import stanza # parsing to UD
import logging
from multiprocessing import Process, Manager, freeze_support # To allow things to sometimes time out.
from queue import Empty as EmptyException
import argparse # command line arguments
import json # reading in dictionaries
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

def getalldens_proc_wrapper(tree, queue, withtrace=False, logfilepfx=None):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+".log", encoding='utf-8', level=logging.DEBUG)
    try:
        dens = treetodrs.getalldens(tree,withtrace)
        queue.put(dens)
    except ConnectionRefusedError:
        logging.exception("unable to append datapoint's dens to queue.")
        queue.put([])
    except Exception as e:
        logging.exception(str(e))
        queue.put([])
    return

def parsepmb_proc(datapointpathdict,outdir,queue,outlistfull,workingqueue,storetypes=False,logfilepfx=None,boxermappingpath=None,maxoutputs=100000):
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
            fulldrsname = os.path.join(dpdir,'drsoutput.clf')
            fullpmbname = datapointpathdict[dpname]['drs']
            if os.path.exists(fulldrsname):
                logging.debug(f"Already found output for {dpname}.")
                outlistfull.append((fullpmbname,fulldrsname))
                continue
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
            treetodrsproc = Process(target=getalldens_proc_wrapper,args=(preprocessed,workingqueue,storetypes,os.path.join(dpdir,'treetodrslog')),daemon=True)
            treetodrsproc.start()
            timeoutsecs = 300 # seconds
            treetodrsproc.join(timeout=timeoutsecs) #
            treetodrsproc.terminate()
            if treetodrsproc.exitcode is None or workingqueue.empty():
                logging.warning(f"Parsing datapoint:{dpname} took too long (over {timeoutsecs} seconds).")
                continue
            ud_drss = workingqueue.get_nowait()
            logging.debug(f"Found {len(ud_drss)} DRS parses for {dpname}. Writing the first {maxoutputs}.")
            if storetypes:
                clflines = clfutils.drses_to_clf([y for x,y,z in ud_drss])
            else:
                clflines = clfutils.drses_to_clf([y for x,y in ud_drss])
            clflines = clflines[:maxoutputs]
            boxermappingdict = {}
            if boxermappingpath:
                boxerdictposfile = os.path.join(boxermappingpath,rightlang+"_lemma_pos_sense_lookup_gold.json")
                boxerdictsensefile = os.path.join(boxermappingpath,rightlang+"_lemma_sense_lookup_gold.json")
                if os.path.exists(boxerdictposfile):
                    with open(boxerdictposfile) as f:
                        boxerdictpos = json.load(f)
                    boxerdictpos = dict((k.lower(), v.lower()) for k,v in boxerdictpos.items())
                else:
                    boxerdictpos = {}
                if os.path.exists(boxerdictsensefile):
                    with open(boxerdictsensefile) as f:
                        boxerdictsense = json.load(f)
                    boxerdictsense = dict((k.lower(), v.lower()) for k,v in boxerdictsense.items())
                else:
                    boxerdictsense = {}
                boxermappingdict.update(boxerdictsense)
                boxermappingdict.update(boxerdictpos)
            # add one always-there useless parse
            clflines = [postprocessing.postprocess_clf(x,boxermappingdict) for x in clflines] + [["b0 REF x1"]]
            fulldrsname = os.path.join(dpdir,'drsoutput.clf')
            with open(fulldrsname, 'w',encoding='utf8') as f:
                nlines = f.write('\n\n'.join(['\n'.join(x) for x in clflines]))
            fullpmbname = datapointpathdict[dpname]['drs']
            logging.debug(f"Successfully performed DRS parsing for {dpname}.")
            outlistfull.append((fullpmbname,fulldrsname))
            # If we want to store all the types of relations and such we should do so.
            if storetypes:
                typespathname = os.path.join(dpdir,"typesused.txt")
                typeslists = [treetodrs.tracetosemtypes(x) for x in ud_drss]
                with open(typespathname,'w',encoding='utf8') as f:
                    nlines = f.write('\n\n'.join('\n'.join(' '.join(y) for y in x) for x in typeslists))
        except EmptyException:
            return
        except ConnectionError:
            queuetries = queuetries+1
            if queuetries>20:
                logging.exception(f"Too many ConnectionErrors (at least {queuetries}):\n"+str(Exception))
                return
        except Exception as e:
            logging.exception(str(Exception))

def parsepmb(pmbdir, outdir, nproc=8, storetypes=False, logfilepfx=None,boxermappingpath=None,maxoutputs=100000):
    if logfilepfx is not None:
        logging.basicConfig(filename=logfilepfx+"-main.log", encoding='utf-8', level=logging.DEBUG,force=True)
    pmbfiles = []
    for path, _, files in tqdm(os.walk(pmbdir),desc="finding files:"):
        pmbfiles = pmbfiles + [os.path.join(path,x) for x in files]
    datapointprefixes = [".".join(x.split(".")[:-2]) for x in pmbfiles]
    datapointpathdict = dict((x,{'drs':x+'.drs.clf','tokens':x+'.tok.off','raw':x+'.raw'}) for x in tqdm(datapointprefixes,desc="getting paths to clf and raw") if x+'.drs.clf' in pmbfiles and x+'.tok.off' in pmbfiles and x+'.raw' in pmbfiles)
    logging.info(f"Found {len(datapointpathdict)} files to process.")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    man = Manager()
    datainputqueue = man.Queue()
    for datapoint in datapointpathdict.keys():
        datainputqueue.put(datapoint)
    dataoutputlistfull = man.list()
    workingqueues = [man.Queue()  for i in range(nproc)]
    procs = [Process(target=parsepmb_proc, args=(datapointpathdict,outdir,datainputqueue,dataoutputlistfull,workingqueues[i],storetypes,(f"{logfilepfx}-proc{i}" if logfilepfx is not None else None),boxermappingpath,maxoutputs), daemon=False) for i in range(nproc)]
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
    return list(dataoutputlistfull)

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run UD parsing and then conversion to PMB for a batch of files.")
    parser.add_argument("-p","--pmbdir",action="store",required=True,dest="pmbdir",help="the directory to (recursively) search for input PMB files")
    parser.add_argument("-o","--outdir",action="store",required=True,dest="outdir",help="the directory in which to write output files")
    parser.add_argument("-f","--fulloutfile",action="store",required=False,dest="fulloutfile",help="the file to write full file pairs for evaluation")
    parser.add_argument("-n","--nproc",action="store",default=8,dest="nproc",type=int, help="how many processes to use when running the parsing?")
    parser.add_argument("-t","--storetypes",action="store_true",dest="storetypes",help="whether to store a special file saying what types were assigned to each word and relation in each DRS")
    parser.add_argument("-l","--logfilepfx",action="store",dest="logfilepfx",help="where to write the log? just the prefix as different endings will be added")
    parser.add_argument("-b","--boxermappingpath",action="store",dest="boxermappingpath",help="optional path to a folder to take Boxer synset mappings from")
    parser.add_argument("-m","--maxoutputs",action="store",default=100000,type=int,dest="maxoutputs",help="Maximum outputs allowed for a single datapoint; by default 100000")
    args = parser.parse_args()
    filepairsfull = parsepmb(args.pmbdir,args.outdir,nproc=int(args.nproc),storetypes=args.storetypes,logfilepfx=args.logfilepfx,maxoutputs=int(args.maxoutputs))
    if args.fulloutfile is not None:
        with open(args.fulloutfile,"w") as f:
            f.write('\n'.join(['\t'.join(x) for x in filepairsfull]))

