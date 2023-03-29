import os
from tqdm import tqdm
import argparse
from ud_boxer.sbn import SBNGraph
from ud_boxer.helpers import smatch_score
import logging
from clfutils import clf_to_sbn

def filepair_to_penman(filepair,maxlength=20000):
    outpairs = []
    pmbfilename = filepair[0]
    outputsbnname = "placeholder.txt"
    if os.path.exists(pmbfilename[:-4]+".penman"):
        pmbpenmanname = pmbfilename[:-4]+".penman"
        computedfilename = filepair[1]
        with open(computedfilename) as f:
            computedclfs = [x.split("\n") for x in f.read().split("\n\n")]
            if len(computedclfs)>maxlength:
                logging.warning(f"Will not convert beyond the first {maxlength} entries in {filepair[1]}")
            for i in range(min(len(computedclfs),maxlength)):
                try:
                    sbn = clf_to_sbn(computedclfs[i])
                    outputsbnname = f"{computedfilename[:-4]}{i}.sbn"
                    with open(outputsbnname,"w") as f:
                        f.write("\n".join(sbn))
                    outputpenmanname = f"{computedfilename[:-4]}.penman"
                    SBNGraph().from_path(outputsbnname).to_penman(outputpenmanname)
                    outpairs.append([pmbpenmanname,outputpenmanname])
                    os.remove(outputsbnname)
                except Exception as e:
                    if os.exists(outputsbnname):
                        os.remove(outputsbnname)
                    logging.error(e)
    return outpairs

def smatch_evaluate_penmanpairs(penmanpairs):
    scores = []
    for pair in penmanpairs:
        try:
            score = smatch_score(pair[0],pair[1])
            if score:
                scores.append(score)
            else:
                scores.append("NA")
        except Exception as e:
            logging.error(e)
            scores.append("NA")
    return scores

def evaluate_from_filepairs(filepairs):
    finalscores = []
    for pair in tqdm(filepairs):
        try:
            logging.debug(f"Converting to Penman: {pair[0]}")
            penmanpairs = filepair_to_penman(pair)
            logging.debug(f"Got {len(penmanpairs)} Penman files for {pair[1]}")
            pairscores = smatch_evaluate_penmanpairs(penmanpairs)
            logging.info(f"Got {len(pairscores)} scores for {pair[1]}")
            finalscores = finalscores + [[pair[0],pair[1],str(score)] for score in pairscores]
            for pair in penmanpairs:
                os.remove(pair[1])
        except Exception as e:
            logging.error(e)
    return finalscores

def make_boxer_penman_filepairs(pmbdir):
    pairs = []
    for path, _, files in tqdm(os.walk(pmbdir)):
        for file in files:
            if file.endswith("drs.penman"):
                boxerpath = os.path.join(path,"predicted","output.penman")
                if os.path.exists(boxerpath):
                    pairs.append([os.path.join(path,file),boxerpath])
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Various tools related to converting to and scoring Penman DRS format.")
    parser.add_argument("-o","--outfile",action="store",required=True,dest="outfile",help="the file in which to write a list of file pairs with scores in clf format")
    parser.add_argument("-f","--filepairs",action="store",required=True,dest="filepairfile",help="the file to read pairs of CLF format DRSes to create and evaluate my own Penman format")
    parser.add_argument("-p","--penmaninput",action="store_true",dest="penmaninput",help="if this flag, then the pairs in filepairs are treated as pairs of Penman format files to score")
    parser.add_argument("-l","--logfile",action="store",dest="logfile",help="where to write the log?")
    args = parser.parse_args()
    if args.logfile:
        logging.basicConfig(filename=args.logfile, encoding='utf-8', level=logging.INFO)
    with open(args.filepairfile) as f:
        filepairs = [x.split("\t") for x in f.read().split("\n")]
    if args.penmaninput:
        scores = smatch_evaluate_penmanpairs(filepairs)
        outlist = [[filepairs[i][0],filepairs[i][1],str(scores[i])] for i in range(min(len(scores),len(filepairs)))]
        with open(args.outfile, "w") as f:
            f.write("\n".join(",".join(x) for x in outlist))
    else:
        finalscores = evaluate_from_filepairs(filepairs)
        with open(args.outfile, "w") as f:
            f.write("\n".join(",".join(x) for x in finalscores))
