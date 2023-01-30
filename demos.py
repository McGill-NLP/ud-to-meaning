import os
import stanza # parsing to UD
import conllu # for reading conllu files
#os.chdir('C:\\Users\\Lola\\OneDrive\\UDepLambda\\computer code')
import conlluutils
import preprocessing
import treetodrs
import clfutils
import argparse

stanzapipeline = None
def get_stanza():
    global stanzapipeline
    if stanzapipeline is None:
        stanzapipeline = stanza.Pipeline(lang='en', processors = 'tokenize,pos,lemma,depparse',logging_level="CRITICAL")
    return stanzapipeline

def pmbdemo(pmbpath, partnum, docnum, simplified=False, returnall = False, withtrace=False):
    # Read the PMB CLF file,
    # (making sure input part number and document number are the right format)
    if isinstance(partnum,int):
        partnum = f'p{partnum:02d}'
    else:
        if not isinstance(partnum, str):
            partnum = str(partnum)
        if not partnum.startswith('p'):
            partnum = 'p'+partnum
    if isinstance(docnum,int):
        docnum = f'd{docnum:04d}'
    else:
        if not isinstance(docnum, str):
            docnum = str(docnum)
        if not docnum.startswith('d'):
            docnum = 'd'+docnum
    with open(os.path.join(pmbpath, str(partnum), str(docnum), r"en.drs.clf")) as f:
        pmbclfraw = f.read()
        pmbclflines = pmbclfraw.split('\n')
    if simplified:
        pmbclflines = clfutils.simplify_clf(pmbclflines)
    pmb_drs = clfutils.clf_to_drs(pmbclflines)
    # Read the raw PMB file,
    with open(os.path.join(pmbpath,str(partnum),str(docnum), r"en.raw")) as f:
        textraw = f.read()
    # Parse with Stanza,
    stanzanlp = get_stanza()
    ud_parse = conlluutils.stanza_to_conllu(stanzanlp(textraw))
    # Preprocess,
    preprocessed = preprocessing.preprocess(ud_parse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed,withtrace=withtrace)
    # Simplify if necessary,
    print(f"Found {len(ud_drss)} parses for this string. Returning {'all of them' if returnall else ('the first' if ud_drss else 'the UD parses and PMB comparison')}.")
    if returnall:
        if withtrace:
            if simplified:
                return ud_parse, preprocessed, [(clfutils.simplify_drs(ud_drs[1]),ud_drs) for ud_drs in ud_drss], pmb_drs
            else:
                return ud_parse, preprocessed, [(clfutils.simplify_drs(ud_drs[1]),ud_drs) for ud_drs in ud_drss], pmb_drs
        else:
            if simplified:
                return ud_parse, preprocessed, [clfutils.simplify_drs(ud_drs[1]) for ud_drs in ud_drss], pmb_drs
            else:
                return ud_parse, preprocessed, [clfutils.simplify_drs(ud_drs[1]) for ud_drs in ud_drss], pmb_drs
    if ud_drss:
        if withtrace:
            if simplified:
                return ud_parse, preprocessed, (clfutils.simplify_drs(ud_drss[0][1]),ud_drss[0]), pmb_drs
            else:
                return ud_parse, preprocessed, (ud_drss[0][1],ud_drss[0]), pmb_drs
        else:
            if simplified:
                return ud_parse, preprocessed, clfutils.simplify_drs(ud_drss[0][1]), pmb_drs
            else:
                return ud_parse, preprocessed, ud_drss[0][1], pmb_drs
    else:
        return ud_parse, preprocessed, None, pmb_drs

def conlludemo(conllupath, returnall = False, withtrace=False):
    # Read the conllu,
    with open(conllupath) as f:
        conlluraw = f.read()
    conlluparse = conllu.parse(conlluraw)[0]
    # Preprocess,
    preprocessed = preprocessing.preprocess(conlluparse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed,withtrace=withtrace)
    # Return all the outputs.
    print(f"Found {len(ud_drss)} parses for this tree. Returning {'all of them' if returnall else ('the first' if ud_drss else 'the UD parses')}.")
    if returnall:
        if withtrace:
            return conlluparse, preprocessed, [(ud_drs[1],ud_drs) for ud_drs in ud_drss]
        else:
            return conlluparse, preprocessed, [ud_drs[1] for ud_drs in ud_drss]
    if ud_drss:
        if withtrace:
            return conlluparse, preprocessed, (ud_drss[0][1], ud_drss[0])
        else:
            return conlluparse, preprocessed, ud_drss[0][1]

def sentencedemo(text, returnall=False, withtrace=False):
    # Parse with Stanza,
    stanzanlp = get_stanza()
    ud_parse = conlluutils.stanza_to_conllu(stanzanlp(text))
    # Preprocess,
    preprocessed = preprocessing.preprocess(ud_parse)
    # Compute the meaning,
    ud_drss = treetodrs.getalldens(preprocessed,withtrace=withtrace)
    # Return all the outputs.
    print(f"Found {len(ud_drss)} parses for this string. Returning {'all of them' if returnall else ('the first' if ud_drss else 'the UD parses')}.")
    if returnall:
        if withtrace:
            return ud_parse, preprocessed, [(ud_drs[1],ud_drs) for ud_drs in ud_drss]
        else:
            return ud_parse, preprocessed, [ud_drs[1] for ud_drs in ud_drss]
    if ud_drss:
        if withtrace:
            return ud_parse, preprocessed, (ud_drss[0][1], ud_drss[0])
        else:
            return ud_parse, preprocessed, ud_drss[0][1]
    else:
        return ud_parse, preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo program to show parsing UD trees to DRS structures.")
    parser.add_argument("-i","--infile",action="store",required=False,dest="infile",default=None,help="either the path to Parallel Meaning Bank (in which case part and document number must also be given) or to a conllu file to take as input. Either this argument or --raw is required.")
    parser.add_argument("-p","--partnum",action="store",required=False,dest="partnum",type=int,default=None,help="the part number in Parallel Meaning Bank of the data point to process.")
    parser.add_argument("-d","--docnum",action="store",required=False,dest="docnum",type=int,default=None,help="the document number in Parallel Meaning Bank of the data point to process.")
    parser.add_argument("-r","--raw",action="store",required=False,dest="raw",default=None,help="an explicit raw sentence to parse to UD and then to DRS. Either this argument or --infile is required.")
    parser.add_argument("-s","--simplified",action="store_true",dest="simplified",help="if using Parallel Meaning Bank input, this option will simplify both PMB and output before printing.") # TODO
    parser.add_argument("-t","--tracefile",action="store",required=False,dest="tracefile",default=None,help="a pdf filename in which to write the trace of the first output.")
    parser.add_argument("-a","--all",action="store",required=False,dest="tracedir",default=None,help="a path in which to write the traces of all output.")
    args = parser.parse_args()
    returnall = args.tracedir is not None
    withtrace = (args.tracedir is not None) or (args.tracefile is not None)
    ans = None
    if args.raw is not None:
        ans = sentencedemo(args.raw, returnall=returnall, withtrace=withtrace)
    elif args.infile is not None:
        if args.partnum is not None and args.docnum is not None:
            ans = pmbdemo(args.infile,args.partnum,args.docnum,simplified=args.simplified,returnall=returnall,withtrace=withtrace)
        else:
            ans = conlludemo(args.infile, returnall=returnall, withtrace=withtrace)
    if ans is not None:
        print("UD Parse:\n")
        ans[0].to_tree().print_tree()
        print("\nPreprocessed UD Parse:\n")
        ans[1].to_tree().print_tree()
        if len(ans) > 2:
            print("\nFirst Computed DRS:\n")
            if withtrace:
                if returnall:
                    ans[2][0][0].pretty_print()
                    if not os.path.exists(args.tracedir):
                        os.mkdir(args.tracedir)
                    for i in range(len(ans[2])):
                        outfile = os.path.join(args.tracedir,f"{i}.pdf")
                        tree = treetodrs.tracetogvtree(ans[2][i][1])
                        tree.render(filename=None,cleanup=True,format="pdf",outfile=outfile)
                else:
                    ans[2][0].pretty_print()
                    tree = treetodrs.tracetogvtree(ans[2][1])
                    tree.render(filename=None,cleanup=True,format="pdf",outfile=args.tracefile + (".pdf" if not args.tracefile.endswith(".pdf") else ""))
            else:
                ans[2].pretty_print()
        if len(ans) > 3:
            print("\nOriginal PMB DRS for comparison:")
            ans[3].pretty_print()
    else:
        parser.print_help()