import numpy as np
import time
import multiprocessing
import itertools
import csv
import os
import os.path
import spacy
from collections import Counter
from datetime import datetime
import pandas as pd
import operator
import getpass
from nltk.corpus import sentiwordnet as swn  
from pathlib import Path
import re

from sentence_splitter import SentenceSplitter, split_text_into_sentences

import sys

# THIS IS TO IMPORT FILES FROM A DIFFERENT FOLDER
base_dir = os.getcwd()
pathResourcesFolder = os.path.join(base_dir, "Resources")
#pathResourcesFolder = "/DATA/Resources/"
sys.path.insert(0, pathResourcesFolder)
import senticnet5
import senti_bignomics
#

#
#import pysentiment as ps  # https://github.com/hanzhichao2000/pysentiment
base_dir = os.getcwd()
pathResourcesFolder = os.path.join(base_dir, "Resources")
pathResourcesFolder = os.path.join(pathResourcesFolder, "pyss")
sys.path.insert(0, pathResourcesFolder)
import pyss as ps  # https://github.com/hanzhichao2000/pysentiment
# intialize harvard and macdonald
hiv4 = ps.HIV4()
lmac = ps.LM()
#



#
#import moral resources
base_dir = os.getcwd()
input_liberty = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "liberty"), "cs_lexicon_final.csv")
df_liberty = pd.read_csv(input_liberty, sep=",", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_liberty.score.min(), df_liberty.score.max()
df_liberty["scoreRescaled"] = (df_liberty.score - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_liberty["scoreRescaled"].max())
#print(df_liberty["scoreRescaled"].min())
#
input_authority = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "moralstrength_annotations"), "authority.tsv")
df_authority = pd.read_csv(input_authority, sep="\t", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_authority.EXPRESSED_MORAL.min(), df_authority.EXPRESSED_MORAL.max()
df_authority["scoreRescaled"] = (df_authority.EXPRESSED_MORAL - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_authority["scoreRescaled"].max())
#print(df_authority["scoreRescaled"].min())
#
input_care = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "moralstrength_annotations"), "care.tsv")
df_care = pd.read_csv(input_care, sep="\t", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_care.EXPRESSED_MORAL.min(), df_care.EXPRESSED_MORAL.max()
df_care["scoreRescaled"] = (df_care.EXPRESSED_MORAL - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_care["scoreRescaled"].max())
#print(df_care["scoreRescaled"].min())
#
input_fairness = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "moralstrength_annotations"), "fairness.tsv")
df_fairness = pd.read_csv(input_fairness, sep="\t", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_fairness.EXPRESSED_MORAL.min(), df_fairness.EXPRESSED_MORAL.max()
df_fairness["scoreRescaled"] = (df_fairness.EXPRESSED_MORAL - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_fairness["scoreRescaled"].max())
#print(df_fairness["scoreRescaled"].min())
#
input_loyalty = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "moralstrength_annotations"), "loyalty.tsv")
df_loyalty = pd.read_csv(input_loyalty, sep="\t", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_loyalty.EXPRESSED_MORAL.min(), df_loyalty.EXPRESSED_MORAL.max()
df_loyalty["scoreRescaled"] = (df_loyalty.EXPRESSED_MORAL - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_loyalty["scoreRescaled"].max())
#print(df_loyalty["scoreRescaled"].min())
#
input_purity = os.path.join(os.path.join(os.path.join(base_dir, "Resources"), "moralstrength_annotations"), "purity.tsv")
df_purity = pd.read_csv(input_purity, sep="\t", header=0, encoding='utf-8')
a_scale, b_scale = -1, 1
x_scale, y_scale = df_purity.EXPRESSED_MORAL.min(), df_purity.EXPRESSED_MORAL.max()
df_purity["scoreRescaled"] = (df_purity.EXPRESSED_MORAL - x_scale) / (y_scale - x_scale) * (b_scale - a_scale) + a_scale
#print(df_purity["scoreRescaled"].max())
#print(df_purity["scoreRescaled"].min())
#




#



def putSpace(input):

    #e.g. input = "Text1. Text2.Text3."

    rx = r"\.(?=\S)"
    result = re.sub(rx, ". ", input)
    result = re.sub(rx, ": ", input)
    #print(result)

    togiveout = result.replace("  "," ")
    togiveout = togiveout.replace("  ", " ")

    return togiveout


def split_into_sentences(text, lang):

    sentences = split_text_into_sentences(text, lang.lower())

    return sentences


def safe_string_cast_to_numerictype(val, to_type, default = None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def FeelIt(tlemma, tpos=None, tokentlemma=None, PrintScr=False, UseSenticNet=False,UseSentiWordNet=False,UseFigas=False,UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):

    computed_polarity = 0.0

    if ((UseLiberty == True) or (UseAuthority == True) or (UseCare == True) or (UseFairness == True) or (UseLoyalty == True) or (UsePurity == True)) :
        # if DEBUG:
        #    print("moral polarity computation with FeelIt")

        liberty_polarity = 0.0
        if (UseLiberty == True):
            if df_liberty is not None:
                if not df_liberty.empty:
                    if tokentlemma and (df_liberty.loc[df_liberty['word'] == tokentlemma, 'scoreRescaled'].empty == False):
                        liberty_polarity = df_liberty.loc[df_liberty['word'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_liberty.loc[df_liberty['word'] == tlemma, 'scoreRescaled'].empty == False:
                        liberty_polarity = df_liberty.loc[df_liberty['word'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        liberty_polarity = 0.0


        authority_polarity = 0.0
        if (UseAuthority == True):
            if df_authority is not None:
                if not df_authority.empty:
                    if tokentlemma and (df_authority.loc[df_authority['LEMMA'] == tokentlemma, 'scoreRescaled'].empty == False):
                        authority_polarity = df_authority.loc[df_authority['LEMMA'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_authority.loc[df_authority['LEMMA'] == tlemma, 'scoreRescaled'].empty == False:
                        authority_polarity = df_authority.loc[df_authority['LEMMA'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        authority_polarity = 0.0


        care_polarity = 0.0
        if (UseCare == True):
            if df_care is not None:
                if not df_care.empty:
                    if tokentlemma and (df_care.loc[df_care['LEMMA'] == tokentlemma, 'scoreRescaled'].empty == False):
                        care_polarity = df_care.loc[df_care['LEMMA'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_care.loc[df_care['LEMMA'] == tlemma, 'scoreRescaled'].empty == False:
                        care_polarity = df_care.loc[df_care['LEMMA'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        care_polarity = 0.0


        fairness_polarity = 0.0
        if (UseFairness == True):
            if df_fairness is not None:
                if not df_fairness.empty:
                    if tokentlemma and (
                            df_fairness.loc[df_fairness['LEMMA'] == tokentlemma, 'scoreRescaled'].empty == False):
                        fairness_polarity = df_fairness.loc[df_fairness['LEMMA'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_fairness.loc[df_fairness['LEMMA'] == tlemma, 'scoreRescaled'].empty == False:
                        fairness_polarity = df_fairness.loc[df_fairness['LEMMA'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        fairness_polarity = 0.0


        loyalty_polarity = 0.0
        if (UseLoyalty == True):
            if df_loyalty is not None:
                if not df_loyalty.empty:
                    if tokentlemma and (df_loyalty.loc[df_loyalty['LEMMA'] == tokentlemma, 'scoreRescaled'].empty == False):
                        loyalty_polarity = df_loyalty.loc[df_loyalty['LEMMA'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_loyalty.loc[df_loyalty['LEMMA'] == tlemma, 'scoreRescaled'].empty == False:
                        loyalty_polarity = df_loyalty.loc[df_loyalty['LEMMA'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        loyalty_polarity = 0.0


        purity_polarity = 0.0
        if (UsePurity == True):
            if df_purity is not None:
                if not df_purity.empty:
                    if tokentlemma and (df_purity.loc[df_purity['LEMMA'] == tokentlemma, 'scoreRescaled'].empty == False):
                        purity_polarity = df_purity.loc[df_purity['LEMMA'] == tokentlemma, 'scoreRescaled'].iloc[0]
                    elif df_purity.loc[df_purity['LEMMA'] == tlemma, 'scoreRescaled'].empty == False:
                        purity_polarity = df_purity.loc[df_purity['LEMMA'] == tlemma, 'scoreRescaled'].iloc[0]
                    else:
                        purity_polarity = 0.0

        ###########

        x = [liberty_polarity, authority_polarity, care_polarity, fairness_polarity, loyalty_polarity, purity_polarity]
        denom_for_average = sum(j != 0 for j in x)
        num_for_average = sum(x)

        computed_polarity = 0
        if denom_for_average != 0:
            computed_polarity = num_for_average / denom_for_average

        ### print("end moral polarity computation")

    elif ((UseSenticNet == True) or (UseSentiWordNet == True) or (UseFigas == True)  ):
        # if DEBUG:
        #    print("sentiment polarity computation with FeelIt")

        valstr = ""
        senti_bignomics_sentiment = 0.0
        if UseFigas==True:
            tosearcsenticnet = tlemma.lower().replace(" ", "_")
            sentibignomicsitem = senti_bignomics.senti_bignomics.get(tosearcsenticnet)
            if sentibignomicsitem and sentibignomicsitem[0]:
                valstr = sentibignomicsitem[0]
                senti_bignomics_sentiment = safe_string_cast_to_numerictype(valstr, float, 0)
                #computed_polarity = senti_bignomics_sentiment
                #return computed_polarity

        # n - NOUN
        # v - VERB
        # a - ADJECTIVE
        # s - ADJECTIVE SATELLITE
        # r - ADVERB

        if tpos == "NOUN":
            posval = "n"
        elif tpos == "VERB":
            posval = "v"
        elif tpos == "ADJ":
            posval = "a"
        elif tpos == "ADV":
            posval = "r"
        else:
            posval = "n"

        #(sentibignomicsitem == False or sentibignomicsitem[0]==False)
        #(senti_bignomics_sentiment == 0)

        avgscSWN = 0
        if (UseSentiWordNet==True) or ((UseSentiWordNet==False) and ((UseFigas==True) and ((not valstr) or valstr == ""))):
            try:
                llsenses_pos = list(swn.senti_synsets(tlemma.lower(), posval))
            except:
                llsenses_pos = []
            if llsenses_pos and len(llsenses_pos) > 0:
                for thistimescore in llsenses_pos:
                    avgscSWN = thistimescore.pos_score() - thistimescore.neg_score()
                    if avgscSWN != 0:
                        break
            if avgscSWN == 0 and posval == "a":
                posval = "s"
                try:
                    llsenses_pos = list(swn.senti_synsets(tlemma.lower(), posval))
                except:
                    llsenses_pos = []
                if llsenses_pos and len(llsenses_pos) > 0:
                    for thistimescore in llsenses_pos:
                        avgscSWN = thistimescore.pos_score() - thistimescore.neg_score()
                        if avgscSWN != 0:
                            break

        sentic_sentiment = 0
        if UseSenticNet == True:
            tosearcsenticnet = tlemma.lower().replace(" ", "_")
            senticitem = senticnet5.senticnet.get(tosearcsenticnet)
            if senticitem and senticitem[7]:
                valstr = senticitem[7]
                sentic_sentiment = safe_string_cast_to_numerictype(valstr, float, 0)


        #if sentic_sentiment != 0:
        #    computed_polarity = (sentic_sentiment + avgscSWN) / 2
        #else:
        #    computed_polarity = avgscSWN


        ###########

        x = [sentic_sentiment, avgscSWN, senti_bignomics_sentiment]
        denom_for_average = sum(j != 0 for j in x)
        num_for_average = sum(x)

        # denom_for_average = 0
        # num_for_average = 0.0
        # if sentic_sentiment != 0:
        #     denom_for_average=denom_for_average+1
        #     num_for_average = sentic_sentiment
        # if avgscSWN != 0:
        #     denom_for_average = denom_for_average + 1
        #     num_for_average = avgscSWN
        # if senti_bignomics_sentiment !=0:
        #     denom_for_average = denom_for_average + 1
        #     num_for_average = senti_bignomics_sentiment

        computed_polarity = 0
        if denom_for_average != 0:
            computed_polarity = num_for_average / denom_for_average

        ### print("end sentiment polarity computation")

    else:
        print("\n!!!ERROR!!! THIS IS NOT POSSIBLE, AT LEAST ONE OPTION SHOULD BE ENABLED, THERE IS SOME MISTAKE...CHECK IT...EXITING NOW")
        sys.exit()

    return computed_polarity


def FeelIt_OverallSentiment(toi, PrintScr=False, UseSenticNet=False, UseSentiWordNet=False, UseFigas=True, ComputeWithMacDonaldOnly=False, ComputeWithHarvardOnly=False, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):

    sentim_all = 0.0

    if (ComputeWithMacDonaldOnly==True) or (ComputeWithHarvardOnly == True):

        sentim_LMD = 0.0
        if ComputeWithMacDonaldOnly == True:
            # MACDONALD OVERALL SENTIMENT CALCULATION:

            if PrintScr == True:
                print("MACDONALD OVERALL  SENTIMENT CALCULATION on FeelIt_Overall")

            countsss = 0
            wc_mac=0
            for xin in toi.sent:
                if (xin.pos_ == "ADJ") | (xin.pos_ == "ADV") | (xin.pos_ == "NOUN") | (xin.pos_ == "PROPN") | (
                        xin.pos_ == "VERB"):

                    countsss = countsss + 1

                    tlemma=xin.lemma_.lower()
                    # compute with mcdonald
                    # listtlemma = lmac.tokenize(tlemma)  # text can be tokenized by other ways
                    listtlemma = [tlemma]
                    lmacscore = lmac.get_score(listtlemma)['Polarity']
                    if lmacscore and lmacscore!=0:
                        if lmacscore>0:
                            wc_mac=wc_mac+1
                        else:
                            wc_mac=wc_mac-1

            if wc_mac!=0 and countsss>0:
                sentim_LMD = wc_mac/countsss

        # END  MACDONALD OVERALL SENTIMENT CALCULATION:

        sentim_HARVARD = 0.0
        if ComputeWithHarvardOnly == True:

            # HARVARD OVERALL SENTIMENT CALCULATION:

            if PrintScr == True:
                print("HARVARD OVERALL  SENTIMENT CALCULATION on FeelIt_Overall")

            countsss = 0
            wc_hv = 0
            for xin in toi.sent:
                if (xin.pos_ == "ADJ") | (xin.pos_ == "ADV") | (xin.pos_ == "NOUN") | (xin.pos_ == "PROPN") | (
                        xin.pos_ == "VERB"):

                    countsss = countsss + 1

                    tlemma = xin.lemma_.lower()
                    # compute with mcdonald
                    # listtlemma = lmac.tokenize(tlemma)  # text can be tokenized by other ways
                    listtlemma = [tlemma]
                    hvscore = hiv4.get_score(listtlemma)['Polarity']
                    if hvscore and hvscore != 0:
                        if hvscore > 0:
                            wc_hv = wc_hv + 1
                        else:
                            wc_hv = wc_hv- 1

            if wc_hv != 0 and countsss>0:
                sentim_HARVARD = wc_hv / countsss

            # END HARVARD OVERALL SENTIMENT CALCULATION

        x = [sentim_LMD, sentim_HARVARD]
        denom_for_average = sum(j != 0 for j in x)
        num_for_average = sum(x)

        sentim_all = 0
        if denom_for_average != 0:
            sentim_all = num_for_average / denom_for_average


    else:

        # FIGAS OVERALL SENTIMENT CALCULATION:

        if PrintScr == True:
            print("FIGAS OVERALL POLARITY CALCULATION on FeelIt_Overall")

        countsss = 0
        for xin in toi.sent:
            if (xin.pos_ == "ADJ") | (xin.pos_ == "ADV") | (xin.pos_ == "NOUN") | (xin.pos_ == "PROPN") | (
                    xin.pos_ == "VERB"):
                sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if sentim_app != 0.0:
                    countsss = countsss + 1
                sentim_all = sentim_all + sentim_app

        if countsss > 0:
            sentim_all = sentim_all / countsss

        # END FIGAS OVERALL SENTIMENT CALCULATION

    return sentim_all




def PREP_token_IE_parsing(xin, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                          LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE, minw, maxw,
                          FoundVerb, t, nountoskip=None, previousprep=None, UseSenticNet=False, UseSentiWordNet=False,
                          UseFigas=True, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):
    listOfPreps = []
    listOfPreps_sentim = []
    lxin_n = [x for x in xin.lefts]
    rxin_n = [x for x in xin.rights]
    if lxin_n:
        for xinxin in lxin_n:
            if xinxin.dep_ == "pobj" and xinxin.pos_ == "NOUN" and IsInterestingToken(
                    xinxin) and xinxin.lemma_.lower() != t.lemma_.lower():
                if (nountoskip):
                    if xinxin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue

                print(
                    "...DEBUG NOUN ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))

                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                other_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xinxin, singleNOUNs,
                                                                               singleCompoundedHITS,
                                                                               singleCompoundedHITS_toEXCLUDE,
                                                                                   filterNOUNs,
                                                                                   filterCOMPOUNDs,
                                                                               LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                               VERBS_TO_KEEP,
                                                                               COMPUTE_OVERALL_POLARITY_SCORE,
                                                                               minw=minw, maxw=maxw,
                                                                               verbtoskip=FoundVerb, nountoskip=t, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                sentim_noun = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if other_list_NOUN and len(other_list_NOUN) > 0:
                    listNoun_app = []
                    FoundNounInlist = "___" + xinxin.lemma_.lower()
                    for modin in other_list_NOUN:
                        FoundNounInlist = FoundNounInlist + "___" + modin
                    listNoun_app.append(FoundNounInlist)
                    other_list_NOUN = listNoun_app

                    listOfPreps.extend(other_list_NOUN)
                else:
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + "___"
                    listOfPreps.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    print("   sentilist: ")
                    print(sentilist)
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # change the sign and increase it
                                else:  # they have same sign
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # increase it
                listOfPreps_sentim.append(sentim_noun)

                print(
                    "...DEBUG END NOUN ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} - sentiment={}".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_, sentim_noun
                    ))

            elif xinxin.dep_ == "pcomp" and xinxin.pos_ == "VERB" and xinxin.lemma_.lower() != t.lemma_.lower():  # and (xinxin.tag_ == "VB" or xinxin.tag_ == "VBG" or xinxin.tag_ == "VBP" or xinxin.tag_ == "VBZ" )

                print(
                    "...DEBUG VERB ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))

                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_VERB, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(xinxin, singleNOUNs,
                                                                                              singleCompoundedHITS,
                                                                                              singleCompoundedHITS_toEXCLUDE,
                                                                                                  filterNOUNs,
                                                                                                  filterCOMPOUNDs,
                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                              VERBS_TO_KEEP,
                                                                                              COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                              t, minw, maxw,
                                                                                              nountoskip=nountoskip,
                                                                                              previousverb=FoundVerb,
                                                                                              UseSenticNet=UseSenticNet,
                                                                                              UseSentiWordNet=UseSentiWordNet,
                                                                                              UseFigas=UseFigas,
                                                                                              UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                              )
                if iterated_list_VERB and len(iterated_list_VERB) > 0:
                    listOfPreps.extend(iterated_list_VERB)
                    if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                        listOfPreps_sentim.extend(list_verbs_sentim_app)
            elif xinxin.dep_ == "prep" and xinxin.pos_ == "ADP":
                if (previousprep):
                    if xinxin.lemma_.lower() == previousprep.lemma_.lower():
                        continue
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xinxin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw, maxw=maxw,
                                                                                                  FoundVerb=FoundVerb,
                                                                                                  t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  previousprep=xin,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                  )

                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listOfPreps.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listOfPreps_sentim.extend(iterated_list_prep_sentim)

            else:

                print(
                    "...DEBUG ITERATE from PREP - NOT CONSIDERED CASE - {} {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))

    if rxin_n:
        for xinxin in rxin_n:
            if xinxin.dep_ == "pobj" and xinxin.pos_ == "NOUN" and IsInterestingToken(
                    xinxin) and xinxin.lemma_.lower() != t.lemma_.lower():
                if (nountoskip):
                    if xinxin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue

                print(
                    "...DEBUG NOUN ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))

                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                other_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xinxin, singleNOUNs,
                                                                               singleCompoundedHITS,
                                                                               singleCompoundedHITS_toEXCLUDE,
                                                                                   filterNOUNs,
                                                                                   filterCOMPOUNDs,
                                                                               LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                               VERBS_TO_KEEP,
                                                                               COMPUTE_OVERALL_POLARITY_SCORE,
                                                                               minw=minw, maxw=maxw,
                                                                               verbtoskip=FoundVerb, nountoskip=t, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                sentim_noun = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if other_list_NOUN and len(other_list_NOUN) > 0:
                    listNoun_app = []
                    FoundNounInlist = "___" + xinxin.lemma_.lower()
                    for modin in other_list_NOUN:
                        FoundNounInlist = FoundNounInlist + "___" + modin
                    listNoun_app.append(FoundNounInlist)
                    other_list_NOUN = listNoun_app

                    listOfPreps.extend(other_list_NOUN)
                else:
                    FoundNounInlist = "___" + xinxin.lemma_.lower() + "___"
                    listOfPreps.append(FoundNounInlist)
                if sentilist and len(sentilist) > 0:
                    print("   sentilist: ")
                    print(sentilist)
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # change the sign and increase it
                                else:  # they have same sign
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # increase it
                listOfPreps_sentim.append(sentim_noun)

                print(
                    "...DEBUG END NOUN ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} - with sentiment={}".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_, sentim_noun
                    ))

            elif xinxin.dep_ == "pcomp" and xinxin.pos_ == "VERB" and xinxin.lemma_.lower() != t.lemma_.lower():
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_VERB, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(xinxin, singleNOUNs,
                                                                                              singleCompoundedHITS,
                                                                                              singleCompoundedHITS_toEXCLUDE,
                                                                                                  filterNOUNs,
                                                                                                  filterCOMPOUNDs,
                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                              VERBS_TO_KEEP,
                                                                                              COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                              t, minw, maxw,
                                                                                              nountoskip=nountoskip,
                                                                                              previousverb=FoundVerb,
                                                                                              UseSenticNet=UseSenticNet,
                                                                                              UseSentiWordNet=UseSentiWordNet,
                                                                                              UseFigas=UseFigas,
                                                                                              UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                              )
                if iterated_list_VERB and len(iterated_list_VERB) > 0:
                    listOfPreps.extend(iterated_list_VERB)
                    if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                        listOfPreps_sentim.extend(list_verbs_sentim_app)
            elif xinxin.dep_ == "prep" and xinxin.pos_ == "ADP":
                if (previousprep):
                    if xinxin.lemma_.lower() == previousprep.lemma_.lower():
                        continue
                print(
                    "...DEBUG VERB ITERATE from PREP {}, {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))
                minw = min(minw, xinxin.i)
                maxw = max(maxw, xinxin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xinxin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw, maxw=maxw,
                                                                                                  FoundVerb=FoundVerb,
                                                                                                  t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  previousprep=xin,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty,
                                                                                                  UseAuthority=UseAuthority,
                                                                                                  UseCare=UseCare,
                                                                                                  UseFairness=UseFairness,
                                                                                                  UseLoyalty=UseLoyalty,
                                                                                                  UsePurity=UsePurity
                                                                                                  )
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listOfPreps.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listOfPreps_sentim.extend(iterated_list_prep_sentim)

            else:

                print(
                    "...DEBUG ITERATE from PREP - NOT CONSIDERED CASE - {} {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                        xinxin.dep_,
                        xinxin.lemma_,
                        xinxin.pos_,
                        xinxin.tag_,
                        t.lemma_,
                    ))

    return listOfPreps, listOfPreps_sentim, minw, maxw



def VERB_token_IE_parsing(FoundVerb, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                          LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE, t, minw,
                          maxw, nountoskip=None, previousverb=None, UseSenticNet=True, UseSentiWordNet=True,
                          UseFigas=True, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):
    listVerbs = []
    listVerbs_sentim = []
    CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITS, FoundVerb)
    FoundNeg = None
    FoundVerbAdverb = ""
    FoundVerbAdverb_sentim = 0
    listFoundModofVB = []
    listFoundModofVB_sentim = []
    l_n = [x for x in FoundVerb.lefts]
    if l_n:
        for xin in l_n:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if xin.dep_ == "neg":
                FoundNeg = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

            # elif xin.dep_== "advmod" and (xin.pos_=="ADV" and(xin.tag_=="RBS" or xin.tag_=="RBR")) and (FoundVerb.lemma_.lower()=="be" or FoundVerb.lemma_.lower()=="have"):  #only for have or be
            #     listFoundModofVB.append(xin.lemma_.lower())
            #     print("...DEBUG VB-advmod_be_have {}: pos:{}, tag:{} ".format(xin.lemma_, xin.pos_, xin.tag_))
            #     fileIE_LOG.write('...DEBUG VB-advmod_be_have {}: pos:{} tag:{}'.format(xin.lemma_, xin.pos_, xin.tag_))
            #     fileIE_LOG.write("\n")
            elif xin.dep_ == "advmod" and (xin.pos_ == "ADV" and (
                    xin.tag_ == "RBS" or xin.tag_ == "RBR")):  # or xin.tag_ == "RB"  #and (FoundVerb.lemma_.lower() == "be" or FoundVerb.lemma_.lower() == "have"):  # only for have or be

                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                    UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                FoundVerbAdverb = FoundVerbAdverb + "__" + xin.lemma_.lower()
                if FoundVerbAdverb_sentim == 0:
                    FoundVerbAdverb_sentim = sentim_app
                else:
                    if (FoundVerbAdverb_sentim > 0 and sentim_app < 0) or (
                            FoundVerbAdverb_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                        FoundVerbAdverb_sentim = FoundVerbAdverb_sentim + sentim_app
                    else:  # they have same sign
                        FoundVerbAdverb_sentim = np.sign(FoundVerbAdverb_sentim) * (
                                abs(FoundVerbAdverb_sentim) + (1 - abs(FoundVerbAdverb_sentim)) * abs(sentim_app))

                print("...DEBUG found-advmod_of_VB {}: pos:{}, tag:{} ({})".format(xin.lemma_, xin.pos_, xin.tag_,
                                                                                   sentim_app))

            elif (xin.dep_ == "acomp" or xin.dep_ == "oprd") and (xin.pos_ == "ADJ" and (xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) and xin.lemma_.lower() != t.lemma_.lower():  # and (xin.left_edge.lemma_.lower() != t.lemma_.lower()) and (xin.right_edge.lemma_.lower() != t.lemma_.lower()):

                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue

                foundadv = ""
                foundadv_sentim = 0
                if lxin_n:
                    for xinxin in lxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  # or xinxin.tag_=="RB"
                            foundadv = foundadv + "__" + xinxin.lemma_.lower()

                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)

                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                                UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  # they have same sign
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))

                            print("...DEBUG found-advmod-adj_of_ADJ {}: pos:{}, tag:{} ({})".format(xinxin.lemma_,
                                                                                                    xinxin.pos_,
                                                                                                    xinxin.tag_,
                                                                                                    sentim_app))

                if rxin_n:
                    for xinxin in rxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  # or xinxin.tag_=="RB"
                            foundadv = foundadv + "__" + xinxin.lemma_.lower()

                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)

                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                                UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  # they have same sign
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))

                            print("...DEBUG found-advmod-adj_of_ADJ {}: pos:{}, tag:{} ({})".format(xinxin.lemma_,
                                                                                                    xinxin.pos_,
                                                                                                    xinxin.tag_,
                                                                                                    sentim_app))

                sentim_compound = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                                UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                if sentim_compound == 0:
                    sentim_compound = foundadv_sentim
                else:
                    if (foundadv_sentim > 0 and sentim_app < 0) or (
                            foundadv_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                        sentim_compound = np.sign(foundadv_sentim) * np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(
                            foundadv_sentim))  # change the sign and increase it

                    else:  # they have same sign
                        sentim_compound = np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(foundadv_sentim))  # increase it

                listFoundModofVB.append((xin.lemma_.lower() + foundadv))
                listFoundModofVB_sentim.append(sentim_compound)

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                print("...DEBUG VB-acomp {}: pos:{}, tag:{} ({})".format(xin.lemma_ + foundadv, xin.pos_, xin.tag_,
                                                                         sentim_compound))


            elif (xin.dep_ == "dobj" or xin.dep_ == "attr") and xin.pos_ == "NOUN" and IsInterestingToken(
                    xin) and xin.lemma_.lower() != t.lemma_.lower():

                if (nountoskip):
                    # print("\n\n\nSTART nountoskip PROBLEM")
                    # fileIE_LOG.write("\n\n\n\nSTART nountoskip PROBLEM")
                    # fileIE_LOG.write("\n")
                    # print(t.sent)
                    # fileIE_LOG.write("\n" + t.sent.text)
                    # fileIE_LOG.write("\n\n")
                    # print("t")
                    # print(t.lemma_)
                    # fileIE_LOG.write("t")
                    # fileIE_LOG.write(t.lemma_)
                    # fileIE_LOG.write("\n")
                    # print("nountoskip")
                    # print(nountoskip.lemma_)
                    # fileIE_LOG.write("nountoskip")
                    # fileIE_LOG.write(nountoskip.lemma_)
                    # fileIE_LOG.write("\n\n")
                    if xin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue

                print("...DEBUG NOUN ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} ".format(xin.dep_,
                                                                                                       xin.lemma_,
                                                                                                       xin.pos_,
                                                                                                       xin.tag_,
                                                                                                       t.lemma_,
                                                                                                       ))
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xin, singleNOUNs, singleCompoundedHITS,
                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                      filterNOUNs,
                                                                                      filterCOMPOUNDs,
                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                  VERBS_TO_KEEP,
                                                                                  COMPUTE_OVERALL_POLARITY_SCORE, minw=minw,
                                                                                  maxw=maxw,
                                                                                  verbtoskip=FoundVerb, nountoskip=t, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                sentim_noun = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if sentilist and len(sentilist) > 0:
                    print("   sentilist: ")
                    print(sentilist)
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # change the sign and increase it
                                else:  # they have same sign
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # increase it
                listFoundModofVB_sentim.append(sentim_noun)

                print(
                    "...DEBUG END NOUN ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} - with sentiment={}".format(
                        xin.dep_,
                        xin.lemma_,
                        xin.pos_,
                        xin.tag_,
                        t.lemma_, sentim_noun
                    ))

                if iterated_list_NOUN and len(iterated_list_NOUN) > 0:
                    #
                    listNoun_app = []
                    for modin in iterated_list_NOUN:
                        FoundNounInlist = "___" + xin.lemma_.lower() + "___" + modin
                        listNoun_app.append(FoundNounInlist)
                    iterated_list_NOUN = listNoun_app
                    #
                    listFoundModofVB.extend(iterated_list_NOUN)
                else:
                    FoundNounInlist = "___" + xin.lemma_.lower() + "___"
                    listFoundModofVB.append(FoundNounInlist)

            #
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower():

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw, maxw=maxw,
                                                                                                  FoundVerb=FoundVerb, t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                  )
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listFoundModofVB.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listFoundModofVB_sentim.extend(iterated_list_prep_sentim)

            if FoundNeg is None:
                for xinxin in lxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
                for xinxin in rxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)


    l_r = [x for x in FoundVerb.rights]
    if l_r:
        for xin in l_r:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if xin.dep_ == "neg":
                FoundNeg = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

            # elif xin.dep_ == "advmod" and (xin.pos_ == "ADV" and (xin.tag_ == "RBS" or xin.tag_ == "RBR")) and (FoundVerb.lemma_.lower() == "be" or FoundVerb.lemma_.lower() == "have"):  # only for have or be
            #     listFoundModofVB.append(xin.lemma_.lower())
            #     print("...DEBUG VB-advmod_be_have {}: pos:{}, tag:{} ".format(xin.lemma_, xin.pos_,xin.tag_))
            #     fileIE_LOG.write('...DEBUG VB-advmod_be_have {}: pos:{} tag:{}'.format(xin.lemma_, xin.pos_,xin.tag_))
            #     fileIE_LOG.write("\n")
            elif xin.dep_ == "advmod" and (xin.pos_ == "ADV" and (
                    xin.tag_ == "RBS" or xin.tag_ == "RBR")):  ## or xin.tag_ == "RB"  and (FoundVerb.lemma_.lower() == "be" or FoundVerb.lemma_.lower() == "have"):  # only for have or be

                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                FoundVerbAdverb = FoundVerbAdverb + "__" + xin.lemma_.lower()
                sentim_app = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                    UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if FoundVerbAdverb_sentim == 0:
                    FoundVerbAdverb_sentim = sentim_app
                else:
                    if (FoundVerbAdverb_sentim > 0 and sentim_app < 0) or (
                            FoundVerbAdverb_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                        FoundVerbAdverb_sentim = FoundVerbAdverb_sentim + sentim_app
                    else:  # they have same sign
                        FoundVerbAdverb_sentim = np.sign(FoundVerbAdverb_sentim) * (
                                abs(FoundVerbAdverb_sentim) + (1 - abs(FoundVerbAdverb_sentim)) * abs(sentim_app))

                print("...DEBUG found-advmod_of_VB {}: pos:{}, tag:{} ({})".format(xin.lemma_, xin.pos_, xin.tag_,
                                                                                   sentim_app))


            elif (xin.dep_ == "acomp" or xin.dep_ == "oprd") and (xin.pos_ == "ADJ" and (
                    xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) and xin.lemma_.lower() != t.lemma_.lower():  # and (xin.left_edge.lemma_.lower() != t.lemma_.lower()) and (xin.right_edge.lemma_.lower() != t.lemma_.lower()):

                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                foundadv = ""
                foundadv_sentim = 0
                if lxin_n:
                    for xinxin in lxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  # or xinxin.tag_=="RB"
                            foundadv = foundadv + "__" + xinxin.lemma_.lower()

                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)

                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                                UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  # they have same sign
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))

                            print("...DEBUG found-advmod-adj_of_ADJ {}: pos:{}, tag:{} ({})".format(xinxin.lemma_,
                                                                                                    xinxin.pos_,
                                                                                                    xinxin.tag_,
                                                                                                    sentim_app))

                if rxin_n:
                    for xinxin in rxin_n:
                        if ((xinxin.dep_ == "advmod" and (xinxin.pos_ == "ADV" and (
                                xinxin.tag_ == "RBS" or xinxin.tag_ == "RBR"))) or (
                                    xinxin.dep_ == "conj" and (xinxin.pos_ == "ADJ" and (
                                    xinxin.tag_ == "JJR" or xinxin.tag_ == "JJS" or xinxin.tag_ == "JJ")))) and xinxin.lemma_.lower() != t.lemma_.lower():  # or xinxin.tag_=="RB"
                            foundadv = foundadv + "__" + xinxin.lemma_.lower()

                            minw = min(minw, xinxin.i)
                            maxw = max(maxw, xinxin.i)

                            sentim_app = FeelIt(xinxin.lemma_.lower(), xinxin.pos_, xinxin, UseSenticNet=UseSenticNet,
                                                UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                            if foundadv_sentim == 0:
                                foundadv_sentim = sentim_app
                            else:
                                if (foundadv_sentim > 0 and sentim_app < 0) or (
                                        foundadv_sentim < 0 and sentim_app > 0):  # they have different signes, I just sum them up
                                    foundadv_sentim = foundadv_sentim + sentim_app
                                else:  # they have same sign
                                    foundadv_sentim = np.sign(foundadv_sentim) * (
                                            abs(foundadv_sentim) + (1 - abs(foundadv_sentim)) * abs(sentim_app))

                            print("...DEBUG found-advmod-adj_of_ADJ {}: pos:{}, tag:{} ({})".format(xinxin.lemma_,
                                                                                                    xinxin.pos_,
                                                                                                    xinxin.tag_,
                                                                                                    sentim_app))

                sentim_compound = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                         UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if sentim_compound == 0:
                    sentim_compound = foundadv_sentim
                else:
                    if (foundadv_sentim > 0 and sentim_compound < 0) or (
                            foundadv_sentim < 0 and sentim_compound > 0):  # they have different signes
                        sentim_compound = np.sign(foundadv_sentim) * np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(
                            foundadv_sentim))  # change the sign and increase it

                    else:  # they have same sign
                        sentim_compound = np.sign(sentim_compound) * (
                                abs(sentim_compound) + (1 - abs(sentim_compound)) * abs(foundadv_sentim))  # increase it

                listFoundModofVB.append((xin.lemma_.lower() + foundadv))
                listFoundModofVB_sentim.append(sentim_compound)

                print("...DEBUG VB-acomp {}: pos:{}, tag:{} ({})".format(xin.lemma_ + foundadv, xin.pos_, xin.tag_,
                                                                         sentim_compound))



            elif (
                    xin.dep_ == "dobj" or xin.dep_ == "attr") and xin.pos_ == "NOUN" and IsInterestingToken(
                xin) and xin.lemma_.lower() != t.lemma_.lower():

                if (nountoskip):
                    # print("\n\n\nSTART nountoskip PROBLEM")
                    # fileIE_LOG.write("\n\n\n\nSTART nountoskip PROBLEM")
                    # fileIE_LOG.write("\n")
                    # print(t.sent)
                    # fileIE_LOG.write("\n" + t.sent.text)
                    # fileIE_LOG.write("\n\n")
                    # print("t")
                    # print(t.lemma_)
                    # fileIE_LOG.write("t")
                    # fileIE_LOG.write(t.lemma_)
                    # fileIE_LOG.write("\n")
                    # print("nountoskip")
                    # print(nountoskip.lemma_)
                    # fileIE_LOG.write("nountoskip")
                    # fileIE_LOG.write(nountoskip.lemma_)
                    # fileIE_LOG.write("\n\n")
                    if xin.lemma_.lower() == nountoskip.lemma_.lower():
                        continue

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                print("...DEBUG NOUN ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} ".format(xin.dep_,
                                                                                                       xin.lemma_,
                                                                                                       xin.pos_,
                                                                                                       xin.tag_,
                                                                                                       t.lemma_,
                                                                                                       ))

                iterated_list_NOUN, sentilist, minw, maxw = NOUN_token_IE_parsing(xin, singleNOUNs, singleCompoundedHITS,
                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                      filterNOUNs,
                                                                                      filterCOMPOUNDs,
                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                  VERBS_TO_KEEP,
                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                  minw=minw, maxw=maxw,
                                                                                  verbtoskip=FoundVerb, nountoskip=t, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                sentim_noun = FeelIt(xin.lemma_.lower(), xin.pos_, xin, UseSenticNet=UseSenticNet,
                                     UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                if sentilist and len(sentilist) > 0:
                    print("   sentilist: ")
                    print(sentilist)
                    for sentin in sentilist:
                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # change the sign and increase it
                                else:  # they have same sign
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # increase it
                listFoundModofVB_sentim.append(sentim_noun)

                print(
                    "...DEBUG END NOUN ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} - with sentiment={}".format(
                        xin.dep_,
                        xin.lemma_,
                        xin.pos_,
                        xin.tag_,
                        t.lemma_, sentim_noun
                    ))

                if iterated_list_NOUN and len(iterated_list_NOUN) > 0:
                    #
                    listNoun_app = []
                    for modin in iterated_list_NOUN:
                        FoundNounInlist = "___" + xin.lemma_.lower() + "___" + modin
                        listNoun_app.append(FoundNounInlist)
                    iterated_list_NOUN = listNoun_app
                    #
                    listFoundModofVB.extend(iterated_list_NOUN)

                else:
                    FoundNounInlist = "___" + xin.lemma_.lower() + "___"
                    listFoundModofVB.append(FoundNounInlist)



            #
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower():

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw,
                                                                                                  maxw=maxw,
                                                                                                  FoundVerb=FoundVerb, t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                  )
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listFoundModofVB.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listFoundModofVB_sentim.extend(iterated_list_prep_sentim)

            if FoundNeg is None:
                for xinxin in lxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
                for xinxin in rxin_n:
                    if (xinxin.dep_ == "neg"):
                        FoundNeg = "__not"
                        minw = min(minw, xinxin.i)
                        maxw = max(maxw, xinxin.i)
    sentim_vb = FeelIt(FoundVerb.lemma_.lower(), FoundVerb.pos_, FoundVerb, UseSenticNet=UseSenticNet,
                       UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
    if not listFoundModofVB or len(listFoundModofVB) <= 0:

        if (FoundVerb.lemma_.lower() != "be" and FoundVerb.lemma_.lower() != "have"):
            FoundVerb_name = FoundVerb.lemma_.lower() + FoundVerbAdverb

            listVerbs.append(FoundVerb_name)

            if FoundVerbAdverb_sentim != 0:
                if sentim_vb == 0:
                    sentim_vb = FoundVerbAdverb_sentim
                else:
                    if (sentim_vb > 0 and FoundVerbAdverb_sentim < 0) or (
                            sentim_vb < 0 and FoundVerbAdverb_sentim > 0):  # they have different signes, I just sum them up
                        sentim_vb = np.sign(FoundVerbAdverb_sentim) * np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            FoundVerbAdverb_sentim))  # change the sign and increase it
                    else:  # they have same sign
                        sentim_vb = np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            FoundVerbAdverb_sentim))  # increase it

            listVerbs_sentim = [sentim_vb]

            print("...DEBUG VB-only {}: pos:{}, tag:{} ({})".format(FoundVerb_name, FoundVerb.pos_, FoundVerb.tag_,
                                                                    sentim_vb))

        else:
            print("...DEBUG VB-BE-HAVE DISCARDED {}: pos:{}, tag:{} ".format(FoundVerb.lemma_, FoundVerb.pos_,
                                                                             FoundVerb.tag_))
            print(t.sent)



    else:
        # #
        # list_app2 = []
        # for modin in listFoundModofVB:
        #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
        #     list_app2.append(FoundVBInlist)
        # listFoundModofVB = list_app2
        # #
        # listVerbs = listFoundModofVB
        #
        minw = min(minw, FoundVerb.i)
        maxw = max(maxw, FoundVerb.i)
        FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___"
        for j in range(0, len(listFoundModofVB)):
            modin = listFoundModofVB[j]
            FoundVBInlist = FoundVBInlist + modin
            if j < len(listFoundModofVB) - 1 and (modin.endswith('__') == False):
                FoundVBInlist = FoundVBInlist + "__"

        for j in range(0, len(listFoundModofVB_sentim)):
            sentin = listFoundModofVB_sentim[j]
            if sentin != 0:
                if sentim_vb == 0:
                    sentim_vb = sentin
                else:
                    if (sentim_vb > 0 and sentin < 0) or (
                            sentim_vb < 0 and sentin > 0):  # they have different signes, I just sum them up
                        sentim_vb = np.sign(sentin) * np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            sentin))  # change the sign and increase it
                        # sentim_vb = np.sign(sentim_vb) * np.sign(sentin) * (
                        #         abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                        #     sentin))  # change the sign and increase it
                    else:  # they have same sign
                        sentim_vb = np.sign(sentim_vb) * (
                                abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
                            sentin))  # increase it

        listVerbs.append(FoundVBInlist)
        listVerbs_sentim.append(sentim_vb)

        # if listVerbs_sentim and len(listVerbs_sentim) > 0:
        #     for sentin in listVerbs_sentim:
        #         if sentin != 0:
        #             if sentim_vb == 0:
        #                 sentim_vb = sentin
        #             else:
        #                 if (sentim_vb > 0 and sentin < 0) or (
        #                         sentim_vb < 0 and sentin > 0):  # they have different signes, I just sum them up
        #                     sentim_vb = (-1) * np.sign(sentim_vb) * (
        #                             abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
        #                         sentin))  # change the sign and increase it
        #                 else:  # they have same sign
        #                     sentim_vb = np.sign(sentim_vb) * (
        #                             abs(sentim_vb) + (1 - abs(sentim_vb)) * abs(
        #                         sentin))  # increase it
        # listVerbs_sentim=[sentim_vb]

    listVerbs_app = []
    if FoundNeg == "__not" and len(listVerbs) > 0:
        for modin in listVerbs:
            listVerbs_app.append(modin + "__not")
        listVerbs = listVerbs_app

    return listVerbs, listVerbs_sentim, minw, maxw


#
# PERSON	People, including fictional.
# NORP	Nationalities or religious or political groups.
# FAC	Buildings, airports, highways, bridges, etc.
# ORG	Companies, agencies, institutions, etc.
# GPE	Countries, cities, states.
# LOC	Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT	Objects, vehicles, foods, etc. (Not services.)
# EVENT	Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART	Titles of books, songs, etc.
# LAW	Named documents made into laws.
# LANGUAGE	Any named language.
# DATE	Absolute or relative dates or periods.
# TIME	Times smaller than a day.
# PERCENT	Percentage, including "%".
# MONEY	Monetary values, including unit.
# QUANTITY	Measurements, as of weight or distance.
# ORDINAL	"first", "second", etc.
# CARDINAL

def IsInterestingToken(t):
    ret = False
    if t.ent_type_ == "" or t.ent_type_ == 'ORG' or t.ent_type_ == 'GPE' or t.ent_type_ == 'PRODUCT' or t.ent_type_ == 'EVENT' or t.ent_type_ == 'LAW' or t.ent_type_ == 'MONEY' or t.ent_type_ == 'QUANTITY' or t.ent_type_ == 'LOC' or t.ent_type_ == 'NORP':  
        ret = True
    return ret



def NOUN_token_IE_parsing(t, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                                          LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP,
                                          COMPUTE_OVERALL_POLARITY_SCORE, minw, maxw, verbtoskip=None, nountoskip=None,
                                          UseSenticNet=False, UseSentiWordNet=False, UseFigas=True, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):
    to_give_back = []
    to_give_back_sentiment = []

    ll = [x for x in t.lefts]
    rr = [x for x in t.rights]

    CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITS, t.lemma_.lower())

    listVerbs = []
    listVerbs_sentim = []
    FoundVerb = None
    if (t.head.pos_ == "VERB" and (
            not t.head is verbtoskip)):  # and (t.head.tag_ == "VB" or t.head.tag_ == "VBG" or t.head.tag_ == "VBP" or t.head.tag_ == "VBZ" )  or t.head.tag_ == "VBD" or t.head.tag_ == "VBN"  # and t.head.lemma_.lower()!="be" and t.head.lemma_.lower()!="have"):

        FoundVerb = t.head

        minw = min(minw, FoundVerb.i)
        maxw = max(maxw, FoundVerb.i)

        lvin_n = [x for x in FoundVerb.lefts]
        rvin_n = [x for x in FoundVerb.rights]
        if lvin_n:
            for vin in lvin_n:
                lvin_inner = [x for x in vin.lefts]
                rvin_inner = [x for x in vin.rights]
                if (
                        vin.dep_ == "xcomp" or vin.dep_ == "advcl") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower() and vin.pos_ == "VERB":  # and (vin.tag_ == "VB" or vin.tag_ == "VBG" or vin.tag_ == "VBP" or vin.tag_ == "VBZ" ):  #or vin.tag_ == "VBD" or vin.tag_ == "VBN"
                    # FoundVerb = vin
                    print("...DEBUG VB-xcomp_advcl {}: pos:{}, tag:{} ".format(vin.lemma_, vin.pos_,
                                                                               vin.tag_))

                    if (not verbtoskip) or (vin.lemma_.lower() != verbtoskip.lemma_.lower()):

                        minw = min(minw, vin.i)
                        maxw = max(maxw, vin.i)
                        list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                            vin, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                            LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE,
                            t, minw, maxw, nountoskip=nountoskip, previousverb=FoundVerb, UseSenticNet=UseSenticNet,
                            UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                        if list_verbs_app and len(list_verbs_app) > 0:
                            # #
                            # list_app2 = []
                            # for modin in list_verbs_app:
                            #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                            #     list_app2.append(FoundVBInlist)
                            # list_verbs_app = list_app2
                            # #
                            listVerbs.extend(list_verbs_app)
                            if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                listVerbs_sentim.extend(list_verbs_sentim_app)

                elif (
                        vin.dep_ == "acomp" or vin.dep_ == "oprd") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower():
                    if lvin_inner:
                        for vinvin in lvin_inner:
                            if (
                                    vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":  # and (vinvin.tag_ == "VB" or vinvin.tag_ == "VBG" or vinvin.tag_ == "VBP" or vinvin.tag_ == "VBZ" ):  #or vinvin.tag_ == "VBD" or vinvin.tag_ == "VBN"
                                # FoundVerb = vinvin
                                print("...DEBUG VB-xcomp_advcl2 {}: pos:{}, tag:{} ".format(vinvin.lemma_, vinvin.pos_,
                                                                                            vinvin.tag_))

                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)

                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                                        vinvin, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                                        LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE,
                                        t, minw, maxw, nountoskip=nountoskip, previousverb=FoundVerb,
                                        UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        # #
                                        # list_app2 = []
                                        # for modin in list_verbs_app:
                                        #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                                        #     list_app2.append(FoundVBInlist)
                                        # list_verbs_app = list_app2
                                        # #
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)

                    if rvin_inner:
                        for vinvin in rvin_inner:
                            if (
                                    vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":  # and (vinvin.tag_ == "VB" or vinvin.tag_ == "VBG" or vinvin.tag_ == "VBP" or vinvin.tag_ == "VBZ" ):   #or vinvin.tag_ == "VBD" or vinvin.tag_ == "VBN"
                                # FoundVerb = vinvin
                                print("...DEBUG VB-xcomp_advcl2 {}: pos:{}, tag:{} ".format(vinvin.lemma_, vinvin.pos_,
                                                                                            vinvin.tag_))

                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):

                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                                        vinvin, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                                        LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE,
                                        t, minw, maxw, nountoskip=nountoskip, previousverb=FoundVerb,
                                        UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        # #
                                        # list_app2 = []
                                        # for modin in list_verbs_app:
                                        #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                                        #     list_app2.append(FoundVBInlist)
                                        # list_verbs_app = list_app2
                                        # #
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
        if rvin_n:
            for vin in rvin_n:
                lvin_inner = [x for x in vin.lefts]
                rvin_inner = [x for x in vin.rights]
                if (
                        vin.dep_ == "xcomp" or vin.dep_ == "advcl") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower() and vin.pos_ == "VERB":  # and (vin.tag_ == "VB" or vin.tag_ == "VBG" or vin.tag_ == "VBP" or vin.tag_ == "VBZ" ):   #or vin.tag_ == "VBD" or vin.tag_ == "VBN"
                    # FoundVerb = vin
                    print("...DEBUG VB-xcomp_advcl {}: pos:{}, tag:{} ".format(vin.lemma_, vin.pos_,
                                                                               vin.tag_))


                    if (not verbtoskip) or (vin.lemma_.lower() != verbtoskip.lemma_.lower()):
                        minw = min(minw, vin.i)
                        maxw = max(maxw, vin.i)
                        list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(
                            vin, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                            LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE,
                            t, minw, maxw, nountoskip=nountoskip, previousverb=FoundVerb, UseSenticNet=UseSenticNet,
                            UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                        if list_verbs_app and len(list_verbs_app) > 0:
                            # #
                            # list_app2 = []
                            # for modin in list_verbs_app:
                            #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                            #     list_app2.append(FoundVBInlist)
                            # list_verbs_app = list_app2
                            # #
                            listVerbs.extend(list_verbs_app)
                            if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                listVerbs_sentim.extend(list_verbs_sentim_app)

                elif (
                        vin.dep_ == "acomp" or vin.dep_ == "oprd") and vin.lemma_.lower() != t.lemma_.lower() and vin.lemma_.lower() != FoundVerb.lemma_.lower():
                    if lvin_inner:
                        for vinvin in lvin_inner:
                            if (
                                    vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":  # and (vinvin.tag_ == "VB" or vinvin.tag_ == "VBG" or vinvin.tag_ == "VBP" or vinvin.tag_ == "VBZ" ):  #or vinvin.tag_ == "VBD" or vinvin.tag_ == "VBN"
                                # FoundVerb = vinvin
                                print("...DEBUG VB-xcomp_advcl2 {}: pos:{}, tag:{} ".format(vinvin.lemma_, vinvin.pos_,
                                                                                            vinvin.tag_))

                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(vinvin,
                                                                                                              singleNOUNs,
                                                                                                              singleCompoundedHITS,
                                                                                                              singleCompoundedHITS_toEXCLUDE,
                                                                                                                  filterNOUNs,
                                                                                                                  filterCOMPOUNDs,
                                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                              VERBS_TO_KEEP,
                                                                                                              COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                              t, minw,
                                                                                                              maxw,
                                                                                                              nountoskip=nountoskip,
                                                                                                              previousverb=FoundVerb,
                                                                                                              UseSenticNet=UseSenticNet,
                                                                                                              UseSentiWordNet=UseSentiWordNet,
                                                                                                              UseFigas=UseFigas,
                                                                                                              UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                              )
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        # #
                                        # list_app2 = []
                                        # for modin in list_verbs_app:
                                        #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                                        #     list_app2.append(FoundVBInlist)
                                        # list_verbs_app = list_app2
                                        # #
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)
                    if rvin_inner:
                        for vinvin in rvin_inner:
                            if (vinvin.dep_ == "xcomp" or vinvin.dep_ == "advcl") and vinvin.lemma_.lower() != t.lemma_.lower() and vinvin.lemma_.lower() != FoundVerb.lemma_.lower() and vinvin.pos_ == "VERB":  # and (vinvin.tag_ == "VB" or vinvin.tag_ == "VBG" or vinvin.tag_ == "VBP" or vinvin.tag_ == "VBZ" ):   #or vinvin.tag_ == "VBD" or vinvin.tag_ == "VBN"
                                # FoundVerb = vinvin
                                print("...DEBUG VB-xcomp_advcl2 {}: pos:{}, tag:{} ".format(vinvin.lemma_, vinvin.pos_,
                                                                                            vinvin.tag_))

                                if (not verbtoskip) or (vinvin.lemma_.lower() != verbtoskip.lemma_.lower()):
                                    minw = min(minw, vinvin.i)
                                    maxw = max(maxw, vinvin.i)
                                    list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(vinvin,
                                                                                                              singleNOUNs,
                                                                                                              singleCompoundedHITS,
                                                                                                              singleCompoundedHITS_toEXCLUDE,
                                                                                                                  filterNOUNs,
                                                                                                                  filterCOMPOUNDs,
                                                                                                              LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                              VERBS_TO_KEEP,
                                                                                                              COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                              t, minw,
                                                                                                              maxw,
                                                                                                              nountoskip=nountoskip,
                                                                                                              previousverb=FoundVerb,
                                                                                                              UseSenticNet=UseSenticNet,
                                                                                                              UseSentiWordNet=UseSentiWordNet,
                                                                                                              UseFigas=UseFigas,
                                                                                                              UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                              )
                                    if list_verbs_app and len(list_verbs_app) > 0:
                                        # #
                                        # list_app2 = []
                                        # for modin in list_verbs_app:
                                        #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                                        #     list_app2.append(FoundVBInlist)
                                        # list_verbs_app = list_app2
                                        # #
                                        listVerbs.extend(list_verbs_app)
                                        if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                                            listVerbs_sentim.extend(list_verbs_sentim_app)

        if (not verbtoskip) or (FoundVerb.lemma_.lower() != verbtoskip.lemma_.lower()):
            minw = min(minw, FoundVerb.i)
            maxw = max(maxw, FoundVerb.i)
            list_verbs_app, list_verbs_sentim_app, minw, maxw = VERB_token_IE_parsing(FoundVerb, singleNOUNs,
                                                                                      singleCompoundedHITS,
                                                                                      singleCompoundedHITS_toEXCLUDE,
                                                                                          filterNOUNs,
                                                                                          filterCOMPOUNDs,
                                                                                      LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                      VERBS_TO_KEEP,
                                                                                      COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                      t, minw=minw, maxw=maxw,
                                                                                      nountoskip=nountoskip,
                                                                                      UseSenticNet=UseSenticNet,
                                                                                      UseSentiWordNet=UseSentiWordNet,
                                                                                      UseFigas=UseFigas,
                                                                                      UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                      )

            if list_verbs_app and len(list_verbs_app) > 0:
                # #
                # list_app2 = []
                # for modin in list_verbs_app:
                #     FoundVBInlist = "___" + FoundVerb.lemma_.lower() + "___" + modin
                #     list_app2.append(FoundVBInlist)
                # list_verbs_app = list_app2
                # #
                listVerbs.extend(list_verbs_app)
                if list_verbs_sentim_app and len(list_verbs_sentim_app) > 0:
                    listVerbs_sentim.extend(list_verbs_sentim_app)

    # ------------------------------------------------------------------------------------------------

    listAMODs = []
    listAMODs_sentim = []
    FoundAMOD = None

    FoundNeg_left = None
    if ll:
        for xin in ll:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if (xin.dep_ == "neg"):
                FoundNeg_left = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif (xin.dep_ == "amod" and (
                    (xin.pos_ == "ADJ" and (xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) or (
                    xin.pos_ == "VERB")) and xin.lemma_.lower() != t.lemma_.lower()):  # and (xin.tag_ == "VB" or xin.tag_ == "VBG" or xin.tag_ == "VBP" or xin.tag_ == "VBZ" ) or xin.tag_ == "VBD" or xin.tag_ == "VBN"

                # if FoundCompound:
                #     if (xin.lemma_.lower() == FoundCompound.lemma_.lower()):
                #         continue
                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit) :
                    continue

                FoundAMOD = xin
                FoundAMOD_name = FoundAMOD.lemma_.lower()
                listAMODs.append(FoundAMOD_name)

                FoundAMOD_sentiment = FeelIt(FoundAMOD.lemma_.lower(), FoundAMOD.pos_, FoundAMOD, UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                listAMODs_sentim.append(FoundAMOD_sentiment)

                minw = min(minw, FoundAMOD.i)
                maxw = max(maxw, FoundAMOD.i)

                # if True:  # DEBUG:
                print("...DEBUG amod {}: pos:{}, tag:{} ({})".format(FoundAMOD.lemma_, FoundAMOD.pos_, FoundAMOD.tag_,
                                                                     FoundAMOD_sentiment))


            #
            elif xin.dep_ == "acl" and xin.lemma_.lower() != t.lemma_.lower():

                if (
                        xin.pos_ == "VERB"):  # and (xin.tag_ == "VB" or xin.tag_ == "VBG" or xin.tag_ == "VBP" or xin.tag_ == "VBZ" ) or xin.tag_ == "VBD" or xin.tag_ == "VBN"   # and t.head.lemma_.lower()!="be" and t.head.lemma_.lower()!="have"):

                    # if (previousverb):
                    #     # print("\n\n\nSTART nountoskip PROBLEM")
                    #     # fileIE_LOG.write("\n\n\n\nSTART nountoskip PROBLEM")
                    #     # fileIE_LOG.write("\n")
                    #     # print(t.sent)
                    #     # fileIE_LOG.write("\n" + t.sent.text)
                    #     # fileIE_LOG.write("\n\n")
                    #     # print("t")
                    #     # print(t.lemma_)
                    #     # fileIE_LOG.write("t")
                    #     # fileIE_LOG.write(t.lemma_)
                    #     # fileIE_LOG.write("\n")
                    #     # print("nountoskip")
                    #     # print(nountoskip.lemma_)
                    #     # fileIE_LOG.write("nountoskip")
                    #     # fileIE_LOG.write(nountoskip.lemma_)
                    #     # fileIE_LOG.write("\n\n")
                    #     if xin.lemma_.lower() == previousverb.lemma_.lower():
                    #         continue

                    print("...DEBUG ACL ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} ".format(xin.dep_,
                                                                                                          xin.lemma_,
                                                                                                          xin.pos_,
                                                                                                          xin.tag_,
                                                                                                          t.lemma_,
                                                                                                          ))
                    minw = min(minw, xin.i)
                    maxw = max(maxw, xin.i)

                    iterated_list_VERB, iterated_list_VERB_sentiment, minw, maxw = VERB_token_IE_parsing(xin,
                                                                                                         singleNOUNs,
                                                                                                         singleCompoundedHITS,
                                                                                                         singleCompoundedHITS_toEXCLUDE,
                                                                                                             filterNOUNs,
                                                                                                             filterCOMPOUNDs,
                                                                                                         LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                         VERBS_TO_KEEP,
                                                                                                         COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                         t, minw, maxw,
                                                                                                         nountoskip=nountoskip,
                                                                                                         previousverb=FoundVerb,
                                                                                                         UseSenticNet=UseSenticNet,
                                                                                                         UseSentiWordNet=UseSentiWordNet,
                                                                                                         UseFigas=UseFigas,
                                                                                                         UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                         )

                    if iterated_list_VERB and len(iterated_list_VERB) > 0:
                        # #
                        # list_app2 = []
                        # for modin in iterated_list_VERB:
                        #     FoundVBInlist = "___" + xin.lemma_.lower() + "___" + modin
                        #     list_app2.append(FoundVBInlist)
                        # iterated_list_VERB = list_app2
                        # #
                        listAMODs.extend(iterated_list_VERB)
                        if iterated_list_VERB_sentiment and len(iterated_list_VERB_sentiment) > 0:
                            listAMODs_sentim.extend(iterated_list_VERB_sentiment)


                else:

                    print(
                        "...DEBUG ACL ITERATE - NOT CONSIDERED CASE - {} {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                            xin.dep_,
                            xin.lemma_,
                            xin.pos_,
                            xin.tag_,
                            t.lemma_,
                        ))
                    print(t.sent)
                    # print("\n")


            # #
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower() and (
                    nountoskip is None):  # nountoskip is None because I make it only for the first noun

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw, maxw=maxw,
                                                                                                  FoundVerb=FoundVerb,
                                                                                                  t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                  )

                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listAMODs.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listAMODs_sentim.extend(iterated_list_prep_sentim)

    FoundNeg_right = None
    if rr:
        for xin in rr:
            lxin_n = [x for x in xin.lefts]
            rxin_n = [x for x in xin.rights]
            if (xin.dep_ == "neg"):
                FoundNeg_right = "__not"
                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)
            elif (xin.dep_ == "amod" and (
                    (xin.pos_ == "ADJ" and (xin.tag_ == "JJR" or xin.tag_ == "JJS" or xin.tag_ == "JJ")) or (
                    xin.pos_ == "VERB")) and xin.lemma_.lower() != t.lemma_.lower()):  # and (xin.tag_ == "VB" or xin.tag_ == "VBG" or xin.tag_ == "VBP" or xin.tag_ == "VBZ" ) or xin.tag_ == "VBD" or xin.tag_ == "VBN"

                # if FoundCompound:
                #     if (xin.lemma_.lower() == FoundCompound.lemma_.lower()):
                #         continue
                if (xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit):
                    continue

                FoundAMOD = xin
                FoundAMOD_name = FoundAMOD.lemma_.lower()
                listAMODs.append(FoundAMOD_name)

                FoundAMOD_sentiment = FeelIt(FoundAMOD.lemma_.lower(), FoundAMOD.pos_, FoundAMOD,
                                             UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet,
                                             UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                listAMODs_sentim.append(FoundAMOD_sentiment)

                minw = min(minw, FoundAMOD.i)
                maxw = max(maxw, FoundAMOD.i)

                # if True:  # DEBUG:
                print("...DEBUG amod {}: pos:{}, tag:{} ({})".format(FoundAMOD.lemma_, FoundAMOD.pos_, FoundAMOD.tag_,
                                                                     FoundAMOD_sentiment))

            #
            elif xin.dep_ == "acl" and xin.lemma_.lower() != t.lemma_.lower():

                if (
                        xin.pos_ == "VERB"):  # or and (xin.tag_ == "VB" or xin.tag_ == "VBG" or xin.tag_ == "VBP" or xin.tag_ == "VBZ" ) xin.tag_ == "VBD" or xin.tag_ == "VBN"   # and t.head.lemma_.lower()!="be" and t.head.lemma_.lower()!="have"):

                    # if (previousverb):
                    #     # print("\n\n\nSTART nountoskip PROBLEM")
                    #     # fileIE_LOG.write("\n\n\n\nSTART nountoskip PROBLEM")
                    #     # fileIE_LOG.write("\n")
                    #     # print(t.sent)
                    #     # fileIE_LOG.write("\n" + t.sent.text)
                    #     # fileIE_LOG.write("\n\n")
                    #     # print("t")
                    #     # print(t.lemma_)
                    #     # fileIE_LOG.write("t")
                    #     # fileIE_LOG.write(t.lemma_)
                    #     # fileIE_LOG.write("\n")
                    #     # print("nountoskip")
                    #     # print(nountoskip.lemma_)
                    #     # fileIE_LOG.write("nountoskip")
                    #     # fileIE_LOG.write(nountoskip.lemma_)
                    #     # fileIE_LOG.write("\n\n")
                    #     if xin.lemma_.lower() == previousverb.lemma_.lower():
                    #         continue

                    print("...DEBUG ACL ITERATE, VB-{} {}: pos:{}, tag:{} - coming from Noun: {} ".format(xin.dep_,
                                                                                                          xin.lemma_,
                                                                                                          xin.pos_,
                                                                                                          xin.tag_,
                                                                                                          t.lemma_,
                                                                                                          ))

                    minw = min(minw, xin.i)
                    maxw = max(maxw, xin.i)

                    iterated_list_VERB, iterated_list_VERB_sentiment, minw, maxw = VERB_token_IE_parsing(xin,
                                                                                                         singleNOUNs,
                                                                                                         singleCompoundedHITS,
                                                                                                         singleCompoundedHITS_toEXCLUDE,
                                                                                                             filterNOUNs,
                                                                                                             filterCOMPOUNDs,
                                                                                                         LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                         VERBS_TO_KEEP,
                                                                                                         COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                         t, minw, maxw,
                                                                                                         nountoskip=nountoskip,
                                                                                                         previousverb=FoundVerb,
                                                                                                         UseSenticNet=UseSenticNet,
                                                                                                         UseSentiWordNet=UseSentiWordNet,
                                                                                                         UseFigas=UseFigas,
                                                                                                         UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity
                                                                                                         )

                    if iterated_list_VERB and len(iterated_list_VERB) > 0:
                        # #
                        # list_app2 = []
                        # for modin in iterated_list_VERB:
                        #     FoundVBInlist = "___" + xin.lemma_.lower() + "___" + modin
                        #     list_app2.append(FoundVBInlist)
                        # iterated_list_VERB = list_app2
                        # #
                        listAMODs.extend(iterated_list_VERB)
                        if iterated_list_VERB_sentiment and len(iterated_list_VERB_sentiment) > 0:
                            listAMODs_sentim.extend(iterated_list_VERB_sentiment)



                else:

                    print(
                        "...DEBUG ACL ITERATE - NOT CONSIDERED CASE - {} {}: pos:{}, tag:{} - coming from Noun: {} ".format(
                            xin.dep_,
                            xin.lemma_,
                            xin.pos_,
                            xin.tag_,
                            t.lemma_,
                        ))

            # #
            elif xin.dep_ == "prep" and xin.pos_ == "ADP" and xin.lemma_.lower() != t.lemma_.lower() and (
                    nountoskip is None):  # nountoskip is None because I make it only for the first noun

                minw = min(minw, xin.i)
                maxw = max(maxw, xin.i)

                iterated_list_prep, iterated_list_prep_sentim, minw, maxw = PREP_token_IE_parsing(xin, singleNOUNs,
                                                                                                  singleCompoundedHITS,
                                                                                                  singleCompoundedHITS_toEXCLUDE,
                                                                                                      filterNOUNs,
                                                                                                      filterCOMPOUNDs,
                                                                                                  LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                                  VERBS_TO_KEEP,
                                                                                                  COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                                  minw=minw, maxw=maxw,
                                                                                                  FoundVerb=FoundVerb,
                                                                                                  t=t,
                                                                                                  nountoskip=nountoskip,
                                                                                                  UseSenticNet=UseSenticNet,
                                                                                                  UseSentiWordNet=UseSentiWordNet,
                                                                                                  UseFigas=UseFigas,
                                                                                                  UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
                
                if iterated_list_prep and len(iterated_list_prep) > 0:
                    listAMODs.extend(iterated_list_prep)
                    if iterated_list_prep_sentim and len(iterated_list_prep_sentim) > 0:
                        listAMODs_sentim.extend(iterated_list_prep_sentim)

    listAMODs_app = []
    if (FoundNeg_left == "__not" or FoundNeg_right == "__not") and len(listAMODs) > 0:
        for modin in listAMODs:
            listAMODs_app.append(modin + "__not")
        listAMODs = listAMODs_app

    if len(listAMODs) > 0:
        to_give_back.extend(listAMODs)

    if len(listVerbs) > 0:
        to_give_back.extend(listVerbs)

    if len(listAMODs_sentim) > 0:
        to_give_back_sentiment.extend(listAMODs_sentim)

    if len(listVerbs_sentim) > 0:
        to_give_back_sentiment.extend(listVerbs_sentim)

    return to_give_back, to_give_back_sentiment, minw, maxw



# def determine_tense_input(tagged, posextractedn):
#     for ww in tagged:
#         if ww.pos_ == "VERB" and ww.dep_ == "aux" and (
#                 ww.tag_ == "VBP" or ww.tag_ == "VBZ") and ww.head.lower_ == "going":
#             lll = [x for x in ww.head.rights]
#             for xxx in lll:
#                 if xxx.pos_ == "VERB" and xxx.tag_ == "VB":
#                     ww.tag_ = "MD"
#     tense = {}
#     future_words = [word for word in tagged if word.tag_ == "MD"]
#     present_words = [word for word in tagged if word.tag_ in ["VBP", "VBZ", "VBG"]]
#     pass_words = [word for word in tagged if word.tag_ in ["VBD", "VBN"]]
#     inf_words = [word for word in tagged if word.tag_ in ["VB"]]
#     valfuture = 0
#     for word in future_words:
#         valfuture = valfuture + 1 / abs(posextractedn - word.i)
#     valpresent = 0
#     for word in present_words:
#         valpresent = valpresent + 1 / abs(posextractedn - word.i)
#     valpass = 0
#     for word in pass_words:
#         valpass = valpass + 1 / abs(posextractedn - word.i)
#     valinf = 0
#     for word in inf_words:
#         valinf = valinf + 1 / abs(posextractedn - word.i)
#     tense["future"] = valfuture
#     tense["present"] = valpresent
#     tense["past"] = valpass
#     return (tense)



# MD	VERB	VerbType=mod	verb, modal auxiliary
# VB	VERB	VerbForm=inf	verb, base form
# VBD	VERB	VerbForm=fin Tense=past	verb, past tense
# VBG	VERB	VerbForm=part Tense=pres Aspect=prog	verb, gerund or present participle
# VBN	VERB	VerbForm=part Tense=past Aspect=perf	verb, past participle
# VBP	VERB	VerbForm=fin Tense=pres	verb, non-3rd person singular present
# VBZ	VERB	VerbForm=fin Tense=pres Number=sing Person=3	verb, 3rd person singular present


def determine_tense_input_old(tagged):
    # text = word_tokenize(sentence)
    # tagged = pos_tag(text)

    tense = {}
    numfuture = len([word for word in tagged if word.tag_ == "MD"])
    numpresent = len([word for word in tagged if word.tag_ in ["VBP", "VBZ", "VBG"]])
    numpass = len([word for word in tagged if word.tag_ in ["VBD", "VBN"]])

    numinf = len([word for word in tagged if word.tag_ in ["VB"]])  # VB	VERB	VerbForm=inf	verb, base form
    if numfuture > 0:
        tense["future"] = numfuture
    else:
        if numinf > 0 and numpresent >= numpass:
            tense["future"] = numinf
        else:
            tense["future"] = 0

    tense["present"] = numpresent
    tense["past"] = numpass

    return (tense)


# http://esl.fis.edu/grammar/rules/future.htm
def determine_tense_input(tagged, posextractedn):
    # text = word_tokenize(sentence)
    # tagged = pos_tag(text)

    # change is/are going to...to future
    for ww in tagged:
        if ww.pos_ == "VERB" and ww.dep_ == "aux" and (
                ww.tag_ == "VBP" or ww.tag_ == "VBZ") and ww.head.lower_ == "going":
            lll = [x for x in ww.head.rights]
            for xxx in lll:
                if xxx.pos_ == "VERB" and xxx.tag_ == "VB":
                    ww.tag_ = "MD"

    tense = {}
    future_words = [word for word in tagged if word.tag_ == "MD"]
    present_words = [word for word in tagged if word.tag_ in ["VBP", "VBZ", "VBG"]]
    pass_words = [word for word in tagged if word.tag_ in ["VBD", "VBN"]]
    inf_words = [word for word in tagged if word.tag_ in ["VB"]]  # VB	VERB	VerbForm=inf	verb, base form

    numfuture = len(future_words)
    numpresent = len(present_words)
    numpass = len(pass_words)
    numinf = len(inf_words)

    valfuture = 0
    for word in future_words:
        valfuture = valfuture + 1 / abs(posextractedn - word.i)

    valpresent = 0
    for word in present_words:
        valpresent = valpresent + 1 / abs(posextractedn - word.i)

    valpass = 0
    for word in pass_words:
        valpass = valpass + 1 / abs(posextractedn - word.i)

    valinf = 0
    for word in inf_words:
        valinf = valinf + 1 / abs(posextractedn - word.i)

    tense["future"] = valfuture
    # if valfuture > 0:
    #     tense["future"] = valfuture
    # else:
    #     if numinf > 0 and numpresent >= numpass:
    #         tense["future"] = numinf
    #     else:
    #         tense["future"] = 0

    tense["present"] = valpresent
    tense["past"] = valpass

    return (tense)

# BES	VERB		auxiliary "be"
# HVS	VERB		forms of "have"
# MD	VERB	VerbType=mod	verb, modal auxiliary
# VB	VERB	VerbForm=inf	verb, base form
# VBG	VERB	VerbForm=part Tense=pres Aspect=prog	verb, gerund or present participle
# VBP	VERB	VerbForm=fin Tense=pres	verb, non-3rd person singular present
# VBZ	VERB	VerbForm=fin Tense=pres Number=sing Person=3	verb, 3rd person singular present
# VBN	VERB	VerbForm=part Tense=past Aspect=perf	verb, past participle
# VBD	VERB	VerbForm=fin Tense=past	verb, past tense




#def keep_token_IE(t, most_frequent_loc_SENTENCE,docDate,docID):
def keep_token_IE(t, most_frequent_loc_SENTENCE, docDate,docID, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs,
                                                                                             filterCOMPOUNDs,
                                  LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP,
                                  COMPUTE_OVERALL_POLARITY_SCORE, MOST_FREQ_LOC_HEURISTIC, UseSenticNet=False,
                                  UseSentiWordNet=False, UseFigas=True, UseHarvard=False, UseLMD=False, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):

    to_give_back = []
    sentiment_to_give_back = []
    spantogiveback = []
    textsentencetogiveback = []
    tensetogiveback =[]
    locationtogiveback = []


    if t.is_alpha and not (t.is_space or t.is_punct or t.is_stop or t.like_num) and t.pos_ == "NOUN":

        CompoundsOfSingleHit = findCompoundedHITsOfTerm(singleCompoundedHITS, t.lemma_.lower())

        if ((t.lemma_.lower() in filterNOUNs) or (t.lemma_.lower() in singleNOUNs) or (CompoundsOfSingleHit)):

            # print(t.lemma_.lower())

            FoundCompound = None

            ll = [x for x in t.lefts]
            if not FoundCompound:
                if ll:
                    for xin in ll:
                        if ((xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit)) and xin.lemma_.lower() != t.lemma_.lower():  # and xin.dep_=="compound":
                            FoundCompound = xin
                            break

            rr = [x for x in t.rights]
            if not FoundCompound:
                if rr:
                    for xin in rr:
                        if ((xin.lemma_.lower() in filterCOMPOUNDs) or (xin.lemma_.lower() in CompoundsOfSingleHit)) and xin.lemma_.lower() != t.lemma_.lower():  # and xin.dep_=="compound":
                            FoundCompound = xin
                            break

            # if True:
            #     if not FoundCompound:
            #         FoundCompound = t
            if FoundCompound or t.lemma_.lower() in singleNOUNs:

                #
                if (singleCompoundedHITS_toEXCLUDE):
                    CompoundsOfSingleHitToExclude = findCompoundedHITsOfTerm(singleCompoundedHITS_toEXCLUDE, t.lemma_.lower())
                    if (CompoundsOfSingleHitToExclude):
                        if ll:
                            for xin in ll:
                                if (xin.lemma_.lower() in CompoundsOfSingleHitToExclude):
                                    return to_give_back, sentiment_to_give_back, spantogiveback, textsentencetogiveback, tensetogiveback, locationtogiveback  # it is empty
                        if rr:
                            for xin in rr:
                                if (xin.lemma_.lower() in CompoundsOfSingleHitToExclude):
                                    return to_give_back, sentiment_to_give_back, spantogiveback, textsentencetogiveback, tensetogiveback, locationtogiveback  # it is empty
                #

                if FoundCompound:
                    minw = min(FoundCompound.i, t.i)
                    maxw = max(FoundCompound.i, t.i)
                else:
                    minw = t.i
                    maxw = t.i

                # spansentence=t.doc[minw:(maxw+1)]
                # print(spansentence)

                if COMPUTE_OVERALL_POLARITY_SCORE == True:
                    # OVERALL SENTENCE SENTIMENT

                    # #COMPUTE OVERALL SENTENCE POLARITY WITH TEXTBLOB:
                    # testimonial = TextBlob(t.sent.text)  # TEXTBLOB
                    # sentencepolarity = testimonial.sentiment
                    # OSpolarity=sentencepolarity.polarity
                    # OSpolarity=sentencepolarity.polarity
                    # # print("overall sentiment with textblob:")
                    # # print(sentencepolarity)
                    # # assessment_sentiment = testimonial.sentiment_assessments
                    # #

                    # COMPUTE OVERALL SENTENCE POLARITY WITH OUR FEELIT:
                    OSpolarity = FeelIt_OverallSentiment(t, UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet,
                                                         UseFigas=UseFigas, ComputeWithHarvardOnly=UseHarvard,
                                                         ComputeWithMacDonaldOnly=UseLMD, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                    #if DEBUG:
                    #    print((str(t.sent.text)))
                    #    print("overall sentiment with FeelIt:")
                    #    print(OSpolarity)

                    toveralltestsentece__ = t.sent.text.replace(" ", "__")
                    to_give_back.append(toveralltestsentece__)

                    sentiment_to_give_back.append(OSpolarity)

                    minw = t.sent.start
                    maxw = t.sent.end - 1

                else:
                    # FINE-GRAINED, ASPECT-BASED SENTIMENT ANALYSIS
                    to_give_back, sentiment_to_give_back, minw, maxw = NOUN_token_IE_parsing(t, singleNOUNs,
                                                                                             singleCompoundedHITS,
                                                                                             singleCompoundedHITS_toEXCLUDE,
                                                                                                 filterNOUNs,
                                                                                                 filterCOMPOUNDs,
                                                                                             LOCATION_SYNONYMS_FOR_HEURISTIC,
                                                                                             VERBS_TO_KEEP,
                                                                                             COMPUTE_OVERALL_POLARITY_SCORE,
                                                                                             minw=minw, maxw=maxw,
                                                                                             UseSenticNet=UseSenticNet,
                                                                                             UseSentiWordNet=UseSentiWordNet,
                                                                                             UseFigas=UseFigas, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                tryl = True
                while tryl == True:
                    if t.doc[minw].pos_ == "VERB":
                        tryl = False
                        for xis in t.doc[minw].lefts:
                            if (xis.dep_ == "aux" and xis.pos_ == "VERB"):
                                minw = xis.i
                                tryl = True
                    else:
                        tryl = False

                tryl = True
                while tryl == True:
                    if t.doc[maxw].pos_ == "VERB":
                        tryl = False
                        for xis in t.doc[maxw].rights:
                            if (xis.dep_ == "aux" and xis.pos_ == "VERB"):
                                maxw = xis.i
                                tryl = True
                    else:
                        tryl = False

                spansentence = t.doc[minw:(maxw + 1)]

                tensedict = determine_tense_input(spansentence, t.i)


                # tense = "NaN"
                # if tensedict["future"] > 0:
                #     tense = "future"
                # else:
                #     tupletense = max(tensedict.items(), key=operator.itemgetter(1))  # [0]
                #     if tupletense[1] > 0:
                #         tense = tupletense[0]
                tense = "NaN"
                tupletense = max(tensedict.items(), key=operator.itemgetter(1))  # [0]
                if tupletense[1] > 0:
                    tense = tupletense[0]


                tensetogiveback = [str(tense)]

                if (tense in VERBS_TO_KEEP) == False:
                    print("Tense (" + tense + ") not in required list...skipping the spanned text!\n")
                    return [], [], [], [], [], []

                # print(spansentence)

                # from pattern.en import parse, Sentence, parse
                # from pattern.en import modality, mood
                # ss=parse(spansentence.text)
                # ss = parse("this is a trial")
                # thismood = mood(ss)

                if sentiment_to_give_back and len(sentiment_to_give_back) > 0:
                    print("   Final STEP: ")
                    print(sentiment_to_give_back)

                    if (UseSenticNet==False) and (UseSentiWordNet==False) and (UseFigas==False) and (UseLiberty==False) and (UseAuthority==False) and (UseCare==False) and (UseFairness==False) and (UseLoyalty==False) and (UsePurity==False):
                        sentim_noun = FeelIt(t.lemma_.lower(), t.pos_, t, UseSenticNet=True,
                                                 UseSentiWordNet=True, UseFigas=True, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False)
                    else:
                        sentim_noun = FeelIt(t.lemma_.lower(), t.pos_, t, UseSenticNet=UseSenticNet,
                                             UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty,
                                             UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness,
                                             UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                    if FoundCompound:  #02032021
                        if (UseSenticNet == False) and (UseSentiWordNet == False) and (UseFigas == False) and (
                                UseLiberty == False) and (UseAuthority == False) and (UseCare == False) and (
                                UseFairness == False) and (UseLoyalty == False) and (UsePurity == False):
                            sentin=FeelIt(FoundCompound.lemma_.lower(), FoundCompound.pos_, FoundCompound, UseSenticNet=True,
                                                 UseSentiWordNet=True, UseFigas=True, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False)
                        else:
                            sentin = FeelIt(FoundCompound.lemma_.lower(), FoundCompound.pos_, FoundCompound,
                                            UseSenticNet=UseSenticNet,
                                            UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseLiberty=UseLiberty,
                                            UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness,
                                            UseLoyalty=UseLoyalty, UsePurity=UsePurity)

                        if sentin != 0:
                            if sentim_noun == 0:
                                sentim_noun = sentin
                            else:
                                if (sentim_noun > 0 and sentin < 0) or (
                                        sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                                    sentim_noun = np.sign(sentin) * np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # change the sign and increase it
                                else:  # they have same sign
                                    sentim_noun = np.sign(sentim_noun) * (
                                            abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                                        sentin))  # increase it

                    print("    sentilist before correction: ")
                    print(sentiment_to_give_back)

                    # #if sentim_noun
                    # for sentin in sentilist:
                    #     if sentin!=0:
                    #         if (sentim_noun > 0 and sentin < 0) or (
                    #                 sentim_noun < 0 and sentin > 0):  # they have different signes, I just sum them up
                    #
                    #             sentim_app = (-1) * np.sign(sentim_noun) * (
                    #                     abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                    #                 sentin))  # change the sign and increase it
                    #             sentiment_to_give_back.append(sentim_app)
                    #
                    #         else:  # they have same sign
                    #             sentim_app = np.sign(sentim_noun) * (
                    #                     abs(sentim_noun) + (1 - abs(sentim_noun)) * abs(
                    #                 sentin))  # increase it
                    #             sentiment_to_give_back.append(sentim_app)
                    #     else:
                    #         sentiment_to_give_back.append(0) #(sentin

                    # if countersent>0:
                    #    sentiment_to_give_back=sentiment_to_give_back/countersenti

                    # sentiment_to_give_back = sentilist

                if to_give_back and len(to_give_back) > 0:

                    if len(to_give_back) == len(sentiment_to_give_back):

                        #
                        listNoun_app = []
                        listSentim_app = []

                        FoundNounInlist_ALLTOGETHER = ""
                        if FoundCompound:
                            FoundNounInlist_ALLTOGETHER = "---" + FoundCompound.lemma_.lower() + " " + t.lemma_.lower() + "---"
                        else:
                            FoundNounInlist_ALLTOGETHER = "---" + t.lemma_.lower() + "---"

                        numberofvaluesent = 0
                        sumofvaluessent = 0

                        for ii in range(len(to_give_back)):  # modin in to_give_back:
                            modin = to_give_back[ii]
                            sentin = sentiment_to_give_back[ii]
                            if FoundCompound:
                                # print('{}\t{}'.format(FoundCompound.lemma_.lower(), t.lemma_.lower()))
                                FoundNounInlist = "---" + FoundCompound.lemma_.lower() + " " + t.lemma_.lower() + "---" + modin
                            else:
                                # print('{}'.format(t.lemma_.lower()))
                                FoundNounInlist = "---" + t.lemma_.lower() + "---" + modin
                            listNoun_app.append(FoundNounInlist)

                            FoundNounInlist_ALLTOGETHER = FoundNounInlist_ALLTOGETHER + "+++" + modin

                            # #
                            # sentim_app = sentim_noun
                            # if sentim_app==0:
                            #     sentim_app=sentin
                            # else:
                            #     if sentin != 0:
                            #         if (sentim_app > 0 and sentin < 0) or (
                            #                 sentim_app < 0 and sentin > 0):  # they have different signes, I just sum them up
                            #
                            #             sentim_app = (-1) * np.sign(sentim_app) * (
                            #                     abs(sentim_app) + (1 - abs(sentim_app)) * abs(
                            #                 sentin))  # change the sign and increase it
                            #             sentiment_to_give_back.append(sentim_app)
                            #
                            #         else:  # they have same sign
                            #             sentim_app = np.sign(sentim_app) * (
                            #                     abs(sentim_app) + (1 - abs(sentim_app)) * abs(
                            #                 sentin))  # increase it
                            #     else:
                            #        sentim_app = 0  # (sentin

                            # if sentin > 0:
                            #     if sentim_noun < 0:
                            #         sentin = (-1) * sentin
                            # else:
                            #     if sentim_noun < 0:
                            #         sentin = (-1) * sentin

                            if sentin != 0 and sentim_noun != 0:
                                sentin = np.sign(sentin) * np.sign(sentim_noun) * abs(sentin)  # change only the sign

                            if "__not" in FoundNounInlist:
                                sentin = (-1) * sentin

                            listSentim_app.append(sentin)

                            if sentin != 0:
                                numberofvaluesent = numberofvaluesent + 1
                                sumofvaluessent = sumofvaluessent + sentin

                            # print("--> " + modin)
                            # fileIE_LOG.write('{}'.format(t.lemma_.lower()))
                            # fileIE_LOG.write("\n--> " + modin)
                            print('--> {} ({})'.format(FoundNounInlist, sentin))  # print('{}'.format(t.lemma_.lower()))
                            #

                        # to_give_back = listNoun_app
                        # sentiment_to_give_back = listSentim_app
                        #
                        listNoun_app2 = []
                        listSentim_app2 = []
                        listNoun_app2.append(FoundNounInlist_ALLTOGETHER)
                        if numberofvaluesent > 0:
                            avgsent_ALLTOGETHER = sumofvaluessent / numberofvaluesent
                        else:
                            avgsent_ALLTOGETHER = 0

                        listSentim_app2.append(avgsent_ALLTOGETHER)
                        to_give_back = listNoun_app2
                        sentiment_to_give_back = listSentim_app2
                        #

                        spantogiveback = [spansentence.text]

                        textsentencetogiveback = [t.sent.text]

                        locationtogiveback.append(most_frequent_loc_SENTENCE.lower())

                        print(
                            "   Found ALLTOGETHER (" + tense + ") = " + FoundNounInlist_ALLTOGETHER + "  ...  with SENTIM = " + str(
                                avgsent_ALLTOGETHER))
                        #
                        print("   Found Chunk (" + tense + ") = " + spansentence.text + "  ...  " + str(tensedict))
                        print("   - Found in sentence (" + str(locationtogiveback) + "): " + t.sent.text)
                        #if DEBUG:
                        #    print("   most_frequent_loc_DOC: " + most_frequent_loc_DOC)
                        #    print("   most_frequent_loc_SENTENCE: " + most_frequent_loc_SENTENCE)
                        #
                        print("\n")


                        #write extraction summary on csv file
                        #writerCSV_ExtractionsSummary.writerow([docDate,docID,t.sent.text,spansentence.text,str(avgsent_ALLTOGETHER)])
                        # print("---pippo, len" + str(len(pippo)))
                        # pippo.append([docDate,docID,t.sent.text,spansentence.text,str(avgsent_ALLTOGETHER)])
                        # print(docDate)
                        # print(docID)
                        # print(t.sent.text)
                        # print(spansentence.text)
                        # print(str(avgsent_ALLTOGETHER))
                        # print("----pippo after , len" + str(len(pippo)))


                    else:
                        print("ERROR! Terms and sentim lists should have same lenght! This happened in Sentence: ")
                        print(t.sent)
                        print("\n")

    return to_give_back, sentiment_to_give_back, spantogiveback, textsentencetogiveback, tensetogiveback, locationtogiveback




def Most_Common(lista):
    mostc = ""
    if lista:
        data = Counter(lista)
        ordered_c = data.most_common()
        mostc = ordered_c[0][0]
        max_freq = ordered_c[0][1]
        for j in range(0, len(ordered_c)):
            if ordered_c[j][1] < max_freq:
                break
    return mostc


def findCompoundedHITsOfTerm(vector, term):
    term = str(term)
    outArray = []
    for x in vector:
        if term.lower() in x.lower():
            compoundMinusTerm = x.lower().replace(term.lower(), "").strip()
            outArray.append(compoundMinusTerm)
    return outArray


def determine_location_heuristic(doc_entities, posextractedn, t,LOCATION_SYNONYMS_FOR_HEURISTIC):
    most_probable_loc = ""
    tagged = []
    for loc in doc_entities:
        if loc.label_ == "GPE" or loc.label_ == "NORP" or loc.label_ == "LOC" or loc.label_ == "ORG":
            tagged.append(loc)
    if len(tagged) > 0:
        unique_loc_labels = []
        unique_loc_values = []
        for loc in tagged:
            x = loc.lemma_.lower()
            if (x in LOCATION_SYNONYMS_FOR_HEURISTIC) or (removearticles(x) in LOCATION_SYNONYMS_FOR_HEURISTIC):
                x = LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower()
            if x not in unique_loc_labels:
                unique_loc_labels.append(x)
                valword = 0
                for word in tagged:
                    y = word.lemma_.lower()
                    if ((y in LOCATION_SYNONYMS_FOR_HEURISTIC) or (
                            removearticles(y) in LOCATION_SYNONYMS_FOR_HEURISTIC)):
                        y = LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower()
                    if y == x:
                        avgpostne = word.start + int((word.end - word.start) / 2)
                        dividendum = abs(posextractedn - avgpostne)
                        if dividendum == 0:
                            dividendum = 1
                        valword = valword + 1 / dividendum
                unique_loc_values.append(valword)
        maxvalue = max(unique_loc_values)
        indices_max = [i for i, x in enumerate(unique_loc_values) if x == maxvalue]
        most_probable_loc = unique_loc_labels[indices_max[0]] 
        for wwind in indices_max:  
            ww = unique_loc_labels[wwind]
            if ww == LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower():
                most_probable_loc = ww
    return most_probable_loc


def removearticles(text):
    removed = re.sub('\s+(a|an|and|the)(\s+)', ' ', " " + text + " ")
    removed = re.sub('  +', ' ', removed)
    removed = removed.strip()
    return removed




def CheckLeapYear(year):
    isleap = False
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                isleap = True
            else:
                isleap = False
        else:
            isleap = True
    else:
        isleap = False
    return isleap


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out