from common_functions import *


def lemmatize_doc_IE_Sentiment(doc,singleNOUNs,singleCompoundedHITS,singleCompoundedHITS_toEXCLUDE,filterNOUNs,filterCOMPOUNDs,LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE,MOST_FREQ_LOC_HEURISTIC,UseSenticNet=False, UseSentiWordNet=False, UseFigas=True, UseHarvard=False, UseLMD=False, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):
    vect = []
    vect_sentiment = []
    vect_spans = []
    vect_text = []
    vect_tense = []
    most_frequent_loc = ""
    if MOST_FREQ_LOC_HEURISTIC is True:
        locations = [loc.lemma_.lower() for loc in doc.ents if
                     (loc.label_ == "GPE" or loc.label_ == "NORP" or loc.label_ == "LOC" or loc.label_ == "ORG" )]
        locations = [LOCATION_SYNONYMS_FOR_HEURISTIC[0].lower() if ((x in LOCATION_SYNONYMS_FOR_HEURISTIC) or (
                removearticles(x) in LOCATION_SYNONYMS_FOR_HEURISTIC)) else x for x in locations]
        most_frequent_loc = Most_Common(locations)
    else:
        most_frequent_loc = ""
    sentencealreadyseen = ""
    for t in doc:
        vec_for_term, vec_for_sent, spansse, texttse, tensesse, locatsse = keep_token_IE(t,most_frequent_loc,"","",singleNOUNs,singleCompoundedHITS,singleCompoundedHITS_toEXCLUDE,filterNOUNs, filterCOMPOUNDs, LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE, MOST_FREQ_LOC_HEURISTIC,UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseHarvard=UseHarvard, UseLMD=UseLMD, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
        if vec_for_term:
            if len(vec_for_term) > 0:
                if COMPUTE_OVERALL_POLARITY_SCORE == True:
                    thissentence = str(t.sent.text)
                    if (thissentence == sentencealreadyseen):
                        continue
                    else:
                        sentencealreadyseen = str(t.sent.text)
                        vect.extend(vec_for_term)
                        vect_sentiment.extend(vec_for_sent)
                        vect_spans.append(spansse)
                        vect_text.append(texttse)
                        vect_tense.append(tensesse)
                else:
                    vect.extend(vec_for_term)
                    vect_sentiment.extend(vec_for_sent)
                    vect_spans.append(spansse)
                    vect_text.append(texttse)
                    vect_tense.append(tensesse)
    return vect, vect_sentiment, vect_spans, vect_text, vect_tense


def get_polarity_standalone(text, include, exclude=None, location=None, tense=['past', 'present', 'future', 'NaN'], oss=False, UseSenticNet=False, UseSentiWordNet=False, UseFigas=False, UseHarvard=False, UseLMD=False, UseLiberty=False, UseAuthority=False, UseCare=False, UseFairness=False, UseLoyalty=False, UsePurity=False):
    # text = ['Today is a beautiful day', 'The economy is slowing down and it is a rainy day']
    # include = ['day', 'economy']
    # exclude=None 
    # location=None
    # tense=['past', 'present', 'future', 'NaN']
    # oss=False
    toINCLUDE = include
    singleCompoundedHITS_toEXCLUDE = exclude
    LOCATION_SYNONYMS_FOR_HEURISTIC = location
    VERBS_TO_KEEP = tense
    COMPUTE_OVERALL_POLARITY_SCORE = oss
    for i in range(len(text)):
        text[i] = re.sub("\n \\n", " ", str(text[i]))
        
    if LOCATION_SYNONYMS_FOR_HEURISTIC and len(LOCATION_SYNONYMS_FOR_HEURISTIC) > 0:
        MOST_FREQ_LOC_HEURISTIC = True
    else:
        MOST_FREQ_LOC_HEURISTIC = False
    singleNOUNs = []
    singleCompoundedHITS = []
    for ii in toINCLUDE:
        if " " in ii:
            singleCompoundedHITS.append(ii)
        else:
            singleNOUNs.append(ii)
            
    currentDT = datetime.now()
    spacy_model_name_EN = 'en_core_web_lg'
    # from timeit import default_timer as timer
    # start = timer()
    # print("spaCy is loading the en_core_web_lg model ...")
    nlp_EN = spacy.load(spacy_model_name_EN) ## this operation takes approximately 10seconds
    # print(timer()-start) ## elapsed time in seconds
    LA_target = 'en'
    docs_lemma = []
    docs_lemma_sentiment = []
    docsspans = []
    docstexttt = []
    docstense = []
    DF_ExtractionsSummary = []

    filterNOUNs = []

    filterCOMPOUNDs = []

    for j in range(len(text)):
        nlp_COUNTRYdoc = nlp_EN(text[j])
        lemmatized_doc, lemmatized_doc_sent, spanss, texttt, tensesss = lemmatize_doc_IE_Sentiment(
            nlp_COUNTRYdoc, singleNOUNs, singleCompoundedHITS, singleCompoundedHITS_toEXCLUDE, filterNOUNs, filterCOMPOUNDs,
            LOCATION_SYNONYMS_FOR_HEURISTIC, VERBS_TO_KEEP, COMPUTE_OVERALL_POLARITY_SCORE, MOST_FREQ_LOC_HEURISTIC,UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseHarvard=UseHarvard, UseLMD=UseLMD, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)
        docs_lemma.append(lemmatized_doc)
        docs_lemma_sentiment.append(lemmatized_doc_sent)
        docsspans.append(spanss)
        docstexttt.append(texttt)
        docstense.append(tensesss)
        for i in range(len(docstexttt[j])):
            includedNOUN = []
            check = (singleNOUNs + singleCompoundedHITS)
            for k in check:
                if k in str(docsspans[j][i]).lower():
                    includedNOUN.append(k)
            DF_ExtractionsSummary.append([j, docstexttt[j][i], docsspans[j][i], docs_lemma[j][i],
                                          docs_lemma_sentiment[j][i], docstense[j][i], includedNOUN])

    DF_ExtractionsSummary = pd.DataFrame(DF_ExtractionsSummary, columns=['Doc_id', 'Text', 'SpannedText', 'Chunk',
                                                                         'Sentiment', 'Tense', 'Include'])
    return DF_ExtractionsSummary


#####




#####

print("\nSTART RUN\n")

#text = ['Unemployment is rising at high speed', 'The economy is slowing down and unemployment is booming']
text = ['The country has restricting personal regulations']

include = ['country']

oss=False

UseFigas=False #False #True
UseSenticNet=False #False #True
UseSentiWordNet=False #False #True

UseHarvard=False  #False #True
UseLMD=False  #False #True

UseLiberty = False #False #True
UseAuthority = False #False #True
UseCare = False #False #True
UseFairness = False #False #True
UseLoyalty = False #False #True
UsePurity = False #False #True

if ((UseHarvard==True) or (UseLMD==True)) and ( (UseLiberty == True) or (UseAuthority == True) or (UseCare == True) or (UseFairness == True) or (UseLoyalty == True) or (UsePurity == True) or (UseSenticNet == True) or (UseSentiWordNet == True) or (UseFigas == True) ):
    print("\n!!!!WARNING!!!\nHarvard and Loughran & Mac Donald are binary sentiment dictionaries, not fine-grained. The computation of the sentiment is based on counting the signs of the sentiment of the words according to these dictionaries.")
    print("The sentiment calculation is possible at the overall sentence level, therefore oss is automatically imposed to True!")
    print("Disabling also any fine-grained sentiment dictionary option and any moral polarity lexicon ...\n")
    oss=True
    UseSenticNet=False
    UseSentiWordNet=False
    UseFigas=False
    UseLiberty=False
    UseAuthority=False
    UseCare=False
    UseFairness=False
    UseLoyalty=False
    UsePurity=False
elif ((UseLiberty == True) or (UseAuthority == True) or (UseCare == True) or (UseFairness == True) or (UseLoyalty == True) or (UsePurity == True)) and ((UseSenticNet == True) or (UseSentiWordNet == True) or (UseFigas == True) or (UseHarvard == True) or (UseLMD == True) ):
    print("\n!!!!WARNING!!!\nA moral polarity computation has been selected")
    print("Disabling any sentiment dictionary option ...\n")
    UseSenticNet = False
    UseSentiWordNet = False
    UseFigas = False
    UseHarvard = False
    UseLMD = False


############

#resp = get_polarity_standalone(text = text, include = include, oss=oss, UseSenticNet=UseSenticNet, UseSentiWordNet=UseSentiWordNet, UseFigas=UseFigas, UseHarvard=UseHarvard, UseLMD=UseLMD, UseLiberty=UseLiberty, UseAuthority=UseAuthority, UseCare=UseCare, UseFairness=UseFairness, UseLoyalty=UseLoyalty, UsePurity=UsePurity)

resp = get_polarity_standalone(text = text, include = include, oss=oss, UseLiberty=True)

print("\nPOLARITY RESULTS:\n\n")
for res in resp.values:
    print(str(res[1]) + " --> " + str(res[3]) + " --> " + str(res[4]) + " " + str(res[5]) )
    print("\n")
#print(resp.values)


print("\nEND\n")