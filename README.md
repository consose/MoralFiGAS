# MoralFiGAS
Extension of the FiGAS (Fine-Grained Aspect-based Sentiment) analysis to the extraction of Moral Values (liberty, care, fairness, loyalty, authority, purity from text 

This package allows Python users to leverage on cutting-hedge NLP techniques to easily run sentiment analysis and moral values extraction on text.
Given a list of texts as input and a list of tokens of interest (ToI), the algorithm analyses the texts and compute the sentiment associated each ToI. Two key features characterize this approach. First, it is *fine-grained*, since words are assigned a polarity score that ranges in \[-1,1\] based on a dictionary. Second, it is *aspect-based*, since the algorithm selects the chunk of text that relates to the ToI based on a set of semantic rules and calculates the sentiment only on that chunk,
rather than the entire text.

The package includes some additional of features, like automatic negation handling, tense detection, location filtering and excluding some words from the sentiment computation. The algorithm only supports English language, as it relies on the *en\_core\_web\_lg* language model from the `spaCy` Python module.

## Installation

(Python version >= 3.7.6) 

Make sure to install the following libraries:

``` bash
pip install utm lxml numpy spacy nltk toolz gensim pandas sentence-splitter sense2vec 
```
(verified spaCy version = 3.5.0)

Set up NLTK:

``` bash
python -m nltk.downloader all
python -m nltk.downloader wordnet
python -m nltk.downloader omw
python -m nltk.downloader sentiwordnet
```

Finally get spacy models:

``` bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

## Example

Let’s assume that you want to compute the sentiment associated to two tokens of interest, namely *unemployment* and *economy*, given the two following sentences.

``` bash
text = ['Unemployment is rising at high speed', 'The economy is slowing down and unemployment is booming']
include = ['unemployment', 'economy']
```
Se select to calculate the sentiment using the SentiBigNomics lexicon (FiGAS economic lexicon) and use the aspect-based approach:

``` bash
UseFigas=True
oss=False
```

Then you launch the algorithm:

``` bash
resp = get_polarity_standalone(text = text, include = include, oss=oss, UseFigas=UseFigas)
```

The output of the function is a list, containing the entire sentence, the chunk pattern which activated the sentiment calculation, the polarity score, and the detected tense:

``` bash
POLARITY RESULTS:

['Unemployment is rising at high speed'] --> ---unemployment---+++___rise______speed___high --> -0.85 ['present']

['The economy is slowing down and unemployment is booming'] --> ---economy---+++slow --> -0.4 ['present']

['The economy is slowing down and unemployment is booming'] --> ---unemployment---+++boom --> -0.8 ['present']
```

Other lexicon are now supported. For sentiment analysis, namely SenticNet, Sentiwordnet, Harvard IV and Loughran-McDonald (this two are word lists, therefore the sentiment can be calculated only at the sentence level, i.e. oss = True.

We now support Moral values lexicons (MoralStrength and Liberty MFD, https://github.com/oaraque/moral-foundations). 
As an exampe of *unemployment* for the token of interest *country*:

``` bash
text = ['The country has restricting personal regulations']
include = ['country']
```

``` bash
resp = get_polarity_standalone(text = text, include = include, oss=oss, UseLiberty=True)
```
getting in output:

``` bash
POLARITY RESULTS:

['The country has restricting personal regulations'] --> ---country---+++___restrict______regulation___personal --> -0.81 ['present']
```




## Citations:

If you use this package, we encourage you to add the following references to the related papers:

<!-- ## References: -->

- Araque, O.; Gatti, L.; and Kalimeri, K. 2020. MoralStrength: Exploiting a moral lexicon and embedding similarity for moral foundations prediction. Knowledge-Based Systems, 191
 
-Araque, O.; Gatti, L.; and Kalimeri, K. 2022. LibertyMFD: A Lexicon to Assess the Moral Foundation of Liberty. In ACM International Conference Proceeding Series, 154 – 160

 - Barbaglia, L.; Consoli, S.; and Manzan, S. 2022. Forecasting with Economic News. Journal of Business and Economic Statistics, to appear. Available at SSRN: <https://ssrn.com/abstract=3698121>

- Consoli, S.; Barbaglia, L.; and Manzan, S. 2020. Fine-grained, aspect-based semantic sentiment analysis within the economic and financial domains. In Proceedings - 2020 IEEE 2nd International Conference on Cognitive Machine Intelligence, CogMI 2020, 52 – 61

- Consoli, S.; Barbaglia, L.; and Manzan, S. 2021. Explaining sentiment from Lexicon. In CEUR Workshop Proceedings, volume 2918, 87 – 95

- Consoli, S.; Barbaglia, L.; and Manzan, S. 2022. Fine-grained, aspect-based sentiment analysis on economic and financial lexicon. Knowledge-Based Systems, 247: 108781



