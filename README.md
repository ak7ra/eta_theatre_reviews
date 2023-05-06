# Exploratory Text Analysis of Theatrical Performance Reviews

Theatrical performances are an art form that are expressed through rich, live storytelling. But how are those expressions received? What kind of discourse occurs from them? This project analyzes approximately 300 theatre performance reviews using Principle Component Analysis, Topic Models, Word Embeddings (word2vec) and Sentiment Analysis. Through these analyses, which are summarized in the file `FINAL_REPORT.ipynb`, insights into theatrical performance reviews are uncovered.

## Source Data

The source data for this analysis was collected from the website https://www.playshakespeare.com/. This website describes itself as 'the best Shakespeare source for plays, news, reviews, and discussion' and contains, amongst other contents, reviews of theatrical performances based on the works of William Shakespeare. The webpages containing the reviews were manually downloaded as HTML files and are stored in the `data` subdirectory of this repository (I do not own the rights to this content, and this content is used for educational purposes). In total, there were 292 HTML files which were later parsed into more analysis-appropriate formats in the file `parser.ipynb`. 

## Metadata
**Name:** Ami Kano <br />
**GitHub Username:** ak7ra <br />
**Project Name:** Exploratory Text Analysis of Theatrical Performance Reviews

## Synopsis

### Required Python Packages

To run the files within this repository, one must have these Python Packages installed:

* `pandas`
* `numpy`
* `nltk`
* `sklearn.feature_extraction.text`
* `sklearn.decomposition`
* `ast`
* `plotly_express`
* `matplotlib.pyplot`
* `seaborn`
* `os`
* `bs4`
* `nltk.stem.porter`
* `nltk.stem.snowball`
* `nltk.stem.lancaster`
* `re`
* `tqdm`
* `scipy.linalg`
* `scipy.spatial.distance`
* `gensim.models`
* `sklearn.manifold`
* `scipy.linalg`
* `scipy.cluster.hierarchy`
