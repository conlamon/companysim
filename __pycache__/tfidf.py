import math
import re
import string
#import nltk


class TfIdf:
    """Tf-idf class implementing http://en.wikipedia.org/wiki/Tf-idf.

     The library constructs an IDF corpus and stopword list either from
     documents specified by the client, or by reading from input files.  It
     computes IDF for a specified term based on the corpus, or generates
     keywords ordered by tf-idf for a specified document.
    """

    def __init__(self, corpus_filename=None, stopword_filename=None,
                 DEFAULT_IDF=1.5):
        """Initialize the idf dictionary.

        If a corpus file is supplied, reads the idf dictionary from it, in the
        format of:
        # of total documents
        term: # of documents containing the term

        If a stopword file is specified, reads the stopword list from it, in
        the format of one stopword per line.

        The DEFAULT_IDF value is returned when a query term is not found in the
        idf corpus.
        """
        self.num_docs = 0
        self.term_num_docs = {}  # term : num_docs_containing_term
        self.stopwords = []
        self.idf_default = DEFAULT_IDF

        if corpus_filename:
            corpus_file = open(corpus_filename, "r")

            # Load number of documents.
            line = corpus_file.readline()
            self.num_docs = int(line.strip())

            # Reads "term:frequency" from each subsequent line in the file.
            for line in corpus_file:
                tokens = line.split(":")
                term = tokens[0].strip()
                frequency = int(tokens[1].strip())
                self.term_num_docs[term] = frequency

        if stopword_filename:
            stopword_file = open(stopword_filename, "r")
            self.stopwords = [line.strip() for line in stopword_file]

    def get_tokens(self, doc):
        """Break a string into tokens, preserving URL tags as an entire token.

        This implementation does not preserve case.
        Clients may wish to override this behavior with their own tokenization.
        """
        str_list = [word.lower().translate(str.maketrans(' ', ' ', string.punctuation))
                    for word in re.split('\s|\.|-|,', str(doc))]

        # If too slow, delete the stop word removal
        #stopwords = set(nltk.corpus.stopwords.words('english'))
        return set(str_list)  #- stopwords

    def add_input_document(self, doc_in):
        """Add terms in the specified document to the idf dictionary."""
        self.num_docs += 1
        words = set(self.get_tokens(doc_in))
        for word in words:
            if word in self.term_num_docs:
                self.term_num_docs[word] += 1
            else:
                self.term_num_docs[word] = 1

    def get_num_docs(self):
        """Return the total number of documents in the IDF corpus."""
        return self.num_docs

    def get_idf(self, term):
        """Retrieve the IDF for the specified term.

        This is computed by taking the logarithm of (
        (number of documents in corpus) divided by (number of documents
        containing this term) ).
        """
        if term in self.stopwords:
            return 0

        if not term in self.term_num_docs:
            return self.idf_default

        return math.log(float(1 + self.get_num_docs()) / (1 + self.term_num_docs[term]))