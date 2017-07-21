from datasketch import MinHashLSHForest, MinHash
from scipy.sparse import lil_matrix
import pandas as pd
import companysim.tfidf as tfidf
#import tfidf as tfidf
import re
import string
import pickle
import sys


class CompanyCorpus(object):

    """
    Class representing a company corpus of text descriptions

    Parameters:

        corpus - pandas dataframe; company descriptions

        idf (optional) - pandas series; series containing the idf words
    """

    def __init__(self, corpus, idf=None):

        self.corpus = self._build_corpus(corpus).copy()

        if idf is not None:
            if isinstance(idf, pd.Series):
                self.idf_vector = idf
                self.idf_vector_created = True
            else:
                raise TypeError("Idf vector must be a pandas series")
        else:
            self.idf_vector = None

        self.idf_vector_created = False
        self.filtered_by_idf = False

    @staticmethod
    def _build_corpus(corpus):

        """
        Drop duplicate companies and set internal variable

        Parameters:

            corpus - pandas dataframe; company descriptions

        """

        if not isinstance(corpus, pd.DataFrame):
            raise TypeError("Corpus must be a pandas DataFrame")

        # Clean up corpus by dropping duplicate companies
        corpus = corpus.drop_duplicates()

        return corpus

    @staticmethod
    def _get_words(phrase):

        """
        Split the descriptions up into individual words

        Parameters:

            phrase - string; description to parse into words
        """

        words = [word.lower().translate(str.maketrans(' ', ' ', string.punctuation))
                 for word in re.split('\s|\.|-|,', str(phrase))]

        return set(words)

    def build_idf(self, description_column_name, out_file=None, csv_location=None):

        """
        Build the inverse document frequency vector

        Parameters:

            description_column_name - string; name of the description column in the dataframe

            out_file (optional) - string; filename to save the idf vector

            csv_location (optional) - string; location of csv file containing the pre-built IDF vector to use
        """

        idfcalc = tfidf.TfIdf()

        for entry in self.corpus.loc[:, description_column_name].values:
            idfcalc.add_input_document(entry)

        idf_list = []
        term_list = []

        for term in idfcalc.term_num_docs:
            idf = idfcalc.get_idf(term)
            idf_list.append(idf)
            term_list.append(term)

        idf_vector = pd.Series(idf_list, index=term_list)
        idf_vector = idf_vector.sort_values(ascending=False)

        if out_file:
            if csv_location:
                idf_vector.to_csv(csv_location + '/' + out_file)
            else:
                print("Error: no location specified for output csv")

        self.idf_vector_created = True
        self.idf_vector = idf_vector

        return idf_list, term_list

    def filter_desc_by_idf(self, description_column_name, number_words_to_cut=50):

        """
        Filter each description by removing words that have low IDF values

        Parameters:

            description_column_name - string; name of the description column in the dataframe

            number_words_to_cut - integer; number of lowest IDF words to remove
        """

        if self.idf_vector_created is False:
            raise BrokenPipeError("IDF vector not created. Must create it before running this function.")

        n = self.idf_vector.shape[0] - number_words_to_cut
        top_n_idf_vec = self.idf_vector.ix[0:n]
        top_words_set = set(top_n_idf_vec.index.values)

        iteration = 1
        temp_col_vec = []

        # Loop through each description
        sys.stdout.write("Filtering descriptions by IDF...")
        for desc in self.corpus.loc[:, description_column_name]:
            words = self._get_words(str(desc))

            # take intersection of the words with the top idf words and add to a list
            final_words = words.intersection(top_words_set)
            temp_col_vec.append(final_words)
            iteration += 1

        sys.stdout.write('\n')
        sys.stdout.write("Done filtering!\n")

        # Set the internal tracker to verify we performed the filter
        self.filtered_by_idf = True

        # Convert the list to a pandas series for ease of use and addition to the corpus dataframe
        final_rare_words_vector = pd.Series(temp_col_vec)

        self.corpus.loc[:, 'rare_words'] = temp_col_vec

        return final_rare_words_vector


class CompanyGraph(object):

    """
    Class representing a company graph

    Parameters:

        company_corpus_instance - CompanyCorpus; a pre-instantiated CompanyCorpus object
    """

    def __init__(self, company_corpus_instance):

        self.graph = None
        self.graph_built = None
        self.company_corpus = company_corpus_instance
        self.dict_of_minhash_keys = {}
        self.lsh_forest = None
        self.company_name_column_name = None
        self.name_to_index_map = None
        self.index_to_name_map = None

    def build_lsh_forest(self, company_name_column_name):

        """
        Build the LSH forest data structure from the sets of parsed description words for each company

        Parameters:

            company_name_column_name - string; name of the company name column in the company corpus dataframe
        """
        # Note: num_perm is a tuning parameter, but has been abstracted away for simplicity
        #       256 has been found to be a good amount. Increasing it may increase accuracy,
        #       but will decrease speed and increase memory usage. Decreasing will decrease accuracy

        lsh_forest = MinHashLSHForest(num_perm=256)

        iteration = 1

        self.company_name_column_name = company_name_column_name
        self.name_to_index_map = dict(zip(self.company_corpus.corpus.loc[:, company_name_column_name],
                                          self.company_corpus.corpus.index))
        self.index_to_name_map = dict(zip(self.company_corpus.corpus.index,
                                          self.company_corpus.corpus.loc[:, company_name_column_name]))

        sys.stdout.write("Performing LSH...")
        for company in self.company_corpus.corpus.iterrows():

            # Utilize the 'datasketch' library to minhash the company descriptions and hash to LSh forest
            company_name = company[1][company_name_column_name]
            if company_name in self.dict_of_minhash_keys:
                continue
            mh = MinHash(num_perm=256)
            if type(company[1]['rare_words']) is float:
                mh.update(str(company[1]['rare_words']).encode('utf8'))
            else:
                for word in company[1]['rare_words']:
                    mh.update(str(word).encode('utf8'))
            self.dict_of_minhash_keys[company_name] = mh
            lsh_forest.add(company_name, mh)

            iteration += 1
        sys.stdout.write('\n')
        sys.stdout.write("Done performing LSH!\n")

        # Need this line below to be able to query LSH forest! (See datasketch docs on LSH forest for reasoning)
        lsh_forest.index()
        self.lsh_forest = lsh_forest

    @staticmethod
    def _get_weighted_jaccard_similarity(company_index_1, company_index_2,
                                         company_words_list, idf_set, idf_map_dict):

        """
        Calculate the weighted Jaccard similarity

        Parameters:

            company_index_1 - integer; index of first company in the pair

            company_index_2 - integer; index of secont company in the pair

            company_words_list - dictionary; table matching company words to the company index

            idf_set - set; a set of the idf words

            idf_map_dict - dictionary; idf value to word look up table
        """

        company_1 = company_words_list[company_index_1]
        company_2 = company_words_list[company_index_2]

        intersection = company_1 & company_2
        union = company_1 | company_2

        if len(union) == 0:
            return 0

        intersection_score = 0.0
        union_score = 0.0
        for word in union:
            if word in idf_set:
                word_score = idf_map_dict[word]
                union_score += word_score
                if word in intersection:
                    intersection_score += word_score

        return intersection_score/union_score

    def build_graph(self, sensitivity):

        """
        Build the graph adjacency matrix based on the LSH hashed values

        Parameters:

            sensitivity - integer; the number of similar companies to query for each company from the LSH forest,
                            larger the better (generally), as too small will not have enough sensitivity
        """

        if self.lsh_forest is None:
            raise ValueError("LSH forest not yet created. Run build_lsh_forest function first.")

        # Setup some needed parameters
        num_companies = self.company_corpus.corpus.shape[0]
        idf_set = set(self.company_corpus.idf_vector.index)
        idf_map_dict = dict(zip(self.company_corpus.idf_vector.index, self.company_corpus.idf_vector.values))
        company_words_list = list(self.company_corpus.corpus.loc[:, 'rare_words'])

        iteration = 0
        # Create a sparse matrix format using Scipy
        company_sim_adjacency_matrix = lil_matrix((num_companies, num_companies))

        print("Building graph...")
        for company1 in self.company_corpus.corpus[self.company_name_column_name]:
            company1_hash = self.dict_of_minhash_keys[company1]
            k_sim_comps = self.lsh_forest.query(company1_hash, sensitivity)
            for company2 in k_sim_comps:
                company1_index = iteration

                company2_index = self.name_to_index_map[company2]

                # Compute the weighted Jaccard similarity
                wjs = self._get_weighted_jaccard_similarity(company_index_1=company1_index,
                                                            company_index_2=company2_index,
                                                            company_words_list=company_words_list,
                                                            idf_set=idf_set,
                                                            idf_map_dict=idf_map_dict)
                # Place value into adjacency matrix
                company_sim_adjacency_matrix[company1_index, company2_index] = wjs
                company_sim_adjacency_matrix[company2_index, company1_index] = wjs

            iteration += 1
        sys.stdout.write('\n')
        print("Done building graph!")

        # noinspection PyTypeChecker
        company_sim_adjacency_matrix.setdiag(0)

        # Scipy csr sparse matrix format is better for accessing rows and performing dot products, so convert to this
        # format for all future use

        company_graph = company_sim_adjacency_matrix.tocsr()
        self.graph = company_graph

    def get_dot_product_score(self, company1, company2):

        """
        Calculate the dot product score using the rows of the (symmetric) adjacency matrix

        Parameters:

            company1 - string; name of the first company

            company2 - string; name of the second company
        """

        if self.graph is None:
            raise ValueError("No graph created. Build the graph first.")

        # Get the vector of first company
        company1_vector_index = self.name_to_index_map[company1]
        company1_vector = self.graph.getrow(company1_vector_index)

        # Get the vector of the second company
        company2_vector_index = self.name_to_index_map[company2]
        company2_vector = self.graph.getrow(company2_vector_index)

        dot_product_score = company1_vector.dot(company2_vector.transpose())[0, 0]

        return dot_product_score

    def get_jaccard_similarity(self, company1, company2):

        """
        Calculate the weighted Jaccard similarity

        Parameters:

            company1 - string; name of the first company

            company2 - string; name of the second company
        """

        if self.graph is None:
            raise ValueError("No graph created. Build the graph first.")

        # Get the vector of first company
        company1_index = self.name_to_index_map[company1]

        # Get the vector of the second company
        company2_index = self.name_to_index_map[company2]
        idf_set = set(self.company_corpus.idf_vector.index)
        idf_map_dict = dict(zip(self.company_corpus.idf_vector.index, self.company_corpus.idf_vector.values))
        company_words_list = list(self.company_corpus.corpus.loc[:, 'rare_words'])

        weighted_jaccard_similarity = self._get_weighted_jaccard_similarity(company_index_1=company1_index,
                                                                            company_index_2=company2_index,
                                                                            company_words_list=company_words_list,
                                                                            idf_set=idf_set,
                                                                            idf_map_dict=idf_map_dict)

        return weighted_jaccard_similarity


def save_graph(graph, filename, location=None):
    """
    Pickle the adjacency matrix  to a file for reuse

    Parameters:

        graph - Scipy sparse matrix; adjacency matrix

        filename - string;

        location (optional) - string; default location is within the library folder
    """

    if location:
        with open(location + '/' + filename, 'wb') as fout:
            pickle.dump(graph, fout)
    else:
        with open(filename, 'wb') as fout:
            pickle.dump(graph, fout)


def load_graph(filename, location=None):
    """
    Load the pickled the adjacency matrix

    Parameters:

        filename - string;

        location (optional) - string; default location is within the library folder
    """

    if location:
        with open(location + '/' + filename, 'rb') as fin:
            data = pickle.load(fin)
            return data
    else:
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)
            return data
