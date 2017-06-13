import unittest
import numpy as np
import pandas as pd
from companysim import CompanyCorpus, CompanyGraph, save_graph, load_graph


class TestCompanyCorpus(unittest.TestCase):

    # ======== Start of tests for CompanyCorpus class

    # Test that the _build_corpus function correctly assigns internal corpus correctly
    def test_build_corpus_from_ndarray(self):

        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])

        cc = CompanyCorpus(test_input)
        self.assertTrue(isinstance(cc.corpus, pd.DataFrame))

    # Test that _build_corpus correctly checks for the right type
    def test_build_corpus_type_error(self):

        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]

        self.assertRaises(TypeError, CompanyCorpus, test_corpus)

    # Test the option to provide a pre built idf vector at object creation
    def test_idf_vector_pre_built(self):
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])

        test_idf = [np.log((1 + 1) / (1 + 1)), np.log((1 + 2) / (1 + 1)), np.log((1 + 2) / (1 + 1))]
        test_terms = ['provider', 'software', 'hardware']
        test_input_idf = pd.Series(test_idf, index=test_terms)

        cc = CompanyCorpus(test_input, idf=test_input_idf)

        self.assertIsInstance(cc.idf_vector, pd.Series)

    # Test that the build_idf function correctly calculates the idf correctly and sets it to the internal value
    def test_idf_vector_creation(self):
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        desired_output = [np.log((1 + 1) / (1 + 1)), np.log((1 + 2) / (1 + 1)), np.log((1 + 2) / (1 + 1))]

        cc = CompanyCorpus(test_input)
        idf_vec, term_vector = cc.build_idf(description_column_name='description')
        print(cc.idf_vector)
        self.assertTrue(all(val in idf_vec for val in desired_output))

    def test_filter_description_by_idf(self):
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        cc = CompanyCorpus(test_input)

        number_to_remove = 2
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)

        desired_output = [{'software'}, {'hardware'}]

        self.assertTrue(all(word in cc.corpus['rare_words'].values for word in desired_output))

    # ======== Start of tests for CompanyGraph class

    def test_build_lsh_forest(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])

        cc = CompanyCorpus(test_input)

        number_to_remove = 1
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)

        # Create a CompanyGraph instance and test building the LSH forest
        cg = CompanyGraph(cc)

        cg.build_lsh_forest(company_name_column_name='domain')
        self.assertTrue(cg.lsh_forest)

    def test_build_graph(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware'],
                       ['company_3', 'Provider of Software technology']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])

        cc = CompanyCorpus(test_input)

        number_to_remove = 1
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)

        # Create a CompanyGraph instance
        cg = CompanyGraph(cc)

        cg.build_lsh_forest(company_name_column_name='domain')

        cg.build_graph(sensitivity=3)

        self.assertIsNotNone(cg.graph)

    def test_get_dot_product_score(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'business software application'], ['company_2', 'hardware technology'],
                       ['company_3', 'consumer software service'], ['company_4', 'consumer saas application']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        cc = CompanyCorpus(test_input)
        number_to_remove = 0
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)
        # Create a CompanyGraph instance
        cg = CompanyGraph(cc)
        cg.build_lsh_forest(company_name_column_name='domain')
        cg.build_graph(sensitivity=3)

        # Run test of function
        #print(cg.graph.todense())
        dot_product_score = cg.get_dot_product_score('company_1', 'company_3')
        #print(dot_product_score)
        self.assertNotEqual(0., dot_product_score)

    def test_get_jaccard_similarity(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware technology'],
                       ['company_3', 'Provider of Software technology'], ['company_4', 'Provider of software service']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        cc = CompanyCorpus(test_input)
        number_to_remove = 1
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)
        # Create a CompanyGraph instance
        cg = CompanyGraph(cc)
        cg.build_lsh_forest(company_name_column_name='domain')
        cg.build_graph(sensitivity=3)

        # print(cg.graph.todense())
        # Run test of function
        jaccard_similarity = cg.get_jaccard_similarity('company_1', 'company_3')
        self.assertNotEqual(0., jaccard_similarity)


    def test_save_graph(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware technology'],
                       ['company_3', 'Provider of Software technology'], ['company_4', 'Provider of software service']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        cc = CompanyCorpus(test_input)
        number_to_remove = 1
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)
        # Create a CompanyGraph instance
        cg = CompanyGraph(cc)
        cg.build_lsh_forest(company_name_column_name='domain')
        cg.build_graph(sensitivity=3)
        save_graph(cg,
                   filename='graph.pickle')

    def test_load_graph(self):
        # Create a CompanyCorpus instance, and initialize it with some data
        test_corpus = [['company_1', 'Provider of software'], ['company_2', 'Provider of hardware technology'],
                       ['company_3', 'Provider of Software technology'], ['company_4', 'Provider of software service']]
        test_input = pd.DataFrame(test_corpus, columns=['domain', 'description'])
        cc = CompanyCorpus(test_input)
        number_to_remove = 1
        cc.build_idf(description_column_name='description')
        cc.filter_desc_by_idf(description_column_name='description',
                              number_words_to_cut=number_to_remove)
        # Create a CompanyGraph instance
        cg = CompanyGraph(cc)
        cg.build_lsh_forest(company_name_column_name='domain')
        cg.build_graph(sensitivity=3)

        cg_load = load_graph(filename='graph.pickle')

        self.assertTrue(np.array_equal(cg.graph.toarray(), cg_load.graph.toarray()))

if __name__ == '__main__':
    unittest.main()
