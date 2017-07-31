## Synopsis

Companysim is a Python module for generating measures of company similarity based on textual description information. The algorithm was developed for the class CS 341 at Stanford University, in conjunction with the venture capital firm Rocketship.vc


## Process

Given pairs of companies and their textual descriptions, companysim generates a "company graph" to demonstrate similarity between the pairs

Similarity is estimated with two measures: 

1. Weighted Jaccard similarity between each pair of descriptions
	- The weights are set using the inverse document frequency (IDF) value for each word 

2. Dot product between the two vectors of weighted edges from the "company graph"
	- A "company graph" is generated where the edges are weighted by the jaccard similarity
	- Each company then has a vector of weighted edges going to other companies 

These similarity measures can then be used as features in a classification algorithm to predict a similarity score between the pairs of companies

We found using a K-nearest neighbor (KNN) binary classification algorithm with euclidean distance metric worked well

## Code Example

```python
import companysim.companysim as cs

company_info = {'company_domain': ['xyz.com', 'abc.com'],
				'description': ['xyz.com is a developer of business software',
				                'abc.com is a business software application company']}
company_list = pd.DataFrame(company_info)

# Setup the parameters
DESCRIPTION_COLUMN = 'description'
NUMBER_OF_WORDS = 3

cc = cs.CompanyCorpus(company_list)
cc.build_idf(description_column_name=DESCRIPTION_COLUMN)

# Filter descriptions by removing the number of
#    words specified in NUMBER_OF_WORDS
cc.filter_desc_by_idf(description_column_name=DESCRIPTION_COLUMN,
                      number_words_to_cut=NUMBER_OF_WORDS)

# Create a CompanyGraph
cg = cs.CompanyGraph(cc)
cg.build_lsh_forest(company_name_column_name=NAME_COLUMN)
cg.build_graph(sensitivity=NUMBER_SIMILAR_COMPANIES)

# Access similarity measures
company1 = 'xyz.com'
company2 = 'abc.com'
cg.get_dot_product_score(company1, company2)
cg.get_jaccard_similarity(company1, company2)
```

## Installation

The easiest way to install companysim is via pip

```
pip install companysim
```

## Contributors

Connor Lamon

Meeran Ismail

Ke Xu

## License

MIT License
