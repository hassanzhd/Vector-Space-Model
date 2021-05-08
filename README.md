# VECTOR SPACE MODEL

The following assignment implements "Vector Space Model" which supports querying of document using terms and a threshold value alpha.

Queries are done on the basis of cosine similarity between tf-idf of query and document vectors.

- tf-idf (product of term frequeny and inverse document frequency)
  - tf-idf = tf \* idf
- tf (term frequency): Number of times term appeared in document
- idf (inverse document frequency) - ratio of total documents and document frequency
  - idf = N / Df<sub>term</sub>
- cosine similarity (how relevant the query and document are to one another)
  - sim(Document, Query) = (Document \* Query) / | Document | \* | Query |

#### DATA SET:

Data set provided comprised of 50 short stories each of which had to be parsed, normalized in order to form an index containing termFrequencies, tf-idf per document and inverseDocumentFrequency per word

#### Query Format:

(space seperated query) alphaValue

#### [Sample queries](http://k180208-ir-assignment2.azurewebsites.net/static/index.txt)

#### [Deployed model on Azure App Services](http://k180208-ir-assignment2.azurewebsites.net)
