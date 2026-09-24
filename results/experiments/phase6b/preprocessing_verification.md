# Fold-local preprocessing verification

`build_feature_pair` fits each vectorizer and chronological creator transformer only on the fold-training rows. The later interval calls `transform` only. Word2Vec/IDF transformer tests likewise verify that transform does not alter fitted vocabulary or IDF state. Sampling is then applied only to the completed training matrix, so removed or duplicated rows cannot change creator-history semantics.
