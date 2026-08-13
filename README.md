The experiments showed that both Top-K and chunk size affect retrieval quality. Top-K=3 and chunk size=300 are currently considered working candidates for further testing. However, they are not final optimal values because some queries still returned incomplete information. Increasing chunk size or Top-K alone did not solve all retrieval issues.

Current Working Configuration
Embedding Model : text-embedding-3-small
Top-K           : 3
Chunk Size      : 300
Chunk Overlap   : 30

Note: These are the current working candidate values based on the initial experiments, not final production values.