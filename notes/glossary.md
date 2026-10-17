# 📖 RAG Glossary

Plain-English definitions of the terms used throughout these notes, in alphabetical order. Each entry points to the module that explains it in depth.

### Agentic RAG
A RAG system where the LLM decides *whether*, *what* and *how many times* to retrieve, usually by calling retrieval as a tool in a loop. Useful for multi-step questions; adds cost and latency. → [Module 7](./07_agentic_rag_and_tools.md)

### ANN (Approximate Nearest Neighbour)
Search that finds *almost* the closest vectors far faster than comparing against every vector. HNSW, IVF and PQ are common ANN index types. → [Module 5](./05_vector_databases.md)

### BM25
A classic keyword-ranking formula. It scores documents higher when they contain the query's words often, when those words are rare in the corpus, and penalizes very long documents slightly. Excellent for exact terms such as IDs and error codes. → [Module 6](./06_retrieval_mechanics.md)

### Chunk
A small piece of a document (often a few hundred tokens) that is embedded and retrieved as one unit. → [Module 4](./04_chunking_strategies.md)

### Chunk overlap
Text repeated at the end of one chunk and the start of the next, so ideas that cross a boundary aren't lost. Typically 10–20% of chunk size. → [Module 4](./04_chunking_strategies.md)

### Citation (grounded citation)
A reference in the answer to the source document a claim came from, such as *(Source: 03_chunking.txt)*. Makes answers verifiable. → [Module 3](./03_prompting_techniques.md)

### Context precision
A retrieval metric: are the relevant chunks ranked near the top of the results? → [Module 9](./09_rag_evaluation.md)

### Context recall
A retrieval metric: did retrieval find all the information needed to answer? → [Module 9](./09_rag_evaluation.md)

### Context window
The maximum number of tokens a model can read in one request, including instructions, retrieved chunks, the question and its answer. → [Module 1](./01_foundations_transformers.md)

### Contextual chunking
Prepending a short description (document title, section heading) to each chunk before embedding, so the chunk makes sense on its own. → [Module 4](./04_chunking_strategies.md)

### Cosine similarity
A measure of how closely two vectors point in the same direction: 1 = same direction, 0 = unrelated, −1 = opposite. The default way to compare text embeddings. → [Module 2](./02_embeddings_and_vectors.md)

### Cross-encoder
A model that reads the question and a chunk *together* and outputs one relevance score. More accurate than vector similarity but slower, so it's used for reranking a short list. → [Module 6](./06_retrieval_mechanics.md)

### Dense retrieval
Retrieval by comparing embedding vectors (semantic similarity), as opposed to sparse keyword matching. → [Module 6](./06_retrieval_mechanics.md)

### Embedding
A list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meanings get vectors that are close together. → [Module 2](./02_embeddings_and_vectors.md)

### Faithfulness
A generation metric: the share of claims in an answer that are supported by the retrieved context. Low faithfulness means hallucination. → [Module 9](./09_rag_evaluation.md)

### Fine-tuning
Further training a model's weights on your own examples. Better for teaching behaviour or style than for adding fresh facts. → [Module 0](./00_introduction_to_rag.md)

### Hallucination
A fluent, confident answer that isn't supported by facts. RAG reduces hallucinations by grounding answers in retrieved documents. → [Module 1](./01_foundations_transformers.md)

### Hit rate @k
The fraction of test questions for which a correct chunk appears in the top *k* retrieved results. → [Module 9](./09_rag_evaluation.md)

### HNSW (Hierarchical Navigable Small World)
A graph-based ANN index: search starts at a coarse layer and hops to ever-closer neighbours. Fast and accurate, memory-hungry; the default in many vector databases. → [Module 5](./05_vector_databases.md)

### Hybrid search
Running keyword (BM25) and vector search together and merging the results, usually with Reciprocal Rank Fusion. → [Module 6](./06_retrieval_mechanics.md)

### HyDE (Hypothetical Document Embeddings)
A query-rewriting trick: ask the LLM to write a hypothetical answer, embed that, and search with it. → [Module 6](./06_retrieval_mechanics.md)

### Ingestion
Everything that happens before search: loading, parsing, cleaning, adding metadata, chunking, embedding and storing documents. → [Module 10](./10_document_ingestion_and_parsing.md)

### LLM-as-a-judge
Using a strong LLM with a rubric to score answers, as a scalable stand-in for human review. → [Module 9](./09_rag_evaluation.md)

### Lost in the middle
The tendency of LLMs to use facts at the start and end of a long prompt more reliably than facts buried in the middle. A reason to send fewer, well-ordered chunks. → [Module 1](./01_foundations_transformers.md)

### Metadata filtering
Restricting retrieval to chunks whose metadata matches a condition (date, source, team, permission) before ranking by similarity. → [Module 6](./06_retrieval_mechanics.md)

### MRR (Mean Reciprocal Rank)
The average of `1 / rank` of the first correct result across test questions. Rewards putting the right chunk at the very top. → [Module 9](./09_rag_evaluation.md)

### Multi-query retrieval
Generating several rephrasings of a question with an LLM, searching with each, and merging the results to improve recall. → [Module 6](./06_retrieval_mechanics.md)

### Prompt injection
Text that tries to override the model's instructions. *Indirect* injection hides it inside documents the system retrieves. → [Module 11](./11_production_rag.md)

### RAG (Retrieval-Augmented Generation)
A pattern where relevant documents are retrieved and added to the prompt so the LLM answers from them instead of from memory alone. → [Module 0](./00_introduction_to_rag.md)

### RAGAS
An open-source library for evaluating RAG systems with metrics such as faithfulness, answer relevancy, context precision and context recall. → [Module 9](./09_rag_evaluation.md)

### Reciprocal Rank Fusion (RRF)
A way to merge ranked lists: each result scores `1 / (k + rank)` in every list it appears in, and the scores are summed. Needs no score normalization. → [Module 6](./06_retrieval_mechanics.md)

### Reranking
A second, more precise scoring pass (often a cross-encoder) over the top candidates from fast retrieval, keeping only the best few. → [Module 6](./06_retrieval_mechanics.md)

### Semantic cache
Reusing a previous answer when a new question's embedding is very close to a cached question's. Saves cost and latency; needs a strict threshold. → [Module 11](./11_production_rag.md)

### Sparse retrieval
Keyword-based retrieval (TF-IDF, BM25) using an inverted index from words to documents. → [Module 6](./06_retrieval_mechanics.md)

### Temperature
A setting that controls randomness in generation. Low values (0–0.3) suit factual RAG answers. → [Module 1](./01_foundations_transformers.md)

### Token
The unit models read and bill by: a word, part of a word, or punctuation. In English, about 4 characters on average. → [Module 1](./01_foundations_transformers.md)

### Top-k
How many results retrieval returns. Too low misses facts; too high adds noise, cost and latency. → [Module 6](./06_retrieval_mechanics.md)

### Vector database
A database built to store embeddings with metadata and find the nearest vectors to a query quickly. Examples: Chroma, Pinecone, pgvector, Weaviate. → [Module 5](./05_vector_databases.md)
