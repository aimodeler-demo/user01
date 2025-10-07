print(past_responses)

print(memory_context)

print(combined_context)



print(messages)



import faiss
import numpy as np

def store_vectors(embeddings, chunk_filenames, index_path="vector_store.index", mapping_path="chunk_filenames.npy"):
    if not embeddings:
        return  # Early exit if embeddings are empty

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, index_path)
    np.save(mapping_path, np.array(chunk_filenames, dtype=object))
    print("PDFs were indexed successfully!")

store_vectors(embeddings, chunk_filenames)


import faiss
import numpy as np

MODEL="text-embedding-ada-002"

def knowledge_base(query):
    index = faiss.read_index("vector_store.index")
    chunk_filenames = np.load("chunk_filenames.npy", allow_pickle=True)

    # Convert query to embedding
    query_response = openai.Embedding.create(input=query, model=MODEL)
    query_embedding = query_response['data'][0]['embedding']

    # Retrieve closest match
    D, I = index.search(np.array([query_embedding]), k=3)

    if I[0][0] < len(chunk_filenames):  # Ensure valid index
        best_match_index = I[0][0]
        best_match_filename = chunk_filenames[best_match_index]
        print(f"Best matching document: {best_match_filename}")
    else:
        print("No relevant document found.")
        return None

    # Load and return relevant content from the best matching document
    with fitz.open(f"documents/{best_match_filename}") as doc:
        text = "\n".join(page.get_text() for page in doc)

    return text[:100]  # Return the first 100 characters