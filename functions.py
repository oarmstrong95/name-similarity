# /**********************************************************************************************************
# Import necessary modules
# /**********************************************************************************************************

import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# /**********************************************************************************************************
# Define a function to get embeddings for a list of names and calculate cosine similarity
# /**********************************************************************************************************
def preprocess_name(name, lemmatizer):
    """
    Preprocesses a name by stripping punctuation, converting to lowercase,
    and applying lemmatization.

    Args:
        name (str): The name to preprocess.

    Returns:
        str: The preprocessed name.
    """
    # Strip punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Convert to lowercase
    name = name.lower()
    # Tokenize the name
    tokens = nltk.word_tokenize(name)
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join tokens back into a single string
    preprocessed_name = ' '.join(lemmatized_tokens)
    return preprocessed_name

# Define a function to get embeddings for a list of names
def get_embeddings(names, model):
    """
    Retrieves embeddings for a list of names using a given model.

    Args:
        names (list): A list of names for which embeddings need to be retrieved.
        model: The model used to embed the names.

    Returns:
        numpy.ndarray: An array of embeddings for the given names.
    """
    embeddings = []
    for name in names:
        response = model.embed_query(name)
        embeddings.append(response)

    # Convert list of embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Check if all embeddings have the same dimension
    embedding_dim = embeddings_array.shape[1]
    assert all(embedding.shape[0] == embedding_dim for embedding in embeddings_array), "Embeddings have different dimensions!"

    return embeddings_array

# Define a function to calculate cosine similarity between embeddings
def calculate_similarity(embeddings):
    """
    Calculates the cosine similarity between the given embeddings.

    Parameters:
        embeddings (array-like): An array-like object containing the embeddings.

    Returns:
        array-like: An array-like object containing the cosine similarity scores.
    """
    return cosine_similarity(embeddings)

# Define a function to find similar names based on a similarity threshold
def find_similar_names(names, model, threshold=0.95):
    """
    Finds similar names from a list of names using a given model and threshold.

    Args:
        names (list): A list of names.
        model: The model used to calculate embeddings.
        threshold (float, optional): The similarity threshold. Defaults to 0.9.

    Returns:
        list: A list of tuples containing similar names and their similarity scores.
    """
    embeddings = get_embeddings(names, model)
    similarity_matrix = calculate_similarity(embeddings)
    similar_names = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if similarity_matrix[i][j] > threshold:
                similar_names.append((names[i], names[j], similarity_matrix[i][j]))
    return similar_names

