# README

## Project Overview

This project is designed to identify individuals mentioned in a text and find similar names within the text given the context. The project leverages OpenAI embeddings to generate name embeddings and uses cosine similarity to identify names that are likely to be the same person.

## Features

- **Name Preprocessing**: Strips punctuation, converts names to lowercase, and applies lemmatization.
- **Embedding Generation**: Uses OpenAI's API to generate embeddings for names.
- **Similarity Detection**: Calculates cosine similarity between name embeddings to find similar names.
- **Contextual Analysis**: Identifies names mentioned in a text and finds similar names within the context.


## File Structure
- main.py: Main script to analyze text and find similar names.
- preprocess.py: Contains the preprocess_name function for name preprocessing.
- similarity.py: Contains the find_similar_names function for finding similar names.
- requirements.txt: List of dependencies.
- .env: Environment variables file (not included in the repository).