
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

models = ["bert-large-uncased"]

def get_sciBERT_embeddings(terms, model):
    # Load SciBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)

    # Tokenize and obtain embeddings for each term
    embeddings = []
    for term in terms:
        # Tokenize the term and convert it to a tensor
        input_ids = tokenizer.encode(term, return_tensors="pt")

        # Obtain the embeddings by passing the input tensor through the model
        with torch.no_grad():
            output = model(input_ids)

        # Extract the embeddings for the [CLS] token
        cls_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(cls_embedding)

    return embeddings

def calculate_cosine_similarity(embeddings):
    # Calculate cosine similarity between all pairs of embeddings
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    return similarity_matrix

# Function to create a spring layout graph from a similarity matrix
def visualize_similarity_matrix(similarity_matrix):
    # Create a graph from the similarity matrix
    G = nx.Graph()

    # Add nodes to the graph
    num_nodes = len(similarity_matrix)
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph based on the similarity matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = similarity_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # Calculate spring layout positions
    pos = nx.spring_layout(G)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)

    # Display edge weights as labels
    edge_labels = {(i, j): f'{G[i][j]["weight"]:.2f}' for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Display the graph
    plt.axis("off")
    plt.show()


# Function to create a spring layout graph from a similarity matrix
def visualize_similarity_matrix_2(similarity_matrix, node_names=None, threshold=0):
    # Create a graph from the similarity matrix
    G = nx.Graph()

    # Add nodes to the graph
    num_nodes = len(similarity_matrix)
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph based on the similarity matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = similarity_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    # Calculate spring layout positions
    pos = nx.spring_layout(G)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)

    # Display node names as labels
    if node_names is not None:
        node_labels = {i: node_names[i] for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Display the graph without edge labels
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Example list of engineering domain terms
    engineering_terms = ["Embedded Systems", "Linux Kernel", "Raspberry Pi", "WebOps", "Backend", "C/C++", "NodeJs", "Nvidia", "GPU", "Apple"]

    # Get SciBERT embeddings for the terms
    for model in models:
        term_embeddings = get_sciBERT_embeddings(engineering_terms, model)

        # Calculate cosine similarity matrix
        similarity_matrix = calculate_cosine_similarity(term_embeddings)

        # Display similarity scores
        # for i in range(len(engineering_terms)):
        #     for j in range(i+1,len(engineering_terms)):
        #             similarity_score = similarity_matrix[i][j]
        #             print(f"Similarity between '{engineering_terms[i]}' and '{engineering_terms[j]}': {similarity_score:.4f}")
        visualize_similarity_matrix_2(similarity_matrix, engineering_terms, threshold=np.mean(similarity_matrix))

