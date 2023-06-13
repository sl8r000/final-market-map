import pandas as pd
import os
import openai
import click
import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import mplcursors
import pickle

def set_openai_api_key(secrets_path):
    with open(secrets_path) as stream:
        secrets = json.load(stream)

    openai.api_key = secrets['openai_api_key']

def get_startups(file_path):
    with open(file_path, "rb") as f:
        records = pickle.load(f)

    return records

def make_clusters(startups):
    embeddings = np.array([row['embedding'] for row in startups])
    # Define the range of cluster numbers to try
    cluster_range = range(1, 21)  # 1 to 10 clusters

    # Calculate the inertia for different numbers of clusters
    inertias = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # Automatically find the optimal number of clusters using the elbow method
    # (you may need to adjust the threshold depending on your data)
    threshold = 0.1
    optimal_clusters = 1
    for i in range(1, len(inertias) - 1):
        diff1 = inertias[i - 1] - inertias[i]
        diff2 = inertias[i] - inertias[i + 1]
        print(diff2/diff1, optimal_clusters)
        if diff2 / diff1 < threshold:
            optimal_clusters = i + 1
            break

    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    return clusters

def write_output(startups, clusters, output_path):
    embeddings = np.array([row['embedding'] for row in startups])
    reducer = umap.UMAP(random_state=42)
    umap_embedding = reducer.fit_transform(embeddings)

    # this is going to create some plain-text rows that can be copy/pasted
    # into a html file. hey, I didn't say it was pretty
    html_rows = []
    for i in range(len(startups)):
        startups[i]['umap'] = umap_embedding[i]
        row = startups[i]
        html_rows.append(
            f"{{ x: {row['umap'][0]}, y: {row['umap'][1]}, text: \"{row['company']}\", url: \"{row['url']}\", cluster: {clusters[i]} }},"
        )
        print(html_rows[-1])

    with open(output_path, 'w') as the_file:
        for row in html_rows:
            the_file.write(row + '\n')

def plot_output(n_clusters, clusters, startups):
    fig, ax = plt.subplots()

    # Create a scatter plot of the 2D UMAP projection with color-coded clusters
    cmap = plt.cm.get_cmap('hsv', n_clusters)
    x_plot = [x['umap'][0] for x in startups]
    y_plot = [y['umap'][1] for y in startups]
    plt.scatter(x_plot, y_plot, c=clusters, cmap=cmap, s=10)
    plt.colorbar(ticks=range(n_clusters), label='Cluster label')
    plt.title('Text Embedding AI Startups')


    # Add tooltips
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        point_index = sel.target.index
        cluster_label = clusters[point_index]
        company = startups[point_index]['company']
        sel.annotation.set_text(f"Point Index: {point_index}\nCompany: {company}\nCluster: {cluster_label}")

    plt.show()


@click.command()
@click.option('--startups', help='file path to embeddings & annotations from make_embeddings.py', default='embeddings.pkl')
@click.option('--output', help='destination for output text', default='output.txt')
@click.option('--secrets', help='json file with openai api key', default='secrets.json')
def main(startups, output, secrets):
    set_openai_api_key(secrets)

    startups = get_startups(startups)
    clusters = make_clusters(startups)
    write_output(startups, clusters, output)
    plot_output(len(clusters), clusters, startups)


if __name__ == '__main__':
    main()

