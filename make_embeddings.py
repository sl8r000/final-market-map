import click
import pandas as pd
import os
import openai
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import pickle
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import json

def clean_text(text):
    cleaned_text = text.lower()  # Convert to lowercase
    cleaned_text = cleaned_text.encode('ascii', errors='ignore').decode()  # Remove non-ASCII characters
    cleaned_text = cleaned_text.replace("'", "")  # Remove single quotes
    cleaned_text = cleaned_text.replace('"', "")  # Remove double quotes
    return cleaned_text

def get_clean_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request failed

    soup = BeautifulSoup(response.content, 'lxml')
    text = ' '.join(soup.stripped_strings)  # Extract all text and join with spaces
    return clean_text(text)

def standardize_url(url):
    parsed = urlparse(url)

    if not parsed.scheme:
        scheme = 'http'
    else:
        scheme = parsed.scheme

    if not parsed.netloc:
        netloc = parsed.path
    else:
        netloc = parsed.netloc

    standardized_url = f"{scheme}://{netloc}"

    return standardized_url

def set_openai_api_key(secrets_path):
    with open(secrets_path) as stream:
        secrets = json.load(stream)

    openai.api_key = secrets['openai_api_key']

def make_embeddings(startups_df, output_path):
    data = []
    for i, row in startups_df.iterrows():
        desc = row['Description']               # startups_df should have columns for 'Description' and...
        url = standardize_url(row['URL'])       # ... 'URL' ...
        try:
            site_text = get_clean_text_from_url(url)
        except:
            site_text = ''
        text_for_embed = (desc + site_text)[:5000]
        embed = openai.Embedding.create(
        model='text-embedding-ada-002',
        input=text_for_embed
        )
        new_pair = {
            "company": row["Company Name"],     # ... and 'Company Name'
            "text": text_for_embed,
            "embedding": embed['data'][0]['embedding'],
            "url": url
        }
        print(i, new_pair['company'], len(text_for_embed), new_pair['embedding'][:10])
        data.append(new_pair)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

@click.command()
@click.option('--startups', help='file path to startups.csv', default='~/Downloads/startups.csv')
@click.option('--output', help='destination for output embeddings', default='embeddings.pkl')
@click.option('--secrets', help='json file with openai api key', default='secrets.json')
def main(startups, output, secrets):
    startups_df = pd.read_csv(startups)
    set_openai_api_key(secrets)

    make_embeddings(startups_df, output)


if __name__ == '__main__':
    main()

