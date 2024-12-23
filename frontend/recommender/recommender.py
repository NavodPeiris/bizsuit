import pandas as pd
import numpy as np
import time
import duckdb
import networkx as nx
from pecanpy import pecanpy
from gensim.models import Word2Vec

from tqdm import tqdm
from joblib import load, dump
from umap import UMAP
import plotly.express as px
from joblib import Parallel, delayed
import faiss


def train_recommender(df):
    try:
        # optimizing data
        # Convert all float32 fields to float16
        df = df.astype({col: 'float16' for col in df.select_dtypes('float32').columns})

        # Convert all float64 fields to float16
        df = df.astype({col: 'float16' for col in df.select_dtypes('float64').columns})

        # Convert all int64 fields to int32
        df = df.astype({col: 'int32' for col in df.select_dtypes('int64').columns})

        # Now df has the required dtypes
        print(df.dtypes)

        df.to_parquet('./recommender/Data/optimised_raw_data.parquet',index=False)

        con = duckdb.connect(database=':memory:', read_only=False)

        print("creating product_view_tbl table...")
        con.execute('''
        CREATE TABLE product_view_tbl AS
        SELECT 
        user_id,
        product_id, 
        CAST(event_time[:-4] as DATETIME) as event_time, 
        user_session
        FROM './recommender/Data/optimised_raw_data.parquet'
        WHERE user_session in (SELECT 
        user_session
        FROM './recommender/Data/optimised_raw_data.parquet'
        WHERE event_type = 'view'
        AND user_session IS NOT NULL
        AND product_id IS NOT NULL
        GROUP BY user_session
        HAVING count(distinct product_id) > 1)
        ORDER BY user_session, CAST(event_time[:-4] as DATETIME)
        ''').df()

        print("creating product_views_graph table...")
        con.execute("""CREATE TABLE product_views_graph AS select product_id, 
        LEAD(product_id, 1, -1) OVER (PARTITION BY user_session ORDER BY event_time) as next_viewed_product_id,
        user_session,
        event_time
        from product_view_tbl
        """).df()

        print("constructing undirected weighted graph...")
        undirected_weighted_graph = con.execute("""
        SELECT CASE
                WHEN product_id > next_viewed_product_id THEN product_id
                ELSE next_viewed_product_id
            END AS pid_1,
            CASE
                WHEN product_id < next_viewed_product_id THEN product_id
                ELSE next_viewed_product_id
            END AS pid_2,
            COUNT(*) AS occurence_ct
        FROM product_views_graph
        WHERE next_viewed_product_id<>-1
        AND product_id IS NOT NULL
        AND product_id != next_viewed_product_id
        GROUP BY 1,
                2
        """).df()

        undirected_weighted_graph.to_parquet('./recommender/Data/ConstructedGraph/undirected_weighted_product_views_graph.parquet',index=False)

        # deepwalk
        GRAPH_FILE_NAME = 'undirected_weighted_product_views_graph.parquet'
        WEIGHTED = True
        DIRECTED = False
        NUM_WALK = 10
        WALK_LEN = 50

        print(GRAPH_FILE_NAME, NUM_WALK, WALK_LEN)

        graph_path = './recommender/Data/ConstructedGraph/{}'.format(GRAPH_FILE_NAME)
        graph = pd.read_parquet(graph_path)

        print("Graph Read from {}".format(graph_path))

        Node_Count = len(set(graph['pid_1'].unique()).union(graph['pid_2'].unique()))
        Edge_Count = len(graph)

        print("Graph {} has Nodes: {} | Edges: {}".format(GRAPH_FILE_NAME, Node_Count, Edge_Count))

        edg_graph_path = './recommender/Data/Edg_Graphs_DataFile/'+GRAPH_FILE_NAME.split('.')[0]+'.edg'
        graph.to_csv(edg_graph_path, sep='\t', index=False, header=False)
        print("Edg Graph Saved")

        # graph random walk generation
        g = pecanpy.SparseOTF(p=1, q=0.5, workers=-1, verbose=True, extend=True)

        g.read_edg(edg_graph_path, weighted=WEIGHTED, directed=DIRECTED)
        walks = g.simulate_walks(num_walks=NUM_WALK, walk_length=WALK_LEN)

        model = Word2Vec(walks,  # previously generated walks
                        hs=1,  # tells the model to use hierarchical softmax
                        sg = 1,  # tells the model to use skip-gram
                        vector_size=128,  # size of the embedding
                        window=5,
                        min_count=1,
                        workers=4,
                        seed=42)

        model.save('./recommender/Model/node2vec_graph_embedding_'+GRAPH_FILE_NAME.split('.')[0]+'.model')

        pid_set = set(graph['pid_1'].unique()).union(graph['pid_2'].unique())
        payload = []
        for pid in tqdm(pid_set):
            try:
                payload.append({'pid': pid, 'embedding_vector': model.wv[str(pid)]})
            except:
                print(pid, "Not Exist")
                pass

        # created embeddings for pids
        embedding_df = pd.DataFrame(payload)
        embedding_df.to_parquet('./recommender/Data/Embedding_Data/node2vec_embedding_df_'+GRAPH_FILE_NAME.split('.')[0]+'.parquet', index=False)

        return "Successfully Trained Model"

    except Exception as err:
        print(f"Error : ${err}")
        return f"Error: ${err}"
    

def generate_recommendations(product_id):

    GRAPH_FILE_NAME = 'undirected_weighted_product_views_graph.parquet'

    # vector search with FAISS
    df_node2vec = pd.read_parquet('./recommender/Data/Embedding_Data/node2vec_embedding_df_{}.parquet'.format(GRAPH_FILE_NAME.split('.')[0]))
    df_node2vec.columns = ['product_id', 'embedding_vector']

    df_full = pd.read_parquet('./recommender/Data/optimised_raw_data.parquet').drop_duplicates(subset=['product_id'])
    df_full = df_full[['product_id', 'category_code']]

    df_node2vec = df_node2vec.merge(df_full, how='left', on='product_id')

    xb = np.array(df_node2vec.embedding_vector.tolist())
    
    # make faiss available
    index = faiss.IndexFlatL2(128)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)

    k = 10                          # we want to see 10 nearest neighbors
    D, I = index.search(xb, k)

    # Get the index corresponding to the product_id
    product_index = df_node2vec[df_node2vec['product_id'].astype(str) == product_id].index

    # If you need the index as an integer (assuming the product exists):
    if not product_index.empty:
        product_index = product_index[0]
    else:
        print("Product ID not found.")
        return []
        
    print(f"product index for product_id {product_id} is {product_index}")

    # similar product reccomendation for index of product_id
    similar_products = df_node2vec[df_node2vec.index.isin(I[product_index])]

    # Drop the 'embedding_vector' column
    similar_products = similar_products.drop(columns=['embedding_vector'])

    # Reset the index (or remove it) to avoid carrying over the old index
    similar_products = similar_products.reset_index(drop=True)

    print(similar_products)

    return similar_products