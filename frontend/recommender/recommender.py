import pandas as pd
import numpy as np
import duckdb
import networkx as nx

from gensim.models import Word2Vec
from node2vec import Node2Vec

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
            COUNT(*) AS occurrence_ct
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

        edge_list = [[x[0], x[1], x[2]] for x in graph[['pid_1', 'pid_1', 'occurrence_ct']].to_numpy()]

        G = nx.Graph()
        G.add_weighted_edges_from(edge_list)

        # Initialize Node2Vec with the graph
        node2vec = Node2Vec(G, dimensions=64, walk_length=WALK_LEN, num_walks=NUM_WALK, p=2, q=1, workers=1)

        # Train the Node2Vec model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Save the model
        model.save('./recommender/Model/graph_embeddings.model')

        return "Successfully Trained Model"

    except Exception as err:
        print(f"Error : ${err}")
        return f"Error: ${err}"
    

def generate_recommendations(product_id):

    # Load the saved Node2Vec model
    model = Word2Vec.load('./recommender/Model/graph_embeddings.model')

    df = pd.read_parquet("./recommender/Data/optimised_raw_data.parquet")

    similar_products = []
    # Find and print the most similar tokens
    for similar_product in model.wv.most_similar(product_id)[:5]:
        prod_id = similar_product[0]
        # Get the `category_code` of the `prod_id`
        category_code = df.loc[df['product_id'] == int(prod_id), 'category_code'].values
        similar_products.append([prod_id,category_code[0]])
        print(similar_product)

    recommendations = pd.DataFrame(similar_products, columns=("Product Id", "Category Code"))

    return recommendations