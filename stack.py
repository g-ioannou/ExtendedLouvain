import networkx as nx
import pandas as pd
from pyvis.network import Network

# creating test data
node_data = {
    "id": [1, 2, 3, 4],
    "Name": ["Me", "Batman", "Superman", "IronMan"],
    "Company": ["Marvel", "DC", "DC", "Marvel"],
}
edge_data = {"source": [1, 1, 2], "target": [4, 2, 3]}

nodes = pd.DataFrame(node_data)
edges = pd.DataFrame(edge_data)

# getting a group id for each company
groups = nodes.groupby("Company")["id"].apply(list).reset_index()
groups["group"] = groups.index

# finding group id for each node from groups dataframe
nodes = nodes.merge(groups, how="inner", on=["Company"])
nodes["title"] = nodes[["Name", "Company"]].apply(lambda x: f"Name:{x[0]} , Company:{x[1]}", axis=1)

nodes = nodes.drop("id_y", axis=1).set_index("id_x")

# collecting node attributes for network x
node_attrs = nodes.to_dict("index")

# creating a network x graph from dataframes
graph = nx.from_pandas_edgelist(edge_data)
nx.set_node_attributes(graph, node_attrs)


pyvis_nt = Network("500px", "500px", heading="Graph")




pyvis_nt.set_options(
    """
    var options = {
        "configure":{
            "enabled":true
        },
        "interaction": {
    "tooltipDelay": 450
            },
        "edges":{
            "smooth":false
        },
        "physics": {
            "hierarchicalRepulsion": {
            "centralGravity": 0,
            "springConstant": 0,
            "nodeDistance": 20,
            "damping": 0.5
            },
            "maxVelocity": 0,
            "minVelocity": 0.01,
            "solver": "hierarchicalRepulsion"
        }
        }"""
)



pyvis_nt.from_nx(graph)


pyvis_nt.show('test_graph.html')