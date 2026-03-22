import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path="Medicare_Advanced/data/medicare_5000.csv"):
    df = pd.read_csv(path, encoding='latin1')
    return df

def build_graph(df):
    """
    Constructs a PyG Data object from the dataframe.
    Nodes: Claims
    Edges: Connect claims that share the same Provider_ID.
    Node Features: Numerical columns + Encoded Categorical.
    """
    print("Building Graph...")
    
    # 1. Encode Provider_ID to find shared edges
    le_provider = LabelEncoder()
    df['Provider_Idx'] = le_provider.fit_transform(df['Provider_ID'])
    
    # 2. Construct Edges (Homogeneous: Claim <-> Claim via Provider)
    # This can be expensive if a provider has many claims.
    # Approach: For each provider, get list of claim indices. Create exhaustive pairs.
    # To save memory, we can use a "Virtual Provider Node" approach or just connect them sequentially?
    # Let's try full connection for now, but limit if too large.
    
    edge_index = []
    
    # Group by Provider
    provider_groups = df.groupby('Provider_Idx')
    
    print(f"Processing {len(provider_groups)} providers for edge generation...")
    for pid, group in provider_groups:
        indices = group.index.values
        if len(indices) < 2:
            continue
        
        # If group is too large (e.g. > 100), we might want to limit edges
        # For < 5000 total rows, it's probably okay.
        # Create all pairs (mesh)
        # torch.combinations might be useful, or just numpy
        
        # Create mesh grid of indices
        # a, b = np.meshgrid(indices, indices)
        # pairs = np.vstack([a.flatten(), b.flatten()])
        # edge_index.append(pairs)
        
        # Optimization: Just connect each node to the "center" of the group? No, that's virtual node.
        # Optimization: Connect in a circle? 1->2->3->1. Too sparse.
        # Let's do full mesh but assume groups aren't huge.
        
        import itertools
        pairs = list(itertools.permutations(indices, 2)) # Directed edges both ways
        edge_index.extend(pairs)
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"Graph has {edge_index.shape[1]} edges.")

    # 3. Node Features
    # Numerical
    num_cols = ['Claim_Amount', 'Number_of_Procedures', 'Days_Admitted', 'Patient_Age']
    scaler = StandardScaler()
    x_num = scaler.fit_transform(df[num_cols].fillna(0))
    
    # Categorical (simple label encoding for now, ideally embeddings)
    cat_cols = ['Procedure_Code', 'Diagnosis_Code', 'Patient_Gender', 'Hospital_Type', 'Insurance_Type']
    x_cat = []
    for col in cat_cols:
        le = LabelEncoder()
        x_cat.append(le.fit_transform(df[col].astype(str)))
    
    x_cat = np.stack(x_cat, axis=1)
    
    # Combined Features
    x = np.hstack([x_num, x_cat])
    x = torch.tensor(x, dtype=torch.float) # GNN usually takes float features
    
    # Labels
    y = torch.tensor(df['Fraud'].values, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = len(df)
    
    return data, df

if __name__ == "__main__":
    df = load_data()
    data, _ = build_graph(df)
    print(data)
    torch.save(data, "Medicare_Advanced/data/graph_data.pt")
    print("Graph saved to Medicare_Advanced/data/graph_data.pt")
