import torch
import torch.nn.functional as F
import numpy as np
from graph_builder import load_data, build_graph
from models.tabnet import TabNetWrapper
from models.ft_transformer import FTTransformerWrapper
from models.gnn import GraphSAGE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# --- Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20

def train_tabular_models(df):
    print("\nTraining Tabular Models...")
    
    # Preprocessing for TabNet (needs simple array)
    # We reuse the graph_builder's logic or do it simply here
    # Let's simple encoding
    
    # ... (Reuse preprocessing from graph_builder for consistency)
    data, _ = build_graph(df) # This gets us the tensors directly
    
    X = data.x.numpy()
    y = data.y.numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. TabNet
    print("Training TabNet...")
    tabnet = TabNetWrapper(input_dim=X.shape[1])
    tabnet.train(X_train, y_train, X_test, y_test, max_epochs=EPOCHS)
    tabnet.save("Medicare_Advanced/models/tabnet_model.zip")
    
    # 2. FT-Transformer
    print("Training FT-Transformer...")
    # FT-Transformer needs separate num and cat
    # For now, let's treat all inputs as numerical (since we already label encoded)
    # Ideally we pass cat_cardinalities.
    # To keep it simple for this MVP, we will use the GNN features (which are floats).
    # This defeats the purpose of FT-Transformer slightly but fine for demo.
    
    model = FTTransformerWrapper(n_num_features=X.shape[1], cat_cardinalities=[])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        # Pass empty cat since we treated all as num
        out = model(X_train_t, None) 
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
            
    torch.save(model.state_dict(), "Medicare_Advanced/models/ft_transformer.pt")

def train_gnn(data):
    print("\nTraining GNN...")
    model = GraphSAGE(data.num_features, 64, 2).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(EPOCHS * 2): # GNNs need more epochs usually
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            pred = out.argmax(dim=1)
            acc = float((pred == data.y).sum()) / len(data.y)
            print(f'Epoch {epoch}: Loss {loss.item():.4f}, Acc: {acc:.4f}')
            
    torch.save(model.state_dict(), "Medicare_Advanced/models/gnn.pt")

if __name__ == "__main__":
    os.makedirs("Medicare_Advanced/models", exist_ok=True)
    df = load_data()
    
    # Train Tabular
    # train_tabular_models(df) # Commented out for now to fast track GNN
    
    # Train GNN
    # Re-build data with edges
    data, _ = build_graph(df)
    train_gnn(data)
    
    # Run tabular too
    train_tabular_models(df)
