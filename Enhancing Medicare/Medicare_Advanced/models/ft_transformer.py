import torch
import torch.nn as nn
import rtdl

class FTTransformerWrapper(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_token=192, n_blocks=3, attention_dropout=0.2, ffn_dropout=0.1, attention_heads=8, linear_dropout=0.2):
        super().__init__()
        
        self.tokenizer = rtdl.FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token
        )
        
        self.transformer = rtdl.FTTransformer.make_default(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_blocks=n_blocks,
            d_token=d_token,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            attention_heads=attention_heads,
            linear_dropout=linear_dropout,
            last_layer_query_idx=None, # None = use CLS token or mean pooling? Wait, rtdl checks this.
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=2
        )
        
    def forward(self, x_num, x_cat):
        return self.transformer(x_num, x_cat)
