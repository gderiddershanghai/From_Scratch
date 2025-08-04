from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768, # size of embedding vector
        intermediate_size=3072, # size of linear layer
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init()
        self.congig = config
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.Layernorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel values [batch size, channels, h, w] -> [batch size, num patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
    
    
class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init()
        self.congig = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values) ->Tuple:
        # [batch size, channels, h, w] -> [batch size, num patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)