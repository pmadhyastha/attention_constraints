from typing import Optional, Tuple
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
import torch.nn as nn
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import argparse
import math

# Constants for exponential attention decay
decay_rate = 82.85603928544775
alpha = 0.3659550432333628

class CustomGPT2Attention(GPT2Attention):
    """
    Custom GPT2 Attention module with different attention mechanisms.

    This module extends the standard GPT2Attention to include exponential decay,
    fixed window, learnable window, and primacy-recency attention mechanisms.

    Args:
        config (GPT2Config): Model configuration.
        layer_idx (int, optional): Layer index. Defaults to None.
        attention_type (str, optional): Type of attention mechanism to use.
            Options are 'exponential', 'fixed_window', 'learnable_window', 'primacy_recency'.
            Defaults to 'exponential'.
        window_size (int, optional): Window size for fixed and learnable window attention. Defaults to 7.
    """
    def __init__(self, config, layer_idx=None, attention_type='exponential', window_size=7):
        super().__init__(config, layer_idx=layer_idx)
        self.attention_type = attention_type

        if attention_type == 'learnable_window':
            # Learnable window size parameter, initialized and constrained to be at least 5.
            self.window_size = nn.Parameter(torch.tensor(float(max(5, window_size))), requires_grad=True)
        elif attention_type == 'primacy_recency':
            # Learnable weights for primacy and recency components.
            self.primacy_weight = nn.Parameter(torch.tensor(0.5))
            self.recency_weight = nn.Parameter(torch.tensor(0.5))
        else:
            # Fixed window size, registered as a buffer (not learnable).
            self.register_buffer('window_size', torch.tensor(float(window_size)))

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Splits the last dimension into (num_heads, attn_head_size)."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape).permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merges the attention heads back to original shape."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        Calculates attention weights based on the specified attention type.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            head_mask (torch.Tensor, optional): Head mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention output and attention weights.
        """
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        seq_len = query.size(-2)
        device = query.device

        if self.attention_type == 'exponential':
            # Exponential decay attention mechanism.
            indices = torch.arange(seq_len, device=device)
            exponential_decay_bias = torch.exp(-torch.abs(indices[None, :] - indices[:, None]) * decay_rate)
            attn_weights = (1 - alpha) * attn_weights + alpha * exponential_decay_bias
        elif self.attention_type == 'fixed_window':
            # Fixed window attention mechanism.
            window_size = int(self.window_size.item())
            causal_mask = torch.ones(seq_len, seq_len, device=device).triu(diagonal=1).bool()
            window_mask = torch.zeros(seq_len, seq_len, device=device).bool()
            for i in range(seq_len):
                start_index = max(0, i - (window_size- 1) )
                window_mask[i, :start_index] = True
            combined_mask = causal_mask | window_mask
            attn_weights = attn_weights.masked_fill(combined_mask, float('-inf'))
        elif self.attention_type == 'learnable_window':
            # Learnable window attention mechanism using Gaussian kernel.
            window_size_squared = torch.clamp(self.window_size ** 2, min=1.0)
            seq_positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            pos_diff = seq_positions.unsqueeze(1) - seq_positions.unsqueeze(0)
            epsilon = 1e-6
            attention_bias = -(pos_diff ** 2) / (2 * window_size_squared + epsilon)
            attn_weights = attn_weights + attention_bias.unsqueeze(0).unsqueeze(0)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        elif self.attention_type == 'primacy_recency':
            # Primacy-Recency based attention mechanism.
            primacy_indices = torch.arange(seq_len, device=device).float() / seq_len
            primacy_weights = torch.exp(-primacy_indices)
            primacy_weights = primacy_weights / torch.sum(primacy_weights)

            recency_indices = torch.flip(primacy_indices, dims=[0])
            recency_weights = torch.exp(-recency_indices)
            recency_weights = recency_weights / torch.sum(recency_weights)

            combined_weights = self.primacy_weight * primacy_weights + self.recency_weight * recency_weights
            combined_weights = combined_weights.unsqueeze(0).repeat(seq_len, 1)

            attn_weights = attn_weights + combined_weights

        if self.scale_attn_by_inverse_layer_idx and self.layer_idx is not None:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self,
                hidden_states: Optional[Tuple[torch.Tensor]],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Implements the forward pass for the custom attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            layer_past (Tuple[torch.Tensor], optional): Past key/value states for caching. Defaults to None.
            attention_mask (torch.FloatTensor, optional): Attention mask. Defaults to None.
            head_mask (torch.FloatTensor, optional): Head mask. Defaults to None.
            encoder_hidden_states (torch.FloatTensor, optional): Encoder hidden states. Defaults to None.
            encoder_attention_mask (torch.FloatTensor, optional): Encoder attention mask. Defaults to None.
            past_key_value (Tuple[Tuple[torch.Tensor]], optional): Past key/value states (legacy). Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.
            use_cache (bool, optional): Whether to use cache for optimization. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]: Attention output and optional past/present states and attention weights.
        """
        query_key_value = self.c_attn(hidden_states)
        query, key, value = query_key_value.split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CustomGPT2Block(GPT2Block):
    """
    Custom GPT2 Block that uses CustomGPT2Attention.

    Replaces the standard GPT2Attention with CustomGPT2Attention to incorporate
    different attention mechanisms.

    Args:
        config (GPT2Config): Model configuration.
        layer_idx (int, optional): Layer index. Defaults to None.
        attention_type (str, optional): Type of attention mechanism to use. Defaults to 'exponential'.
        window_size (int, optional): Window size for window-based attention types. Defaults to 7.
    """
    def __init__(self, config, layer_idx=None, attention_type='exponential', window_size=7):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomGPT2Attention(config, layer_idx=layer_idx, attention_type=attention_type, window_size=window_size)

class CustomGPT2Model(GPT2LMHeadModel):
    """
    Custom GPT2 Language Model with pluggable attention types.

    Extends GPT2LMHeadModel to allow different attention mechanisms to be used
    within the transformer blocks.

    Args:
        config (GPT2Config): Model configuration.
        attention_type (str, optional): Type of attention mechanism to use. Defaults to 'exponential'.
        window_size (int, optional): Window size for window-based attention types. Defaults to 7.
    """
    def __init__(self, config, attention_type='exponential', window_size=7):
        super().__init__(config)
        self.attention_type = attention_type
        self.window_size = window_size

        # Replace standard GPT2 blocks with custom blocks
        self.transformer.h = nn.ModuleList([
            CustomGPT2Block(config, layer_idx=i, attention_type=attention_type, window_size=window_size)
            for i in range(config.n_layer)
        ])

        # Reinitialize LM head and tie weights
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights()

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the pretrained model to a specified directory, including custom configuration.
        """
        super().save_pretrained(save_directory, **kwargs)
        config_dict = {
            "attention_type": self.attention_type,
            "window_size": self.window_size
        }
        with open(os.path.join(save_directory, "custom_config.json"), "w") as f:
            import json
            json.dump(config_dict, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Loads a pretrained model from a specified directory, including custom configuration.
        """
        import json
        custom_config_path = os.path.join(pretrained_model_name_or_path, "custom_config.json")
        custom_config = {}
        if os.path.exists(custom_config_path):
            with open(custom_config_path, "r") as f:
                custom_config = json.load(f)

        attention_type = custom_config.get("attention_type", kwargs.pop("attention_type", "exponential"))
        window_size = custom_config.get("window_size", kwargs.pop("window_size", 7))

        config = GPT2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config, attention_type=attention_type, window_size=window_size)
        model = model._load_state_dict_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return model

    def _load_state_dict_from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        pytorch_bin_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")

        if os.path.exists(pytorch_bin_path):
            print(f"Loading model weights from existing pytorch_model.bin: {pytorch_bin_path}")
            state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        elif os.path.exists(safetensors_path):
            print(f"Converting model weights from safetensors to pytorch_model.bin...")
            state_dict = load_file(safetensors_path, device="cpu") # Load from safetensors
            torch.save(state_dict, pytorch_bin_path) # Save as pytorch_model.bin
            print(f"Successfully converted and saved to pytorch_model.bin: {pytorch_bin_path}")
        else:
            raise FileNotFoundError(f"Neither pytorch_model.bin nor model.safetensors found in {pretrained_model_name_or_path}")

        self.load_state_dict(state_dict, strict=False)
        return self

    def print_learned_parameters(self):
        """Prints the learned parameters specific to the custom attention mechanisms."""
        print("\n" + "="*50)
        print(f"LEARNED PARAMETERS FOR {self.attention_type.upper()} ATTENTION")
        print("="*50)
        for i, layer in enumerate(self.transformer.h):
            if self.attention_type == 'learnable_window':
                window_size = layer.attn.window_size.item()
                print(f"Layer {i}: Window Size = {window_size:.4f} (Rounded: {int(round(abs(window_size)))})")
            elif self.attention_type == 'primacy_recency':
                primacy = layer.attn.primacy_weight.item()
                recency = layer.attn.recency_weight.item()
                print(f"Layer {i}: Primacy Weight = {primacy:.4f}, Recency Weight = {recency:.4f}")
            elif self.attention_type == 'fixed_window':
                window_size = layer.attn.window_size.item()
                print(f"Layer {i}: Fixed Window Size = {window_size:.4f}")
            elif self.attention_type == 'exponential':
                print(f"Layer {i}: Using global decay_rate = {decay_rate:.4f}, alpha = {alpha:.4f}")
        print("="*50)


# Dataset and DataLoader
class TextDataset(Dataset):
    """
    Dataset for text processing.

    Reads text from a file, tokenizes it using the provided tokenizer, and
    creates fixed-size blocks of tokens.

    Args:
        tokenizer (GPT2Tokenizer): Tokenizer to use.
        file_path (str): Path to the text file.
        block_size (int): Size of token blocks.
    """
    def __init__(self, tokenizer, file_path, block_size):
        assert os.path.isfile(file_path), f"File not found: {file_path}"

        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text) # Tokenize the entire text at once
        self.examples = []

        # Create non-overlapping blocks of tokens
        for i in range(0, len(tokens) - block_size + 1, block_size):
            self.examples.append(torch.tensor(tokens[i:i + block_size], dtype=torch.long))

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, i):
        """Returns the example at index i."""
        return self.examples[i]


def train(model, tokenizer, train_file, block_size=1024, batch_size=4, epochs=3, lr=5e-5, device='cuda', save_dir="trained_model"):
    """
    Trains the GPT-2 model on the given text file.

    Args:
        model (PreTrainedModel): Model to train.
        tokenizer (GPT2Tokenizer): Tokenizer.
        train_file (str): Path to the training text file.
        block_size (int, optional): Maximum sequence length. Defaults to 1024.
        batch_size (int, optional): Batch size. Defaults to 4.
        epochs (int, optional): Number of training epochs. Defaults to 3.
        lr (float, optional): Learning rate. Defaults to 5e-5.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        save_dir (str, optional): Directory to save the trained model. Defaults to "trained_model".

    Returns:
        PreTrainedModel: Trained model.
    """
    model.to(device)
    model.train()

    dataset = TextDataset(tokenizer, train_file, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Parameter groups for optimizer: separate learning rate for window_size if learnable
    params = []
    for name, p in model.named_parameters():
        if "window_size" in name:
            params.append(p)

    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if "window_size" not in n], 'weight_decay': 0.01},
        {'params': params, 'lr': lr * 1}  # Potentially higher LR for window_size
    ], lr=lr)

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    # Save trained model and tokenizer
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Training complete. Model saved to '{save_dir}' directory.")
    return model


def load_model(model_path, device='cuda'):
    """
    Loads a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model directory.
        device (str, optional): Device to load the model to ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Tuple[PreTrainedModel, GPT2Tokenizer]: Loaded model and tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = CustomGPT2Model.from_pretrained(model_path) # Loads custom model
    model.to(device)
    model.print_learned_parameters() # Display learned parameters for custom attention

    return model, tokenizer


if __name__ == '__main__':
    # Argument parsing setup
    parser = argparse.ArgumentParser(description='Train or load a GPT-2 model with custom attention.')
    parser.add_argument('--train_file', type=str, help='Path to the training text file.')
    parser.add_argument('--block_size', type=int, default=512, help='Maximum sequence length (block size).')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the saved tokenizer directory (e.g., pretrained GPT-2 tokenizer).')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--save_dir', type=str, default='trained_model', help='Directory to save the trained model.')
    parser.add_argument('--model_type', type=str, default='custom', choices=['custom', 'vanilla'], help='Type of model to use (custom or vanilla GPT2).')
    parser.add_argument('--window_size', type=int, default=7, help='Window size for windowed attention types.')
    parser.add_argument('--attention_type', type=str, default='exponential', choices=['exponential', 'fixed_window', 'learnable_window', 'primacy_recency'], help='Type of attention mechanism to use.')
    parser.add_argument('--load_model', type=str, help='Path to a trained model to load instead of training.')

    args = parser.parse_args()

    # Model loading or training logic
    if args.load_model:
        model, tokenizer = load_model(args.load_model, device=args.device)
        print(f"Model loaded from {args.load_model}")
    else:
        # Ensure required arguments for training are provided
        if not args.train_file or not args.tokenizer_path:
            parser.error("--train_file and --tokenizer_path are required for training.")

        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token is None: # Set padding token if not already set
            tokenizer.pad_token = tokenizer.eos_token

        config = GPT2Config.from_pretrained('gpt2')

        if args.model_type == 'custom':
            model = CustomGPT2Model(config, attention_type=args.attention_type, window_size=args.window_size)
        else:
            model = GPT2LMHeadModel(config) # Use vanilla GPT2 if specified

        print("\n--- Model Structure Inspection ---")
        for i, layer in enumerate(model.transformer.h):
            print(f"Layer {i}: Attention Type = {type(layer.attn)}") # Verify attention type in each layer
        print("--- End Model Structure Inspection ---\n")

        trained_model = train(model, tokenizer, args.train_file, args.block_size, args.batch_size, args.epochs, args.lr, args.device, args.save_dir)
        print("Training finished.")