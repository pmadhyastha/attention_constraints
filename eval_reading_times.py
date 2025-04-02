import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
import re
import numpy as np
import os
import pickle
import json # Import json for saving/loading
from tqdm import tqdm
from scipy.stats import pearsonr
import statsmodels.api as sm

from typing import Optional, Tuple
from safetensors.torch import load_file, save_file



# Constants / Hyperparameters (previously identified)
decay_rate = 82.85603928544775
alpha = 0.3659550432333628

# --- Custom Model Classes ---

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



def get_surprisal(prompt: str, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(word_list: list, tokenizer, token_separator: str, cased: bool = True):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
    plainseq = " ".join(word_list)
    b = [re.sub(token_separator, "", item) for item in tokenizer.tokenize(plainseq)]
    c = []
    if cased:
        for element in word_list:
            temp_list = []
            while "".join(temp_list) != element:
                temp_list.append(b.pop(0))
            c.append(temp_list)
    else:
        for element in word_list:
            temp_list = []
            while "".join(temp_list) != element.lower():
                temp_list.append(b.pop(0))
            c.append(temp_list)
    return c

def get_surprisal_tokens(word_tokens: list, model, tokenizer, token_separator: str, cased: bool = True):
    s = get_surprisal(" ".join(word_tokens), tokenizer, model)
    toks = tok_maker(word_tokens, tokenizer, token_separator, cased)
    theindex = 0
    out = []
    for index, word in enumerate(toks[1:]):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out.append(surp)
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out.append(sum(surp))
    return out    

# --- Main Execution Logic ---

def main(model_path: str, data_file: str, output_dir: str):
    """
    Loads data, computes surprisal using the custom model specified by args,
    performs regression analysis, and saves results to JSON files.
    """
    print("--- Starting Analysis ---")
    print(f"Model Path/Name: {model_path}")
    print(f"Data File: {data_file}")
    print(f"Output Directory: {output_dir}")

    # --- Setup ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output folder: '{output_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print(f"Loading input data from: '{data_file}'")
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input data file not found at '{data_file}'")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Load Tokenizer and Model ---
    print(f"Loading tokenizer from: '{model_path}'")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set PAD token to EOS token.")
    except Exception as e:
        print(f"Error loading tokenizer from '{model_path}': {e}")
        return

    print(f"Loading custom model based on: '{model_path}'")
    try:
        config = GPT2Config.from_pretrained(model_path)
        # Instantiate custom model, passing hyperparameters
        custom_model = CustomGPT2Model.from_pretrained(
            model_path
        )
        custom_model.to(device)
        custom_model.eval()
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return


    all_results = [] # To store results from all datasets
    datasets = ['meco', 'provo', 'ucl', 'ucl_spr', 'brown', 'natstor'] # Or get from data keys
    token_separator = "Ġ" # Specific to GPT-2 tokenizer

    print("\n--- Processing Datasets ---")
    for dataname in datasets:
        print(f"\nProcessing dataset: {dataname.upper()}")

        if dataname not in data:
            print(f"Warning: Data for '{dataname}' not found in '{data_file}'. Skipping.")
            continue

        dataset_info = data[dataname]
        sentences = dataset_info["sent"]
        freq = dataset_info["freq"]
        length = dataset_info["len"]
        fp = dataset_info["fp"]

        # --- Calculate or Load Surprisal ---
        # Include hyperparams in filename for clarity if they change often
        # model_name_safe = model_path.replace('/','_') # Make path safe for filename
        # surprisal_filename = os.path.join(output_dir, f'{dataname}_{model_name_safe}_custom_surprisal_dr{decay_rate:.2f}_a{alpha:.2f}.json')
        # Simpler filename for now:
        model_name_part = os.path.basename(model_path) # Get last part of path/name
        surprisal_filename = os.path.join(output_dir, f'{dataname}_{model_name_part}_surprisal.json')


        s_custom = None
        print("Computing surprisal values...")
        s_custom = []
        for sent_words in tqdm(sentences, desc=f"Surprisal {dataname}", unit="sent"):
            word_surprisals = get_surprisal_tokens(
                sent_words, custom_model, tokenizer, token_separator, cased=True
            )
            if word_surprisals: # Check if list is not empty
                s_custom.extend(word_surprisals)
            # else: # Already prints warnings inside the function if needed
            #     print(f"Warning: Empty surprisal list for sentence: {' '.join(sent_words)}.")

        print(f"Saving computed surprisal to: {surprisal_filename}")
        try:
            with open(surprisal_filename, 'w') as f:
                json.dump(s_custom, f) # Save compact JSON
        except Exception as e:
            print(f"Error saving surprisal file '{surprisal_filename}': {e}")

        # --- Perform Analysis ---
        print("Performing regression analysis...")

        num_words_in_surprisal = len(s_custom)
        if num_words_in_surprisal == 0:
            print("Error: No valid surprisal values calculated/loaded. Skipping analysis.")
            all_results.append({"corpus": dataname, "model": model_path, "error": "No surprisal data"})
            continue

        # Check and handle alignment (IMPORTANT!)
        original_lengths = {"fp": len(fp), "len": len(length), "freq": len(freq)}
        min_len = min(num_words_in_surprisal, len(fp), len(length), len(freq))

        if min_len != num_words_in_surprisal or min_len != original_lengths["fp"]:
            print(f"CRITICAL WARNING: Length mismatch. Surprisal: {num_words_in_surprisal}, FP: {original_lengths['fp']}, Len: {original_lengths['len']}, Freq: {original_lengths['freq']}.")
            print("This likely means fp/len/freq include first words, while surprisal doesn't.")
            print(f"Trimming all lists to minimum consistent length: {min_len}")
            s_custom = s_custom[:min_len]
            fp = fp[:min_len]
            length = length[:min_len]
            freq = freq[:min_len]
            if min_len == 0:
                print("Error: Trimming resulted in zero data points. Skipping analysis.")
                all_results.append({"corpus": dataname, "model": model_path, "error": "Zero data points after alignment"})
                continue


        # Prepare data for regression
        X = np.column_stack((length, freq, s_custom))
        X = sm.add_constant(X, has_constant='add')
        y = np.array(fp)

        # Fit OLS model
        try:
            model_fit = sm.OLS(y, X).fit()

            # Calculate correlations
            corr_fp_s_custom, p_fp_s = pearsonr(fp, s_custom)
            corr_freq_s_custom, p_freq_s = pearsonr(freq, s_custom)
            corr_len_s_custom, p_len_s = pearsonr(length, s_custom)

            dataset_results = {
                "corpus": dataname,
                "model_path": model_path,
                "decay_rate": decay_rate,
                "alpha": alpha,
                "num_observations": model_fit.nobs,
                "pearson_r": { # Store correlations in a sub-dict
                    "surprisal_vs_fp": corr_fp_s_custom,
                    "surprisal_vs_freq": corr_freq_s_custom,
                    "surprisal_vs_length": corr_len_s_custom,
                },
                "regression": { # Store regression stats in a sub-dict
                    "rsquared": model_fit.rsquared,
                    "rsquared_adj": model_fit.rsquared_adj,
                    "aic": model_fit.aic,
                    "bic": model_fit.bic,
                    "loglikelihood": model_fit.llf,
                    "f_statistic": model_fit.fvalue,
                    "f_pvalue": model_fit.f_pvalue,
                    # Optionally add coefficients and p-values
                    "params": dict(zip(model_fit.params.index, model_fit.params.values)),
                    "pvalues": dict(zip(model_fit.pvalues.index, model_fit.pvalues.values)),
                }
            }
            all_results.append(dataset_results)
            print(f"Analysis complete for {dataname}. AIC: {model_fit.aic:.2f}, R2: {model_fit.rsquared:.4f}")

        except Exception as e:
            print(f"Error during regression analysis for {dataname}: {e}")
            all_results.append({
                "corpus": dataname,
                "model_path": model_path,
                "decay_rate": decay_rate,
                "alpha": alpha,
                "error": f"Regression failed: {e}"
            })

    # --- Save Final Results ---
    summary_filename = os.path.join(output_dir, 'summary_results.json')
    print(f"\nSaving summary of all results to: {summary_filename}")
    try:
        with open(summary_filename, 'w') as f:
            json.dump(all_results, f, indent=4)
    except Exception as e:
        print(f"Error saving summary results file '{summary_filename}': {e}")

    print("\n--- Analysis Finished ---")
    return all_results


# --- Argument Parsing and Execution ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute word surprisal using a custom GPT-2 model and analyze its relation to linguistic variables.")

    parser.add_argument(
        "--model_path",
        type=str,
        help=f"Path or Hugging Face name of the GPT-2 model/tokenizer"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help=f"Path to the input data pickle file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Directory to save output JSON files"
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    results = main(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
    )

    print("\n--- Execution Summary ---")
    for res in results:
        if 'error' in res:
            print(f"Dataset: {res['corpus']}, Error: {res['error']}")
        elif 'regression' in res:
            print(f"Dataset: {res['corpus']}, N: {res.get('num_observations', 'N/A')}, AIC: {res['regression'].get('aic', 'N/A'):.2f}, R2: {res['regression'].get('rsquared', 'N/A'):.4f}")
        else:
            print(f"Dataset: {res['corpus']}, Status: Unknown (no regression results or error)")