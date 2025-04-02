import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np # Still potentially useful, though pandas plot handles positioning

# --- Configuration ---
BASE_DIR = Path("surprisal_exps/10m")
SUMMARY_FILENAME = "summary_results.json"
VALUE_TO_EXTRACT = "rsquared" # Key within the 'regression' dictionary
METRIC_NAME = "R-squared" # For plot labels
OUTPUT_FILENAME = "10m_rsquared_plot.png"
PLOT_DPI = 300 # Dots Per Inch - common value for high quality, increase if needed

# --- Data Extraction ---
all_results = []

# Iterate through potential model directories inside BASE_DIR
for model_dir in BASE_DIR.iterdir():
    if model_dir.is_dir():
        # Extract a cleaner model name for the legend
        model_name_raw = model_dir.name
        if model_name_raw.startswith("fixed_win_"):
             model_name_clean = f"Fixed Window {model_name_raw.split('_')[-1]}"
        else:
             model_name_clean = model_name_raw.replace('_', ' ').title()

        summary_file = model_dir / SUMMARY_FILENAME

        if summary_file.exists() and summary_file.is_file():
            print(f"Processing: {summary_file}")
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)

                # Ensure data is a list
                if not isinstance(data, list):
                    print(f"  Warning: Expected a list in {summary_file}, got {type(data)}. Skipping.")
                    continue

                # Extract data for each corpus within the file
                for corpus_data in data:
                    try:
                        corpus_name = corpus_data.get("corpus")
                        regression_data = corpus_data.get("regression")

                        if corpus_name and regression_data:
                            r_squared = regression_data.get(VALUE_TO_EXTRACT)
                            if r_squared is not None:
                                all_results.append({
                                    "model": model_name_clean, # Use the cleaned name
                                    "corpus": corpus_name,
                                    METRIC_NAME: r_squared
                                })
                            else:
                                print(f"  Warning: '{VALUE_TO_EXTRACT}' not found in regression data for corpus '{corpus_name}' in {summary_file}")
                        else:
                             print(f"  Warning: Missing 'corpus' or 'regression' key in an entry within {summary_file}")

                    except Exception as e:
                        print(f"  Error processing entry in {summary_file}: {e}")
                        print(f"  Problematic entry: {corpus_data}")


            except json.JSONDecodeError:
                print(f"  Error: Could not decode JSON from {summary_file}")
            except Exception as e:
                print(f"  Error reading or processing {summary_file}: {e}")
        else:
            print(f"Skipping: {summary_file} (not found or not a file)")

# --- Data Preparation ---
# --- Data Preparation ---
if not all_results:
    print("Error: No results were extracted. Cannot create plot.")
else:
    df = pd.DataFrame(all_results)
    print("\nExtracted Data Head:")
    print(df.head())

    # Pivot the data for easy plotting: Corpora as index, Models as columns
    try:
        df_pivot = df.pivot_table(index='corpus', columns='model', values=METRIC_NAME)

        # --- Corrected Model Ordering Logic ---
        all_original_models = df_pivot.columns.tolist()

        # Separate fixed window models from others
        fixed_models = [m for m in all_original_models if m.startswith("Fixed Window")]
        non_fixed_models = [m for m in all_original_models if not m.startswith("Fixed Window")]

        # Sort fixed window models numerically by window size
        sorted_fixed_models = sorted(fixed_models, key=lambda x: int(x.split()[-1]))

        # Sort other models alphabetically
        sorted_non_fixed_models = sorted(non_fixed_models)

        # Combine the sorted lists for the final order
        final_model_order = sorted_fixed_models + sorted_non_fixed_models
        # --- End Corrected Logic ---


        # Sort columns (models) by the final calculated order and index (corpora) alphabetically
        # Ensure only existing columns are selected in the final order list before applying
        final_model_order_existing = [m for m in final_model_order if m in df_pivot.columns]
        df_pivot = df_pivot.sort_index(axis=0)[final_model_order_existing] # Apply custom model order

        print("\nPivoted Data for Plotting:")
        print(df_pivot)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(15, 8)) # Create figure and axes objects
        df_pivot.plot(kind='bar', ax=ax, width=0.8) # Plot on the created axes

        ax.set_title(f'{METRIC_NAME} by Corpus and Model Type (10m Experiments)', fontsize=16)
        ax.set_xlabel("Corpus", fontsize=12)
        ax.set_ylabel(METRIC_NAME, fontsize=12)
        ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels directly on the axes
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor") # Ensure proper alignment of rotated labels

        # Calculate dynamic y-limits with a bit of padding
        min_val = df_pivot.min().min()
        max_val = df_pivot.max().max()
        # Define padding based on the range, but handle cases where min/max are very close
        data_range = max_val - min_val
        padding = data_range * 0.05 if data_range > 1e-6 else 0.01 # Add 5% or a small fixed amount

        ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.01, max_val + padding)) # Ensure ylim is reasonable for R^2

        ax.legend(title='Model Type', bbox_to_anchor=(1.02, 1), loc='upper left') # Move legend outside plot
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout BEFORE saving for best results with bbox_inches='tight'
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust right margin slightly more for legend

        # --- Saving the Plot ---
        try:
            plt.savefig(
                OUTPUT_FILENAME,
                dpi=PLOT_DPI,
                bbox_inches='tight', # Crucial to include labels/legend properly
                pad_inches=0.1 # Add slight padding around the tight box
            )
            print(f"\nPlot successfully saved to: {OUTPUT_FILENAME}")
        except Exception as e:
            print(f"\nError saving plot: {e}")

        plt.close(fig) # Close the figure to free memory (good practice)


    except Exception as e:
        print(f"\nError during data pivoting or plotting: {e}")
        print("Check the structure of the extracted data.")
        print(df)