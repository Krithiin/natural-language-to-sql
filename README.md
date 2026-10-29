# Natural Language to SQL Generation

## About
This project converts questions written in plain English into SQL queries using a database schema.

## Technologies Used
- Python
- Hugging Face Transformers
- FLAN-T5-small
- LoRA fine-tuning
- Sentence Transformers
- Streamlit

## Dataset
The project uses the `b-mc2/sql-create-context` dataset. A sample of 1,000 examples was used, divided into training, validation, and test sets.

## Method
1. Load and prepare the dataset.
2. Generate embeddings for semantic search.
3. Fine-tune FLAN-T5-small using LoRA.
4. Compare the base model with the fine-tuned model.
5. Save predictions and display generated SQL through a simple interface.

## Files
- `Natural_Language_to_SQL_Simple.ipynb` — project notebook
- `app.py` — Streamlit interface
- `sql_lora_adapter/` — saved fine-tuned adapter
- `project_results.json` — training and evaluation results
- `sql_model_comparison.csv` — model comparison
- `sql_test_predictions.csv` — test predictions

## Running the Interface
Install the required libraries and run:

```bash
streamlit run app.py
```

Review generated SQL before using it on a real database.
