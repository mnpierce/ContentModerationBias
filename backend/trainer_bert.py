import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import evaluate

# --- Configuration ---
MODEL_CHECKPOINT = 'bert-base-uncased' 
MAX_LEN = 128 

# Adjust based on machine resources
TRAIN_BATCH_SIZE = 256 
EVAL_BATCH_SIZE = 512 

LEARNING_RATE = 2e-5 
WEIGHT_DECAY = 0.01 
EPOCHS = 2 
OUTPUT_DIR = "./results_trainer_with_test" 
INPUT_CSV = "csv_data/trimmed_train.csv" 
OUTPUT_PREDICTIONS_CSV = "evaluation_predictions_test_trainer.csv"
FP16 = torch.cuda.is_available()

# Identity columns for final output
IDENTITY_COLUMNS = ["asian","atheist","bisexual","black",
                    "buddhist","christian","female","heterosexual",
                    "hindu","homosexual_gay_or_lesbian","intellectual_or_learning_disability","jewish",
                    "latino","male","muslim","other_disability",
                    "other_gender","other_race_or_ethnicity","other_religion","other_sexual_orientation",
                    "physical_disability","psychiatric_or_mental_illness","transgender","white"]

# --- Helper Functions ---

# Tokenization function
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
def tokenize_function(examples):
    return tokenizer(examples["comment_text"], max_length=MAX_LEN, truncation=True)

# Metrics
accuracy_metric = evaluate.load("accuracy")
auc_metric = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate Accuracy
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    # Calculate AUC
    auc = float('nan') # Default
    
    # Softmax for toxic class
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1].numpy()
    auc = auc_metric.compute(references=labels, prediction_scores=probs)["roc_auc"]

    return {
        "accuracy": acc,
        "roc_auc": auc
    }

# Main
if __name__ == "__main__":
    # 1. Load and Prepare Data
    print("Loading and preparing data...")
    df = pd.read_csv(INPUT_CSV) 
    # Ensure text is string
    df['comment_text'] = df['comment_text'].astype(str)
    df['binary_target'] = (df['target'] >= 0.5).astype(int)
    # Rename target column to 'labels' for Trainer
    df = df.rename(columns={'binary_target': 'labels'})

    print(f"Dataset shape: {df.shape}") 
    print(f"Number of toxic comments: {df['labels'].sum()}") 
    print(f"Number of non-toxic comments: {len(df) - df['labels'].sum()}") 

    # Keep required columns
    df_subset = df[['comment_text', 'labels', 'target'] + IDENTITY_COLUMNS].reset_index(drop=False).rename(columns={'index': 'original_index'})

    # 2. Split Data (Train/Validation/Test)
    print("Splitting data into Train, Validation, and Test sets...")
    # First split into Train (80%) and Temp (20%)
    train_df, temp_df = train_test_split(
        df_subset,
        test_size=0.2, # validation + test
        random_state=42,
        stratify=df_subset['labels']
    )
    # Split Temp (20%) into Validation (10%) and Test (10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['labels']
    )

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # 3. Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df) 

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset       
    })
    print("Dataset splits created:")
    print(dataset_dict)

    # 4. Initialize Tokenizer and Tokenize Data
    print("Tokenizing data...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # Remove columns not needed by the model
    tokenized_datasets = tokenized_datasets.remove_columns(["comment_text", "target", "original_index"] + IDENTITY_COLUMNS)
    print("Tokenized datasets prepared:")
    print(tokenized_datasets)

    # 5. Initialize Model
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2
    ) 

    # 6. Define Training Arguments
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS, 
        learning_rate=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY, 
        per_device_train_batch_size=TRAIN_BATCH_SIZE, 
        per_device_eval_batch_size=EVAL_BATCH_SIZE, 
        # --- Core Training Settings ---
        logging_dir=f"{OUTPUT_DIR}/logs", 
        logging_strategy="steps", 
        logging_steps=100, 
        eval_strategy="epoch", 
        save_strategy="epoch",   
        save_total_limit=2,         
        # Load the best model found during training
        load_best_model_at_end=True,
        # Use AUC to determine the best model
        metric_for_best_model="roc_auc",
        greater_is_better=True, 
        # --- Performance Optimizations ---
        fp16=FP16, # Enable mixed precision training if CUDA available
        # Adjust based on machine resources
        dataloader_num_workers=16, 
        dataloader_pin_memory=True, 
        # Visualization
        report_to="tensorboard",
    )

    # 7. Initialize Trainer
    print("Initializing Trainer...")
    # Data collator handles dynamic padding within each batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        # Validation set for eval during training
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ) 

    # 8. Train the Model
    print("Starting model training...")
    train_result = trainer.train() 

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 9. Evaluate the Best Model on Validation Set
    print("Evaluating the best model on the validation set...")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"]) #
    print("--- Validation Set Performance (Best Model) ---")
    print(eval_metrics)
    trainer.log_metrics("eval_validation", eval_metrics) 
    trainer.save_metrics("eval_validation", eval_metrics)

    # 10. Make Predictions on the **TEST SET** using the Best Model
    print("Making predictions on the TEST set...")
    predictions_output_test = trainer.predict(tokenized_datasets["test"])

    # Extract test results
    logits_test = predictions_output_test.predictions
    final_preds_test = np.argmax(logits_test, axis=-1)
    final_probs_test = torch.softmax(torch.tensor(logits_test, dtype=torch.float32), dim=1)[:, 1].numpy()
    true_labels_test = predictions_output_test.label_ids

    # Print final classification report for the **TEST SET**
    print("\n--- FINAL TEST SET PERFORMANCE (Best Model) ---")
    print(f"Accuracy: {accuracy_score(true_labels_test, final_preds_test):.4f}")
    print(f"AUC: {roc_auc_score(true_labels_test, final_probs_test):.4f}")
    print(classification_report(true_labels_test, final_preds_test, target_names=["non-toxic", "toxic"]))

    # 11. Prepare and Save Final Output CSV based on **TEST SET**
    print(f"Saving TEST set predictions and analysis data to {OUTPUT_PREDICTIONS_CSV}...")
    # Get the original TEST data using the indices stored in the test_dataset
    original_test_data = test_df.copy()

    # Add prediction results from the TEST set
    original_test_data['predicted_binary'] = final_preds_test
    original_test_data['predicted_decimal'] = final_probs_test

    # Rename columns for clarity
    original_test_data = original_test_data.rename(columns={
        'labels': 'human_binary',
        'target': 'human_decimal'
        })

    # Select and order columns for the final CSV
    output_columns = [
        'comment_text', 'human_binary', 'predicted_binary',
        'human_decimal', 'predicted_decimal'
    ] + IDENTITY_COLUMNS
    final_results_df = original_test_data[output_columns]

    final_results_df.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print("Done.")