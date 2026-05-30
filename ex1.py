import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load mrpc
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

    # subsampling for debug runs
    if args.max_train_samples != -1:
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        raw_datasets["test"] = raw_datasets["test"].select(range(args.max_predict_samples))

    # setup tokenizer (use base model for training, or specific dir for predict)
    tokenizer_path = args.model_path if (args.do_predict and not args.do_train) else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # prep features for bert
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # dynamic padding via collator
    # note: explicitly setting return_tensors="pt" here bypasses a known formatting bug
    # with torchvision.io in current colab environments.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    if args.do_train:
        # Define the exact naming string requested
        run_name = f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"

        # Create a unique directory for this specific configuration's best model
        save_dir = f"./model_{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Initialize Weights & Biases with the custom name
        wandb.init(
            project="anlp_ex1",
            name=run_name,
            config=vars(args)
        )

        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator)

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        metric = evaluate.load("accuracy")

        print(f"Starting training. Saving best checkpoint to: {save_dir}")
        best_eval_acc = 0.0

        for epoch in range(args.num_train_epochs):
            model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs} [Train]"):
                batch = {k: v.to(device) for k, v in batch.items()}

                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({"train/loss": loss.item()})

            model.eval()
            for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs} [Eval]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                predictions = torch.argmax(outputs.logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            eval_acc = metric.compute()["accuracy"]
            print(f"Epoch {epoch+1} - Validation Accuracy: {eval_acc:.4f}")
            wandb.log({"eval/accuracy": eval_acc, "epoch": epoch+1})

        # Save final trained model
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # append config results for the submission artifact
        with open("res.txt", "a") as f:
            f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_acc:.4f}\n")

        wandb.finish()

    if args.do_predict:
        print(f"Loading model from {args.model_path} for prediction...")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.to(device)
        model.eval()

        # keeping raw dataset reference to extract string sentences for output formatting
        predict_dataset_raw = raw_datasets["test"]
        predict_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size, collate_fn=data_collator)

        all_predictions = []
        for batch in tqdm(predict_dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().tolist())

        # format: Sentence1###Sentence2###Prediction
        with open("predictions.txt", "w", encoding="utf-8") as f:
            for i in range(len(all_predictions)):
                s1 = predict_dataset_raw[i]["sentence1"]
                s2 = predict_dataset_raw[i]["sentence2"]
                pred = all_predictions[i]
                f.write(f"{s1}###{s2}###{pred}\n")


if __name__ == "__main__":
    main()
