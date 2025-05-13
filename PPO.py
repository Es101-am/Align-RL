import torch
from tqdm import tqdm
import wandb
import argparse
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Set default reward type (can be overwritten by CLI argument)
default_reward_type = "neutral"

# CLI argument for reward type
parser = argparse.ArgumentParser()
parser.add_argument("--reward_type", type=str, choices=["positive", "negative", "neutral"], default=default_reward_type)
args = parser.parse_args()
reward_type = args.reward_type

def prepare_dataset(cfg, source="imdb", min_len=2, max_len=8):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(source, split="train")
    data = data.rename_columns({"text": "review"})
    data = data.filter(lambda x: len(x["review"]) > 200)

    sample_length = LengthSampler(min_len, max_len)

    def truncate_and_encode(entry):
        input_ids = tokenizer.encode(entry["review"])[:sample_length()]
        entry["input_ids"] = input_ids
        entry["query"] = tokenizer.decode(input_ids)
        return entry

    data = data.map(truncate_and_encode)
    data.set_format(type="torch")
    return data

def custom_collate(batch):
    return {key: [item[key] for item in batch] for key in batch[0]}

if __name__ == "__main__":
    ppo_cfg = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    wandb.init(mode="offline")

    imdb_data = prepare_dataset(ppo_cfg)

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)
    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg.model_name)

    tok = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
    tok.pad_token = tok.eos_token

    trainer = PPOTrainer(ppo_cfg, policy_model, ref_policy, tok, dataset=imdb_data, data_collator=custom_collate)

    device = trainer.accelerator.device
    if trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
    reward_args = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

    min_out_len, max_out_len = 4, 16
    length_selector = LengthSampler(min_out_len, max_out_len)

    gen_config = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tok.eos_token_id,
    }

    for step_idx, sample_batch in tqdm(enumerate(trainer.dataloader)):
        input_prompts = sample_batch["input_ids"]

        responses = []
        for prompt in input_prompts:
            num_tokens = length_selector()
            gen_config["max_new_tokens"] = num_tokens
            full_output = trainer.generate(prompt, **gen_config)
            generated = full_output.squeeze()[-num_tokens:]
            responses.append(generated)

        sample_batch["response"] = [tok.decode(g) for g in responses]

        combined_texts = [q + r for q, r in zip(sample_batch["query"], sample_batch["response"])]
        sentiment_outputs = reward_model(combined_texts, **reward_args)

        rewards = []
        for output in sentiment_outputs:
            pos_score = output[1]["score"]
            neg_score = output[0]["score"]

            if reward_type == "positive":
                reward_val = pos_score
            elif reward_type == "negative":
                reward_val = neg_score
            elif reward_type == "neutral":
                reward_val = 1.0 - abs(pos_score - neg_score)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

            rewards.append(torch.tensor(reward_val))

        metrics = trainer.step(input_prompts, responses, rewards)
        trainer.log_stats(metrics, sample_batch, rewards)

    policy_model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    tok.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
