# Align-RL
PPO Sentiment Fine-Tuning on IMDB Reviews

This project demonstrates how to fine-tune a language model using Proximal Policy Optimization (PPO) with the IMDB sentiment dataset, leveraging different reward strategies. The goal is to train a language model that generates responses aligned with a desired sentiment: positive, negative, or neutral.

🧠 Training Usage

```bash
python train.py --reward_type [positive|negative|neutral]
```

Examples:

```bash
python train.py --reward_type positive   # Learn to generate positive reviews
python train.py --reward_type neutral    # Learn to generate more balanced reviews
```

---

## 📝 Reward Strategies Explained

### 1. Positive

```python
reward = p_pos
```

Maximize the model’s confidence in generating positively classified text.

### 2. Negative

```python
reward = p_neg
```

Encourage the model to generate text with more negative sentiment.

### 3. Neutral

```python
reward = 1 - abs(p_pos - p_neg)
```

Reward neutrality by minimizing the gap between positive and negative scores.

---

## 🧪 Evaluation

While this repo focuses on training, you can later evaluate generated samples using:

```python
pipeline("sentiment-analysis")(text)
```

---

## 📁 Output

After training, the fine-tuned model and tokenizer will be saved to:

```
gpt2-imdb-pos-v2/
```

Use `from_pretrained()` to load them later.

---

## 📚 Acknowledgments

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
* [lvwerra/gpt2-imdb](https://huggingface.co/lvwerra/gpt2-imdb)

---

## 📬 License

This project is open-sourced under the MIT License.
