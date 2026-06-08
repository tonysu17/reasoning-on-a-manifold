# Base-model verification - DeepSeek-R1-Distill-Qwen-1.5B

Distilled model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

## Aggregate ranking (lower = closer base)

| Rank | Candidate | HF ID | Aggregate delta |
|------|-----------|-------|------------------|
| 1 | `qwen-math` | `Qwen/Qwen2.5-Math-1.5B` | 0.098263 |
| 2 | `qwen-math-instruct` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | 0.191928 |
| 3 | `qwen-base` | `Qwen/Qwen2.5-1.5B` | 1.127415 |

## Top-level cosine similarity

| Candidate | embed_tokens cos | lm_head cos |
|-----------|------------------|-------------|
| `qwen-math-instruct` | 0.839095 | 0.845637 |
| `qwen-math` | 0.993611 | 0.972564 |
| `qwen-base` | 0.232458 | 0.218574 |

## Verdict

**Likely:** `qwen-math` (`Qwen/Qwen2.5-Math-1.5B`) is the base; margin over runner-up is 0.0937.