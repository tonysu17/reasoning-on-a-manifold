# Chain quality report: `data/chains_R1-1.5B.json`

Generated: 2026-05-27T11:26:24
Total chains: 1000
Max-tokens setting: 8192

## 1. Structural integrity

| Check | Count |
|---|---|
| missing fields | 0 (OK) |
| empty chain | 0 (OK) |
| empty prompt | 0 (OK) |
| has error field | 0 (OK) |
| zero tokens | 0 (OK) |
| dup task ids | 0 (OK) |
| full text mismatch | 0 (OK) |

## 2. Category distribution

| Category | Count | Delta vs expected |
|---|---|---|
| mathematical_logic | 100 |  |
| spatial_reasoning | 100 |  |
| verbal_logic | 100 |  |
| pattern_recognition | 100 |  |
| causal_reasoning | 100 |  |
| probabilistic_thinking | 100 |  |
| systems_thinking | 100 |  |
| creative_problem_solving | 100 |  |
| scientific_reasoning | 100 |  |
| lateral_thinking | 100 |  |

## 3. Token-length distribution per category

| Category | mean | median | min | max | % at max |
|---|---|---|---|---|---|
| causal_reasoning | 2837 | 1428 | 730 | 8192 | 22.0% |
| creative_problem_solving | 4397 | 1696 | 890 | 8192 | 44.0% |
| lateral_thinking | 7856 | 8192 | 603 | 8192 | 95.0% |
| mathematical_logic | 5685 | 8192 | 1043 | 8192 | 55.0% |
| pattern_recognition | 6252 | 8192 | 574 | 8192 | 66.0% |
| probabilistic_thinking | 6320 | 8192 | 818 | 8192 | 59.0% |
| scientific_reasoning | 3008 | 1432 | 766 | 8192 | 24.0% |
| spatial_reasoning | 6906 | 8192 | 535 | 8192 | 71.0% |
| systems_thinking | 1630 | 1309 | 824 | 8192 | 4.0% |
| verbal_logic | 5630 | 8192 | 654 | 8192 | 62.0% |
| **OVERALL** | **5052** | **8192** | **535** | **8192** | **50.2%** |

## 4. Truncation analysis

- Chains at max_tokens: **502 (50.2%)**
- Chains with closing `</think>`: 501 (50.1%)
- Chains *without* closing `</think>`: **499** (likely truncated)

Cross-tabulation:

| Status | Count |
|---|---|
| Hit max AND no closing think (truncated mid-thinking) | 499 |
| Hit max AND has closing think (ended right at limit) | 3 |
| Short AND no closing think (terminated some other way) | 0 |
| Short AND has closing think (clean finish) | 498 |

## 5. Prompt integrity

- Prompt length: mean=596, median=582, range=[241, 1257]
- Distinct template headers: 496
- Prompts containing `<think>` tag: 1000 of 1000

## 6. Content anomalies

| Check | Count |
|---|---|
| non ascii in first 1000 | 222 |
| contains cjk | 5 |
| contains kana | 0 |
| repetition loops | 3 |

## 7. Token-count histogram

```
      <1k (  64) #####
     1-2k ( 304) ########################
     2-3k (  67) #####
     3-4k (  23) #
     4-5k (  19) #
     5-6k (  10) 
     6-7k (   7) 
     7-8k (   3) 
   8-8.2k (   1) 
     8192 ( 502) ########################################
```

## 8. Sample chain openings (random seed=42)

### `SYST_054`  (systems_thinking, n_tokens=1310)

**Instruction:** A social media platform changes its algorithm to prioritize 'meaningful interactions' over 'passive consumption.' Six months later: average session time decreases by 25%, user-reported well-being incr...

**Chain start:** Okay, so I'm trying to understand this situation where a social media platform is changing its algorithm to prioritize 'meaningful interactions' over 'passive consumption.' There are several negative effects observed after six months: session time decreases by 25%, well-being increases by 10%, adver...

### `SPAT_016`  (spatial_reasoning, n_tokens=8192)

**Instruction:** Imagine a rectangular room that is 12 meters long, 9 meters wide, and 6 meters high. A spider is located at the center of one of the 12m × 9m walls, 1 meter below the ceiling. A fly is located at the ...

**Chain start:** Okay, so I have this problem where there's a spider and a fly in a rectangular room. The room is 12 meters long, 9 meters wide, and 6 meters high. The spider is on one of the 12m by 9m walls, right in the center, but 1 meter below the ceiling. The fly is on the opposite wall, also in the center, but...

### `MATH_026`  (mathematical_logic, n_tokens=8192)

**Instruction:** Let P(x) be a predicate on the natural numbers. Consider these two statements:
A: ∀n ∈ ℕ, P(n) → P(n+1)
B: ∃m ∈ ℕ such that P(m) and ∀k ∈ ℕ, (k < m → ¬P(k))

If both A and B are true, what can you con...

**Chain start:** Okay, so I have this problem where I need to figure out what we can conclude about the set of natural numbers for which a predicate P(n) holds true, given that two statements A and B are both true. Let me try to break this down step by step. /  / First, let me restate the problem to make sure I understa...
