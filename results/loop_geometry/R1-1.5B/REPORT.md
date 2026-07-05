# E9.0 loop-geometry report

Shards: 884

## Probe (E9.0a)

| layer | token AUC (grouped OOF) | chain AUC | chain p | probe↔meandiff cos |
|--:|--:|--:|--:|--:|
| 15 | 0.994 | 1.000 | 8.4e-146 | 0.159 |
| 16 | 0.994 | 1.000 | 8.4e-146 | 0.165 |
| 17 | 0.994 | 1.000 | 8.4e-146 | 0.177 |

## Loop-direction vs steering vectors (H-B gate: |cos| ≳ 0.3)

| behaviour | layer | arm | cos (probe) | z | cos (meandiff) |
|---|--:|---|--:|--:|--:|
| backtracking | 17 | single | +0.056 | 2.2 | +0.312 |
| backtracking | 17 | manifold_k3 | +0.048 | 1.9 | +0.376 |
| backtracking | 17 | manifold_k5 | +0.057 | 2.2 | +0.336 |
| uncertainty-estimation | 15 | single | +0.043 | 1.7 | +0.411 |
| uncertainty-estimation | 15 | manifold_k3 | +0.073 | 2.9 | +0.571 |
| uncertainty-estimation | 15 | manifold_k5 | +0.046 | 1.8 | +0.436 |
| example-testing | 15 | single | -0.000 | 0.0 | -0.203 |
| example-testing | 15 | manifold_k3 | -0.034 | 1.3 | -0.308 |
| example-testing | 15 | manifold_k5 | -0.016 | 0.6 | -0.250 |
| adding-knowledge | 17 | single | -0.082 | 3.2 | -0.621 |
| adding-knowledge | 17 | manifold_k3 | -0.132 | 5.2 | -0.851 |
| adding-knowledge | 17 | manifold_k5 | -0.112 | 4.4 | -0.769 |

## State-collapse precedence (E9.0b)

| layer | metric | Δ loop (pre−base) | Δ clean | Wilcoxon p | MW p (loop vs clean) |
|--:|---|--:|--:|--:|--:|
| 15 | pr | -2.948 | -4.762 | 3.602423071114324e-36 | 0.999896332819512 |
| 15 | unif | 0.006 | 0.006 | 0.007224600197607497 | 0.4507516150044101 |
| 16 | pr | -2.784 | -4.546 | 1.3924997327455061e-35 | 0.999893041401643 |
| 16 | unif | 0.014 | 0.005 | 3.118976200682624e-11 | 0.04291425633830919 |
| 17 | pr | -3.153 | -4.570 | 1.8636650843865403e-39 | 0.9992961694806572 |
| 17 | unif | 0.037 | 0.018 | 3.590716452224849e-41 | 0.0008177194070790187 |
