# Overfitting and Replication Audit

Run ID: `overfit_audit_20260605T232524Z`

## Bottom Line

The independently rebuilt original logic reproduced the prior conservative result, but the audit found validation weaknesses that make the right wording narrower. At 1.5 NQ points of slippage per leg, the replicated original logic produced 2,022 trades, $61,301.12 PnL, win rate 66.57%, profit factor 1.377, and monthly annualized Sharpe 2.848.

The clean-eval rerun avoids using the test fold for LightGBM early stopping. That cleaner test produced $60,511.45 at 1.5 pt/leg. The purged clean-eval rerun produced $56,144.58. The label-shuffle null produced $0.00, which is the key overfitting sanity check.

## What Survived

- The raw NQ data exists locally and is hash-locked in `audit_manifest.json`.
- The original conservative metrics were independently reproduced without importing the original strategy module.
- Direct future-looking feature leakage was not found in the static feature scan.
- The old `sh` value is rejected as a headline Sharpe ratio and retained only as a trade-level t-statistic.
- The 2.5 pt/leg execution stress remains a real failure boundary.

## What Did Not Survive Cleanly

- The original script used the test fold as LightGBM's early-stopping evaluation set, so it is not a perfectly pure out-of-sample process.
- The original pandas slicing shares one boundary timestamp per fold between train and test; the audit adds a purged rerun to remove that issue.
- Same-bar adverse ordering is severe: at 1.5 pt/leg it produced $-175,264.18.
- Monte Carlo permutation of final trade PnL is downgraded to path-order drawdown evidence, not proof of a stable edge.

## Claim Matrix Summary

| verdict   |   count |
|:----------|--------:|
| supported |      19 |
| rejected  |       1 |
| corrected |       1 |

## Leakage Checks

| check                             | verdict   | evidence                                                                                                                                                   |
|:----------------------------------|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| future_shift_in_features          | pass      | No negative shift found in feature assignments.                                                                                                            |
| future_shift_in_labels            | allowed   | Negative shift appears in labels only: 103: f["lv"] = ((h.shift(-1) - l.shift(-1)) > 1.2 * a14).astype(int) ; 104: f["ld"] = (c.shift(-1) > c).astype(int) |
| original_fold_boundary_overlap    | warn      | Original pandas slices share 5 timestamp rows across train/test boundaries.                                                                                |
| test_fold_used_for_early_stopping | warn      | Original LightGBM fit uses eval_set=[(Xte, te[label])] for early stopping; audit adds train-tail clean eval.                                               |
| old_monte_carlo_meaning           | warn      | Original Monte Carlo permutes final trade PnL, preserving total PnL; it is a path-order drawdown test, not proof of edge.                                  |

## Main Metrics

| variant                                    |   slippage_points_per_leg |   trades |   total_pnl |   avg_trade |   win_rate |   profit_factor |   max_trade_order_drawdown |   trade_t_stat_not_sharpe |   monthly_count |   monthly_positive |   monthly_sharpe_ann |   monthly_worst |   daily_active_count |   daily_sharpe_ann_active_days |   daily_worst |
|:-------------------------------------------|--------------------------:|---------:|------------:|------------:|-----------:|----------------:|---------------------------:|--------------------------:|----------------:|-------------------:|---------------------:|----------------:|---------------------:|-------------------------------:|--------------:|
| independent_directional_failure_case       |                       0.5 |     2022 |    -6274.53 |       -3.1  |     0.2839 |           0.975 |                  -14095.2  |                    -0.452 |              16 |                  8 |               -0.481 |        -7036    |                  321 |                         -0.4   |      -2283    |
| independent_straddle_slippage_stress       |                       0.5 |     2022 |   142181    |       70.32 |     0.6993 |           2.035 |                   -2414.33 |                    13.117 |              16 |                 16 |                6.711 |         1890.43 |                  321 |                         10.456 |      -1135.81 |
| independent_straddle_slippage_stress       |                       1   |     2022 |   101741    |       50.32 |     0.6889 |           1.68  |                   -2769.28 |                     9.386 |              16 |                 16 |                4.831 |          389.19 |                  321 |                          7.534 |      -1255.81 |
| independent_straddle_slippage_stress       |                       1.5 |     2022 |    61301.1  |       30.32 |     0.6657 |           1.377 |                   -6288.05 |                     5.655 |              16 |                 13 |                2.848 |        -2570.81 |                  321 |                          4.548 |      -1410.29 |
| independent_straddle_slippage_stress       |                       2   |     2022 |    20861.1  |       10.32 |     0.6316 |           1.118 |                  -19830.1  |                     1.925 |              16 |                  9 |                0.926 |        -5530.81 |                  321 |                          1.543 |      -1590.29 |
| independent_straddle_slippage_stress       |                       2.5 |     2022 |   -19578.9  |       -9.68 |     0.5861 |           0.898 |                  -39627.9  |                    -1.806 |              16 |                  6 |               -0.815 |        -8591.32 |                  321 |                         -1.437 |      -1776.03 |
| clean_eval_straddle_slippage_stress        |                       0.5 |     1925 |   137511    |       71.43 |     0.6992 |           2.058 |                   -2097.96 |                    13.038 |              16 |                 16 |                6.791 |         2292.95 |                  321 |                          9.989 |       -903.1  |
| clean_eval_straddle_slippage_stress        |                       1   |     1925 |    99011.4  |       51.43 |     0.6899 |           1.699 |                   -2519.04 |                     9.388 |              16 |                 16 |                4.89  |          740.19 |                  321 |                          7.236 |       -983.1  |
| clean_eval_straddle_slippage_stress        |                       1.5 |     1925 |    60511.4  |       31.43 |     0.6712 |           1.393 |                   -5378.91 |                     5.737 |              16 |                 14 |                2.914 |        -1819.81 |                  321 |                          4.427 |      -1210.53 |
| clean_eval_straddle_slippage_stress        |                       2   |     1925 |    22011.5  |       11.43 |     0.6416 |           1.132 |                  -15847.7  |                     2.087 |              16 |                  7 |                1.011 |        -4379.81 |                  321 |                          1.604 |      -1450.53 |
| clean_eval_straddle_slippage_stress        |                       2.5 |     1925 |   -16488.5  |       -8.57 |     0.5927 |           0.909 |                  -35063.1  |                    -1.563 |              16 |                  5 |               -0.71  |        -7469.11 |                  321 |                         -1.191 |      -1690.53 |
| purged_clean_eval_straddle_slippage_stress |                       0.5 |     1923 |   133065    |       69.2  |     0.6947 |           2.014 |                   -3017.26 |                    12.729 |              16 |                 16 |                6.912 |         2292.95 |                  320 |                          9.748 |      -1886.93 |
| purged_clean_eval_straddle_slippage_stress |                       1   |     1923 |    94604.6  |       49.2  |     0.6843 |           1.661 |                   -3633.26 |                     9.05  |              16 |                 16 |                4.941 |          296.93 |                  320 |                          6.967 |      -2046.93 |
| purged_clean_eval_straddle_slippage_stress |                       1.5 |     1923 |    56144.6  |       29.2  |     0.6641 |           1.361 |                   -5539.88 |                     5.371 |              16 |                 14 |                2.866 |        -2263.07 |                  320 |                          4.136 |      -2206.93 |
| purged_clean_eval_straddle_slippage_stress |                       2   |     1923 |    17684.6  |        9.2  |     0.6349 |           1.105 |                  -16136.6  |                     1.692 |              16 |                  8 |                0.861 |        -5117.41 |                  320 |                          1.297 |      -2366.93 |
| purged_clean_eval_straddle_slippage_stress |                       2.5 |     1923 |   -20775.4  |      -10.8  |     0.5939 |           0.887 |                  -37822.5  |                    -1.987 |              16 |                  5 |               -0.947 |        -8197.41 |                  320 |                         -1.511 |      -2526.93 |
| label_shuffle_straddle_slippage_stress     |                       0.5 |        0 |        0    |        0    |     0      |         nan     |                       0    |                   nan     |               0 |                  0 |              nan     |          nan    |                    0 |                        nan     |        nan    |
| label_shuffle_straddle_slippage_stress     |                       1   |        0 |        0    |        0    |     0      |         nan     |                       0    |                   nan     |               0 |                  0 |              nan     |          nan    |                    0 |                        nan     |        nan    |
| label_shuffle_straddle_slippage_stress     |                       1.5 |        0 |        0    |        0    |     0      |         nan     |                       0    |                   nan     |               0 |                  0 |              nan     |          nan    |                    0 |                        nan     |        nan    |
| label_shuffle_straddle_slippage_stress     |                       2   |        0 |        0    |        0    |     0      |         nan     |                       0    |                   nan     |               0 |                  0 |              nan     |          nan    |                    0 |                        nan     |        nan    |
| label_shuffle_straddle_slippage_stress     |                       2.5 |        0 |        0    |        0    |     0      |         nan     |                       0    |                   nan     |               0 |                  0 |              nan     |          nan    |                    0 |                        nan     |        nan    |

## Block Bootstrap

| block   |   sims |   p_positive |   total_pnl_p05 |   total_pnl_median |   total_pnl_worst |   max_drawdown_median |   max_drawdown_worst |   annualized_sharpe_p05 |   annualized_sharpe_median |
|:--------|-------:|-------------:|----------------:|-------------------:|------------------:|----------------------:|---------------------:|------------------------:|---------------------------:|
| day     |   2000 |            1 |         41841.9 |            60984.8 |          20505.6  |              -3975.78 |             -9896.75 |                   3.308 |                      4.58  |
| month   |   2000 |            1 |         32790.9 |            60104.7 |           4434.58 |              -4660.11 |            -13781.9  |                   1.849 |                      2.939 |
| week    |   2000 |            1 |         36590.5 |            61211   |          10518.2  |              -4912.35 |            -12768.7  |                   2.296 |                      3.498 |

## Feature Ablations

| variant                  |   slippage_points_per_leg |   trades |   total_pnl |   avg_trade |   win_rate |   profit_factor |   max_trade_order_drawdown |   trade_t_stat_not_sharpe |   monthly_count |   monthly_positive |   monthly_sharpe_ann |   monthly_worst |   daily_active_count |   daily_sharpe_ann_active_days |   daily_worst | feature_mode    |
|:-------------------------|--------------------------:|---------:|------------:|------------:|-----------:|----------------:|---------------------------:|--------------------------:|----------------:|-------------------:|---------------------:|----------------:|---------------------:|-------------------------------:|--------------:|:----------------|
| ablation_all             |                       1.5 |     1925 |     60511.4 |       31.43 |     0.6712 |           1.393 |                   -5378.91 |                     5.737 |              16 |                 14 |                2.914 |        -1819.81 |                  321 |                          4.427 |      -1210.53 | all             |
| ablation_no_time         |                       1.5 |     1367 |     55814.9 |       40.83 |     0.6789 |           1.553 |                   -3539.08 |                     6.222 |              16 |                 13 |                3.264 |        -1517.07 |                  319 |                          4.82  |      -1231.34 | no_time         |
| ablation_time_only       |                       1.5 |     1187 |     25709.9 |       21.66 |     0.6824 |           1.286 |                   -9552.14 |                     3.499 |              16 |                 10 |                1.334 |        -3663.53 |                  241 |                          3.358 |      -1203.72 | time_only       |
| ablation_volatility_only |                       1.5 |     1134 |     57152.4 |       50.4  |     0.7072 |           1.694 |                   -3634.46 |                     6.886 |              16 |                 13 |                3.366 |         -815.29 |                  315 |                          5.73  |      -1086.77 | volatility_only |

## Cross-Instrument Sanity Check

| variant              |   slippage_points_per_leg |   trades |   total_pnl |   avg_trade |   win_rate |   profit_factor |   max_trade_order_drawdown |   trade_t_stat_not_sharpe |   monthly_count |   monthly_positive |   monthly_sharpe_ann |   monthly_worst |   daily_active_count |   daily_sharpe_ann_active_days |   daily_worst | instrument   | data_path                                                                                          |
|:---------------------|--------------------------:|---------:|------------:|------------:|-----------:|----------------:|---------------------------:|--------------------------:|----------------:|-------------------:|---------------------:|----------------:|---------------------:|-------------------------------:|--------------:|:-------------|:---------------------------------------------------------------------------------------------------|
| cross_ES_same_logic  |                       1.5 |      251 |   -11846.9  |      -47.2  |     0.3307 |           0.427 |                  -14636.1  |                    -4.435 |               4 |                  1 |               -3.331 |        -6861.35 |                   64 |                         -6.603 |       -958.74 | ES           | data/raw/databento_es_1m.csv  |
| cross_MGC_same_logic |                       1.5 |      228 |    -9625.39 |      -42.22 |     0.057  |           0.041 |                   -9579.28 |                   -23.972 |              15 |                  0 |               -3.102 |        -2303.22 |                  112 |                        -18.847 |       -291.8  | MGC          | data/raw/databento_mgc_1m.csv |

## Final Portfolio Wording

Use: Built an independently audited NQ futures research pipeline with raw-data hashing, walk-forward replication, leakage checks, clean-eval and purged reruns, slippage stress, block-bootstrap Monte Carlo, and failure analysis. Do not claim a proven profitable trading bot.

