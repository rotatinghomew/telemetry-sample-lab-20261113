# 多指标采样方案（planner）

`python3 -m telemetry_sample_lab.planner` 提供 `design`、`sample`、`estimate`、`replay`
四个子命令。全部使用 UTF-8 JSON，`--input`、`--output` 必填，`sample`/`estimate`/`replay`
另需 `--plan`。

```bash
python3 -m telemetry_sample_lab.planner design   --input fixtures/design.json --output plan.json
python3 -m telemetry_sample_lab.planner sample   --input fixtures/design.json --plan plan.json --output sample.json
python3 -m telemetry_sample_lab.planner estimate --input sample.json --plan plan.json --output estimate.json
python3 -m telemetry_sample_lab.planner replay   --input fixtures/design.json --plan plan.json --output replay.json --seeds 200
```

## 设计输入与"不读未来结果"

`design` 只读 `seed`、`baseline_probability`、`population_sizes`、`budget`、`metrics`、
`pilot`。`replay_population` 完全不被读取：

* 从输入里删掉 `replay_population`，计划与指纹逐字节相同（有测试）。
* 把 `replay_population` 的失败结果全部取反，计划与指纹仍然相同（有测试）。
* `design_input_digest` 只覆盖设计键；计划保存后可以对着交付数据复核，而不必回看未来结果。

`sample`/`replay` 读取 `population_rows`（或 `replay_population`）作为交付帧，这发生在
设计之后，不会回流到设计。

## 抽样设计

**选择单位是设备，不是记录。** 分层为 `tier × (设备是否含稀少应用行)`，即
`entry|rare`、`entry|standard`、`high|rare`、`high|standard`。每个分层按
`(seed, stratum, device_id)` 的哈希排序取前 `n_h` 台设备，被抽中设备的**全部行**上传。
因此：

* 每条保留样本带 `selection_unit`（`device_id`）、`selection_unit_id`、
  `inclusion_probability`，同设备各行包含概率相同，等于 `n_h / M_h`；
* 设备内多条调用相关，估计与区间按整群（cluster）设计计算，不能套记录级 SRS 公式。

**下一期设备规模**：用 pilot 设备数 ×（该 tier 下一期记录总数 ÷ pilot 记录总数）估计，
精确值与被四舍五入的整数都写进计划（`estimated_devices_exact` / `estimated_devices`）。
层规模估计偏差的影响单独输出在 `population_bias_sensitivity`：保持分配不变，把各层规模
乘以 0.5/0.8/1.0/1.25/2.0，列出计划半宽变化，并标出哪些层变成普查
（`n_h >= M_h`，fpc 项消失）。

**规划方差**（每个指标、每个层）：

1. 设备级残差 `z_d = t_d - R·a_d`，`a_d` 为该设备在指标域内的记录数，`t_d` 为字段和；
2. pilot 方差乘以上侧卡方因子 `df / χ²_{0.10,df}`（`df = 设备数-1`）；
3. 若该层 pilot 方差为 0（零失败或全失败），改用单侧 Clopper-Pearson 界给出的
   **正** 方差下限，并同时取"调整后方差"与下限的较大者作为保守先验；
4. **零失败不等于零方差**：计划里 `variance_floor_kind`、`binomial_floor`、
   `rate_bounds` 记录了下限来源与事件率区间；评估时若样本方差为 0 也回落到该先验。

**预算分配**：在整数格点上**穷举**各层设备数（每层至少 `--min-devices` 台，默认 2），按
优先级字典序优化：先最小化优先级 1 目标的最大"计划半宽/目标"比，再最小化和方，然后
优先级 3，最后最小化成本。每层至少保留设备，因此不会为了预算删除稀少层；若连"每层最少
设备"都超预算，`design` 直接失败并给出最低所需预算。

**预算是硬上限还是期望上限**（计划里 `spend` 明确写出）：

* `hard`：按各层 pilot 中**单台设备最大成本**预留，任何种子下实际支出都不超过
  `budget.amount`。执行时如果交付设备比 pilot 更贵，会**先**按最坏抽取（该层最贵的
  `n_h` 台）收缩各层设备数，再用收缩后的 `n_h/M_h` 作为包含概率；每层至少保留 1 台，
  连"每层 1 台"都付不起时拒绝执行，而不是超预算或丢层。收缩记录在
  `effective_allocation`、`hard_budget_shrink`、`hard_budget_worst_case`。
* `expected`：按各层**平均**设备成本规划，实际支出随种子波动，只保证期望不超预算；
  计划写 `expected_ceiling`，不写 `hard_ceiling`。

## 估计与区间

指标是域内均值（失败率等）的比值估计：

```
R = Σ_d w_d t_d / Σ_d w_d a_d,   w_d = 1/π_d = M_h/n_h
Var(R) = Σ_h (M_h/A)² (1 - n_h/M_h) S2_h / n_h,   A = Σ_d w_d a_d
```

* `M_h` 是交付数据里真实的设备数，不是计划的 `N_h`；包含概率在 `estimate` 中会与
  `M_h/n_h` 校验，不一致直接报错；
* 每个层用选中的 `z_d` 算样本方差；少于 2 台设备或样本方差为 0 时使用计划冻结的先验，
  并在 `variance_source`、`variance_used`、`degrees_of_freedom_used` 里注明来源；
* 区间为 `t(0.975, df)`，`df` 用 Satterthwaite 合成；二值指标使用 cluster-t 的
  logit 变换区间，保证落在 (0,1) 内（`interval_method = cluster-t-logit`）；
* 若某层在交付中整体缺失，执行会给出 `warnings` / `empty_strata`，估计不会把它当作
  零贡献以外的信息。

**与百分之一基线对比**：`replay` 同时跑同一批交付数据上的行级 Bernoulli(1%) 方案，
按行用 Wilson 区间（零事件时给出宽区间，而不是零宽），输出两者并列的
`defined_replays`、`bias`、`rmse`、`median_half_width`、`coverage` 与成本分布。
基线的选择单位是 `row_id`，计划的选择单位是 `device_id`；两种区间公式各按自己的设计计算。

## 精度不可达时的输出

`feasibility` 如实报告：

* `status`（`met` / `partially_met`）、`targets_met`、`targets_unmet`；
* 每个目标的计划半宽、`shortfall_absolute`、`ratio_to_target`；
* `required_budget_all_targets`（同时满足全部目标的最低 worst-case 预算）与
  `budget_gap`；
* 每个指标单独在预算内能做到的最好半宽、单独满足该目标所需的最低预算。

不会用超预算或删除稀少层来"宣称成功"。

## 风险说明（计划内 `risk_notes` 与 `pilot_diagnostics`）

* **先导样本不稳定**：pilot 设备少时卡方正则化因子大，计划区间偏保守；执行时优先用
  样本方差，仅在样本方差为 0 或设备不足时回落到先验。警告列出设备数少于 15 的层。
* **稀少层零失败**：按 Clopper-Pearson 下限给出正先验；`zero_events_in_sample` 标记
  样本零事件，估计注释说明区间由先验支撑。
* **层规模估计有偏**：分配使用 `N_h`，执行与估计使用交付的 `M_h`，包含概率按
  `n_h/M_h` 重算；`population_bias_sensitivity` 给出规模偏差对半宽的影响。
* **预算模式**：见上文 `spend`。

## 产物结构

| 命令 | `schema` | 关键字段 |
| --- | --- | --- |
| design | `telemetry-sample-plan/1` | `design`、`metrics[].stratum_components`、`allocation`、`spend`、`precision`、`feasibility`、`population_bias_sensitivity`、`risk_notes`、`plan_fingerprint` |
| sample | `telemetry-sample-execution/1` | `sample[]`（含 `selection_unit_id`、`inclusion_probability`）、`strata[]`、`spent`、`within_budget`、`effective_allocation`、`warnings` |
| estimate | `telemetry-sample-estimate/1` | `metrics[].estimate/half_width/interval_*/variance_source`、`summary` |
| replay | `telemetry-sample-replay/1` | `spend`、`plan`、`baseline_uniform`、`comparison`、`truth_from_replay` |

`plan_fingerprint` 覆盖除自身以外的全部计划字段；`sample`/`estimate` 都校验它，
计划被改动后不会被"照旧执行"。
