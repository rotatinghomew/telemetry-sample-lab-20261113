# telemetry_sample_lab

用可重放的哈希抽样采集终端遥测，并保留每条样本的包含概率和成本。

`inspect` 用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。
`planner` 提供多指标、按设备分层的抽样方案：`design`、`sample`、`estimate`、
`replay`。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q telemetry_sample_lab
python3 -m unittest discover -s tests -v
python3 -m telemetry_sample_lab inspect --input fixtures/design.json --output /tmp/telemetry-sample.json
python3 -m telemetry_sample_lab.planner design   --input fixtures/design.json --output /tmp/plan.json
python3 -m telemetry_sample_lab.planner sample   --input fixtures/design.json --plan /tmp/plan.json --output /tmp/sample.json
python3 -m telemetry_sample_lab.planner estimate --input /tmp/sample.json --plan /tmp/plan.json --output /tmp/estimate.json
python3 -m telemetry_sample_lab.planner replay   --input fixtures/design.json --plan /tmp/plan.json --output /tmp/replay.json --seeds 200
```

`design` 只读设计期字段（pilot、population_sizes、budget、metrics、seed 等），不读
`replay_population`；计划保存后以指纹校验，未来结果无法回流。设计口径、估计公式、
预算模式、零失败处理和不可达目标的报告方式见 `docs/planner.md`。

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`telemetry_sample_lab/sampling.py` 是最初的观测计算；`planner.py` 是四步命令入口，
`design.py`、`execute.py`、`estimate.py`、`replay.py` 分别负责设计、执行、估计和回放
校验；`common.py` 提供方差上界、下限与指纹工具；`io.py` 负责严格 JSON 读取和原子输出；
`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

pilot 是设计时已知的试采；population_sizes 是下一期规模估计。replay_population 仅供计划执行和重复回放验证。device_id 是设备簇，row_id 是调用记录；cost 是上传一条的整数预算单位。metrics 给出谓词、域、优先级和容许半宽。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
