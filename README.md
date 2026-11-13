# telemetry_sample_lab

用可重放的哈希抽样采集终端遥测，并保留每条样本的包含概率和成本。

当前公开命令是 `inspect`，用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q telemetry_sample_lab
python3 -m unittest discover -s tests -v
python3 -m telemetry_sample_lab inspect --input fixtures/design.json --output /tmp/telemetry-sample.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`telemetry_sample_lab/sampling.py` 提供当前观测计算；`telemetry_sample_lab/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

pilot 是设计时已知的试采；population_sizes 是下一期规模估计。replay_population 仅供计划执行和重复回放验证。device_id 是设备簇，row_id 是调用记录；cost 是上传一条的整数预算单位。metrics 给出谓词、域、优先级和容许半宽。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
