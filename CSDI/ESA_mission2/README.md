# ESA Mission 2 — CSDI 全量随机遮盖实验

本目录在 ESA-ADB Mission 2 的 100 个遥测通道上运行 PhysioNet 风格的 CSDI 随机遮盖训练，协议与 `ESA_mission1` 保持一致，仅按 Mission 2 官方口径调整参数。

## 固定实验口径

- ESA-ADB 21 个月训练 / 21 个月测试：原始时间戳 `<= 2001-10-01` 为训练，`> 2001-10-01` 为测试。
- 只使用 100 个遥测通道，不使用 ESA-ADB 另外编码的 priority≥3 telecommand，因此输入维度是 100 而不是官方多变量文件的 104。
- 严格复刻官方 Mission 2 预处理：
  - 通道 29–46 在原始不规则采样上先做 `np.diff(value, append=value[-1])`。
  - 非数值（字符串）通道按官方脚本 `pd.factorize` 转为类别整数。
  - 按 18 秒零阶保持重采样，复刻异常样本还原及全局 `ffill().bfill()`。
  - Mission 2 标注只含 anomaly / rare event，两类之外的类别会直接报错（官方脚本没有 communication gap 分支）。
- 训练总计 32,000 步（约 5 万个训练窗口的 8 遍覆盖），不设验证集，正式结果固定使用最终模型。
- 梯度处理与 Mission 1 相同：全局梯度范数裁剪 1.0，NaN/Inf 梯度元素置零并记录数量。
- 测试缺失率为 10%、50%、90%，默认 256 个均匀分布窗口、每窗 50 个生成样本。
- 采样分块基准先做预热，再按"显存安全候选中实测耗时最短"选择，避免选到已溢出到系统内存的慢分块。

## 命令

```powershell
python .\preprocess.py --config .\config.yaml
python -m pytest .\tests -v
python .\run_mission2.py --mode smoke --config .\config.yaml
python .\run_mission2.py --mode train --config .\config.yaml
python .\run_mission2.py --mode evaluate --config .\config.yaml
python .\run_mission2.py --mode full --config .\config.yaml
```

`full` 会依次执行预处理（如缺失）、preflight 校验、smoke run、分块基准、完整训练、CSDI 与两种朴素基线评估，并生成 JSON、CSV、中文 Markdown 报告和汇总图。

## 与 Mission 1 的差异

| 项目 | Mission 1 | Mission 2 |
|---|---|---|
| 通道数 | 76 | 100 |
| 重采样间隔 | 30 秒 | 18 秒 |
| 官方划分点 | 2007-01-01（84/84 月） | 2001-10-01（21/21 月） |
| 差分通道 | 4–11 | 29–46 |
| 字符串通道 | 无 | `pd.factorize` |
| communication gap 标签 | 有 | 无（官方无此分支） |

注意：两个任务的网格间隔不同，同一缺失率下任务难度不同，跨任务只能做定性比较。

预处理产物按训练/测试分开保存为通道优先的内存映射 `.npy`，以避免一次性载入全部数据。
