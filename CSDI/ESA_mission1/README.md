# ESA Mission 1 — CSDI 全量随机遮盖实验

本目录在 ESA-ADB Mission 1 的 76 个遥测通道上运行 PhysioNet 风格的 CSDI 随机遮盖训练。

## 固定实验口径

- ESA-ADB 84 个月训练/84 个月测试：原始时间戳 `<= 2007-01-01` 为训练，`> 2007-01-01` 为测试。
- 只使用 76 个遥测通道，不使用 ESA-ADB 另外编码的 11 条高优先级 telecommand，因此输入维度是 76 而不是官方多变量文件的 87。
- 通道 4–11 在原始不规则采样上先做一阶差分；之后按 30 秒零阶保持重采样，并复刻异常样本还原及全局 `ffill().bfill()`。
- 训练总计 32,000 步，不设验证集，正式结果固定使用最终模型。
- 为保留异常段的原始归一化幅度并避免极端异常值造成参数溢出，训练采用全局梯度范数裁剪 1.0；每步裁剪前范数写入日志。
- 若极端异常输入导致个别梯度元素成为 NaN/Inf，仅将这些梯度元素置零并记录精确数量；不裁剪或改写输入数据。
- 若全部元素有限但 float32 聚合梯度范数溢出，该步梯度整体缩放为零并记录范数溢出标记，防止污染参数。
- 测试缺失率为 10%、50%、90%，默认 256 个均匀分布窗口、每窗 50 个生成样本。

## 命令

```powershell
python .\preprocess.py --config .\config.yaml
python -m unittest discover -s .\tests -v
python .\run_mission1.py --mode smoke --config .\config.yaml
python .\run_mission1.py --mode train --config .\config.yaml
python .\run_mission1.py --mode evaluate --config .\config.yaml --checkpoint .\results\<run>\model_final.pth
python .\run_mission1.py --mode full --config .\config.yaml
```

`full` 会训练最终模型、自动选择安全采样分块、估算评估耗时、运行 CSDI 与两种朴素基线，并生成 JSON、CSV、中文 Markdown 报告和汇总图。

预处理产物按训练/测试分开保存为通道优先的内存映射 `.npy`，以避免一次性载入约 14 年 × 76 通道的数据。
