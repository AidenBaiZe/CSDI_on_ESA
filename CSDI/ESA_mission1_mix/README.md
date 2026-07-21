# ESA Mission 1 Random–Structured Mix CSDI

本实验在每个训练窗口上独立选择 target strategy：

- 50% 使用论文的 Random strategy，遮盖比例从 `[0, 1]` 均匀采样；
- 50% 使用结构遮盖，并在连续时间块、整窗通道块、通道–时间矩形块之间等概率选择；
- 结构遮盖严重度从 `[0.1, 0.9]` 均匀采样。

这是 **Random–Structured Mix**，不是论文原版 Random–Historical Mix。它复用
`ESA_mission1/data/processed` 的数据及前两组实验保存的固定评估窗口，不修改已有结果。

正式输出位于 `results/mission1_random_structured_mix_seed1`。

## 命令

```powershell
python ESA_mission1_mix/run_mix.py --mode preflight
python -m pytest ESA_mission1_mix/tests -q
python ESA_mission1_mix/run_mix.py --mode protocols
python ESA_mission1_mix/run_mix.py --mode smoke
python ESA_mission1_mix/run_mix.py --mode train --resume
python ESA_mission1_mix/run_mix.py --mode evaluate-mix
python ESA_mission1_mix/run_mix.py --mode report
```

完整流水线：

```powershell
python ESA_mission1_mix/run_mix.py --mode full --resume
```

需要后台运行时可使用：

```powershell
powershell -ExecutionPolicy Bypass -File ESA_mission1_mix/run_background.ps1
```

`config.yaml` 中的 `mix_mask.random_probability` 可用于后续进行
`0.25 / 0.50 / 0.75` 消融。正式比较时应为每个概率使用独立输出目录，避免覆盖检查点。
