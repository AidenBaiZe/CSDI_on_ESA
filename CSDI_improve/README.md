# ESA Mission 1 频率分析

本目录包含仅使用 `CSDI/ESA_mission1/data/processed/train` 训练段生成的频率可利用性分析。
主报告是自包含文件 `report.html`，不需要本地服务器或网络资源。

## 主要产物

- `report.html`：技术报告，包含 4 张原生图表、1 张参数表、方法、限制和实验建议。
- `frequency_analysis.py`：完整可复现分析脚本。
- `summary.json`：核心结论和推荐参数。
- `analysis.sqlite`：报告使用的结构化分析结果。
- `channel_horizon_metrics.csv`：76 个通道在 5 种上下文尺度下的频率统计。
- `window_spectral_metrics.csv`：窗口级频谱统计。
- `mask_robustness.csv`：遮挡比例、代理方法和上下文的聚合比较。
- `mask_robustness_detail.csv`：窗口/通道级代理频谱比较。
- `error_join.csv`、`error_associations.csv`：与现有 Structured-Only CSDI 连续块误差的后验关联。
- `source_notes.md`：报告结构映射、图表设计和证据清单。

## 复跑

```powershell
python .\frequency_analysis.py `
  --source-root "F:\2026小学期\CSDI" `
  --output-root "F:\2026小学期\CSDI_improve"
```

依赖：Python 3.10+、NumPy、pandas、SciPy。

分析固定随机种子为 `20260719`，频谱设计证据只来自 Mission 1 训练段。测试集的 seed-1
通道误差只用于提出后续实验假设，不参与频率上下文或 STFT 参数选择。
