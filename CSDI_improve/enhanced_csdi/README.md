# Enhanced CSDI（ESA Mission 1）

这是完整架构实现，不是消融脚本。原始 `CSDI` 目录保持不变，新增代码全部位于本目录。

## 已接入的计算链

1. 条件观测与含噪目标沿用 CSDI 的双通道输入。
2. Temporal Attention 学习同一通道跨时间的关系。
3. Feature Attention 通过已验证的“一阶差分 Pearson Top-8”图获得软注意力偏置，让每个通道优先关注相关通道。
4. Mask-aware STFT 只使用条件可见值，生成与原时间轴对齐的频率条件。
5. 图偏置和频率分支均使用零初始化可学习门控；模型从原 CSDI 行为附近开始训练，再逐步决定两类信息的权重。
6. 保留 CSDI 的噪声预测头、训练损失与反向扩散采样流程。

## 运行

```powershell
cd "F:\2026小学期\CSDI_improve\enhanced_csdi"
python run_enhanced.py build-graph
python -m pytest tests -q
python run_enhanced.py smoke
```

确认冒烟测试通过后，才启动完整训练：

```powershell
python run_enhanced.py train
```

训练完成后的固定协议评估：

```powershell
python run_evaluation.py smoke
python run_evaluation.py primary
```

`primary` 运行 9 个结构化缺失协议和 2 个 ESA gap 协议；`full` 另外加入
random 10%/50%/90%，共 14 项。已完成的协议会自动复用结果，可断点续跑。

核心参数在 `config.yaml`。当前保持 ESA Mission 1 既有窗口长度 96，以便与已有 CSDI 基线直接比较；频率参数采用 `n_fft=64, hop=16`，图为差分 Pearson Top-8。
