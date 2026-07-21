# ESA Mission 2 六通道 CSDI

本目录包含 Mission 2 六通道实验的独立副本，包括预处理数据、优化遮盖、模型、训练、评估、绘图和测试代码。

## 数据与遮盖

- 通道：9、10、13、14、74、86
- 时间范围：2000-01-01 至 2003-06-30
- 采样间隔：30 秒
- 窗口：96 步，步长 48
- 训练遮盖：50% 随机遮盖 + 50% ESA 真实标签形状遮盖
- 历史形状范围：5%–80%
- 随机遮盖范围：10%–90%

预处理数据位于 `data/aligned.npz`，数据说明位于 `data/manifest.json`。

## 检查

```powershell
python -m unittest discover -s .\tests -v
python .\run_esa.py --help
```

## 快速测试

```powershell
python .\run_esa.py --mode smoke --device cuda:0
```

## 完整训练

```powershell
python .\run_esa.py --mode baseline --seed 1 --device cuda:0 --epochs 200 --nsample 10 --max-eval-batches 10
```

默认训练结果写入 `results` 下的独立时间戳目录。训练完成后可生成结果图：

```powershell
python .\plot_results.py .\results\<结果目录>
```

`config.yaml` 是默认 Mission 2 配置，`config_mission2.yaml` 为同一配置的显式命名副本。
