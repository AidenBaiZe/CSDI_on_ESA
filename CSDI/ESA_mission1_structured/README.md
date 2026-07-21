# ESA Mission 1 Structured-Only CSDI

本实验复用 `ESA_mission1/data/processed` 的 76 通道数据和 ESA-ADB 84/84 月划分。新模型训练中不使用随机散点遮盖，只使用以下三类结构掩码：

- 全通道连续时间块；
- 部分通道整窗缺失；
- 部分通道与连续时间段的笛卡尔积。

三类训练掩码等概率抽取，请求严重度连续均匀分布于 `[0.1, 0.9]`。随机 10%、50%、90% 协议仅用于测试纯结构训练的泛化代价。

正式输出位于 `results/mission1_structured_only_seed1`。旧的 `mission1_structured_mix_seed1` 是未完成的废弃尝试，不属于正式实验。

常用入口：

```powershell
python ESA_mission1_structured/run_structured.py --mode preflight
python ESA_mission1_structured/run_structured.py --mode protocols
python ESA_mission1_structured/run_structured.py --mode smoke
python ESA_mission1_structured/run_structured.py --mode full --resume
```

`full` 按预检、协议、隔离进程冒烟、32,000 步训练、Random-CSDI 五项核心结构对照、Structured-Only-CSDI 十四项评估和报告生成的顺序运行。后台启动器只会等待 `TheBazaar` 退出，不会结束该进程或其他用户程序。
