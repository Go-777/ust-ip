---
name: 不使用JD Cloud平台
description: 训练和开发不再使用JD Cloud Notebook，不要建议SSH到nb-*机器
type: feedback
---

不再使用JD Cloud平台（Notebook nb-m7wkdajepx / nb-qajs2a611p 等）进行训练或开发。

**Why:** 用户明确表示以后都不用JD Cloud平台。

**How to apply:**
- 不要建议SSH到 nb-* 机器
- 训练环境改为HKUST SuperPod（通过Slurm sbatch提交）或其他用户指定的环境
- 涉及服务器操作时，询问用户当前使用的环境，而非默认推荐JD Cloud