---
name: Qwen模型选型方案
description: MemSkill项目确认纯GRPO方案（无PPO），各模块Qwen模型选型和部署决策
type: project
---

## 方案确认（2026-07-28）

**核心决策：纯GRPO训练Designer，不使用PPO Controller。** PPO相关代码为历史遗留。

| 模块 | 模型 | 部署方式 | API名称 | 价格/百万Token |
|------|------|----------|---------|------|
| Designer | Qwen3.5-9B (Dense) | 本地全参数GRPO | - | GPU成本 |
| Skill Selector | Qwen3.6-Plus | 百炼API | qwen3.6-plus | 输入2/输出~8元 |
| Executor | Qwen3.6-Plus | 百炼API | qwen3.6-plus | 输入2/输出~8元 |
| QA Judge | Qwen3.7-Max | 百炼API | qwen3.7-max | 输入12/输出36元 |

**Why:** 全部使用Qwen生态统一技术栈；Designer用Dense模型方便全参数GRPO训练；Judge需要最强判断力作为reward信号来源；Selector和Executor统一用Plus平衡能力和成本。PPO Controller方案已废弃，纯GRPO通过Designer LLM直接生成skill方案更简洁高效。

**How to apply:**
- API统一用阿里云百炼(DashScope): https://dashscope.aliyuncs.com/compatible-mode/v1
- Designer从ModelScope下载 Qwen/Qwen3.5-9B
- 训练框架: OpenRLHF (Ray分布式 + vLLM推理加速)
- GPU需求: 2~4 × A100-80G (或H800)
- GRPO超参: G=8, clip=0.2, KL_coef=0.05, lr=5e-6, temp=0.7
- 训练入口: train_grpo.py (不是main.py)
- PPO相关代码(controller.py, main.py)为历史遗留，当前不使用