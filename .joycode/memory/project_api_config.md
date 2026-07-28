---
name: DashScope API配置
description: MemSkill项目使用阿里云百炼DashScope API，qwen-plus模型
type: project
---

API配置（2026-07-28确认可用）：
- Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
- Model: qwen-plus
- API Key: 环境变量 DASHSCOPE_API_KEY（sk-ws-前缀，百炼安全升级后格式）
- 格式: OpenAI兼容

**Why:** 项目已从京东云tokenPlan迁移到阿里云百炼DashScope，所有京东云相关配置已移除。

**How to apply:**
- 所有角色(selector/executor/judge/designer)统一使用 --api-base 参数或 DASHSCOPE_API_BASE 环境变量
- Key通过 DASHSCOPE_API_KEY 环境变量传递
- 不再使用京东云、SSH tunnel等方案