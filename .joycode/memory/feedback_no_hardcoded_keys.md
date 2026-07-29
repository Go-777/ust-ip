---
name: 禁止硬编码API Key
description: 永远不要在代码或脚本中硬编码API key，必须使用环境变量引用
type: feedback
---

永远不要在任何文件（脚本、代码、配置）中硬编码API key或密钥。

**Why:** 用户曾因脚本中硬编码DashScope API key被push到GitHub，导致密钥泄露，需要rewrite整个git历史来清除。

**How to apply:** 
- 创建/修改脚本时，API key必须通过 `${DASHSCOPE_API_KEY}` 等环境变量引用
- 如需示例，用占位符如 `your-key-here` 或 `REDACTED`
- 绝不将真实key值写入任何会被git追踪的文件