# GRPO Pipeline v4 — 进化管理机制修复

**日期**: 2026-07-29  
**版本**: v4 (evolution management)  
**前序**: v3 (record_usage → update_stats bugfix, commit fc0dc43)

---

## 问题诊断

通过对比原始 Pipeline (src/trainer.py 双层循环: PPO + Designer Evolution) 与 GRPO Pipeline 的架构差异，发现以下核心问题：

| 问题 | 表现 | 根因 |
|------|------|------|
| Apply门槛过严 | iter0后best_avg=0.667，后续好candidate被丢弃 | `improved = avg_reward > best_avg_reward` 才apply |
| Skill多样性为零 | 所有candidate都refine insert | prompt缺少usage统计和历史信息引导 |
| 无parse retry | parse失败直接reward=0 | 浪费LLM调用，降低有效样本率 |
| 无evolution history | 每轮prompt无前几轮信息 | LLM反复做同样的refine |

---

## 修复内容

### 1. Apply门槛改为 threshold-based（不再依赖 improved）

**文件**: `src/grpo_trainer.py` GRPOTrainingLoop.run()

**之前**:
```python
improved = avg_reward > self.best_avg_reward
if best_candidate_text and improved:
    self._apply_best_candidate(...)
```

**之后**:
```python
improved = avg_reward > self.best_avg_reward  # 仍用于early_stop追踪
should_apply = best_reward > self.config.min_apply_threshold  # 0.3
if best_candidate_text and should_apply:
    self._apply_best_candidate(...)
```

**效果**: 每轮只要best_reward > 0.3就apply，不再被第一轮的高分"卡死"。

### 2. Parse失败自动retry

**文件**: `src/grpo_trainer.py` GRPODataPreparer.prepare_grpo_batch()

- parse失败后重新调用LLM采样一条（temperature+0.1），最多retry `max_parse_retries` 次
- 默认 max_parse_retries=1

### 3. Evolution History 注入

**文件**: `src/grpo_trainer.py` GRPOTrainingLoop._build_evolution_history()

- 构建最近3轮的变化摘要（action, name, type, reward）
- 注入到 refinement_prompt_template 的 `{evolution_history}` 占位符
- 让LLM知道前几轮做了什么，避免重复

### 4. Skill Usage Stats 注入

**文件**: `src/grpo_trainer.py` GRPOTrainingLoop._build_skill_usage_stats()

- 构建当前每个skill的usage_count和avg_reward统计
- 标记 usage=0 的技能为 "Under-explored"
- 注入到 refinement_prompt_template 的 `{skill_usage_stats}` 占位符
- 引导LLM优先探索未被使用的skill

---

## 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `src/grpo_trainer.py` | 修改 | GRPOConfig新增2字段 + run()改apply逻辑 + 新增2个方法 + prepare_grpo_batch加retry |
| `src/config.py` | 修改 | 新增 grpo_min_apply_threshold / grpo_max_parse_retries 默认值和argparse参数 |
| `train_grpo.py` | 修改 | refinement_prompt_template新增2个占位符 + build_grpo_config映射 + 循环打印优化 |
| `train_mini_explore.sh` | 重写 | v4版本脚本，5轮，包含新参数 |

---

## 新增命令行参数

```
--grpo-min-apply-threshold FLOAT  Apply best candidate if best_reward > this (default: 0.3)
--grpo-max-parse-retries INT      Max retries per candidate on parse failure (default: 1)
```

---

## 实验路线图

### Phase 1: 本地验证 (已完成)
- `python -c "from src.grpo_trainer import ..."` 语法+逻辑验证 ✓

### Phase 2: SuperPod 单iteration验证
```bash
# SSH到SuperPod后:
cd ~/ust-ip && git pull ustip main
bash train_mini_explore.sh  # 5轮完整测试
```

**预期变化**:
- iter0: apply (best_reward > 0.3) → 正常
- iter1+: 即使avg_reward < iter0，只要best > 0.3也会apply
- 不同iteration应该看到不同skill被refine（不再全是insert）

### Phase 3: 小批量验证 (6 cases, 5 iters)
- 观察op_bank中update/delete的usage_count是否增长
- 观察evolution_history是否影响LLM的输出多样性

### Phase 4: 全量运行 (20+ cases, 10+ iters)
- 最终生成高质量GRPO training data用于OpenRLHF fine-tuning

---

## 与原始Pipeline的剩余差距

| 原始Pipeline有 | GRPO Pipeline 当前状态 | 优先级 |
|---------------|----------------------|--------|
| Rollback to best snapshot | 无（暂不需要，threshold-based更灵活） | 低 |
| Retry evolution (max 3次) | 有parse retry（但不是完整的evolution retry） | 中 |
| Evolution feedback (before/after对比) | 有evolution_history（简化版） | 已解决 |
| Embedding+clustering case selection | 顺序chunk+shuffle（够用） | 低 |
| new_action_bias | 无（通过usage stats引导替代） | 已解决 |

---

## 备注

- GRPOConfig 是 `@dataclass`，新字段有默认值，向后兼容
- refinement_prompt_template 的 `{evolution_history}` 和 `{skill_usage_stats}` 在首轮为空字符串，不影响LLM输出
- early_stop 仍基于 `improved`（avg_reward持续不提升则停止），但apply和early_stop解耦