# MemSkill 训练成本估算 & 代码逻辑审查

> 基于 locomo10.json (10 conversations) + 默认超参的分析
> 日期: 2024-07-24

---

## 一、训练成本估算

### 1.1 数据规模

| 指标 | 数值 |
|------|------|
| 对话数 (conversations) | 10 |
| Sessions/对话 | 19 (full-session mode) |
| Turns/对话 | ~420 (avg 22 turns/session) |
| QA/对话 | ~200 (range 105-260) |
| 训练时QA采样率 | 1.0 (全量) |

### 1.2 主训练循环 LLM 调用量

```
outer_epochs = 10, inner_epochs = 100, batch_size = 4
总 inner epochs = 10 × 100 = 1,000
```

**每个 inner_epoch 调用量：**

| 阶段 | 模型 | 调用数/epoch | Input tokens/call | Output tokens/call |
|------|------|------|------|------|
| Selector | Qwen3.6-Plus | 4 × 19 = 76 | ~2,000 | ~200 |
| Executor | Qwen3.6-Plus | 4 × 19 = 76 | ~3,000 | ~500 |
| QA Response | Qwen3.6-Plus | 4 × 200 = 800 | ~2,000 | ~200 |
| QA Judge (可选) | Qwen3.7-Max | 4 × 200 = 800 | ~1,500 | ~100 |

**总调用量(1000 epochs)：**

| 阶段 | 总调用数 | 总Input tokens | 总Output tokens |
|------|------|------|------|
| Selector | 76,000 | 152M | 15.2M |
| Executor | 76,000 | 228M | 38M |
| QA Response | 800,000 | 1,600M | 160M |
| **小计 (无Judge)** | **952,000** | **1,980M** | **213.2M** |
| QA Judge (可选) | 800,000 | 1,200M | 80M |
| **小计 (含Judge)** | **1,752,000** | **3,180M** | **293.2M** |

**Designer Evolution (共10次):**
- analysis + refinement: ~40 calls × (input 4000 + output 2000) = 忽略不计

### 1.3 API 费用计算

#### 方案A: reward_metric=f1 (无Judge，推荐起步方案)

| 模型 | Input (M tokens) | Output (M tokens) | Input费 | Output费 | 合计 |
|------|------|------|------|------|------|
| Qwen3.6-Plus | 1,980 | 213.2 | ¥3,960 | ¥1,706 | **¥5,666** |

#### 方案B: reward_metric=llm_judge (含Judge)

| 模型 | Input (M tokens) | Output (M tokens) | Input费 | Output费 | 合计 |
|------|------|------|------|------|------|
| Qwen3.6-Plus | 1,980 | 213.2 | ¥3,960 | ¥1,706 | ¥5,666 |
| Qwen3.7-Max | 1,200 | 80 | ¥14,400 | ¥2,880 | ¥17,280 |
| **总计** | | | | | **¥22,946** |

### 1.4 GRPO 数据准备 (train_grpo.py, 独立流程)

```
max_iterations = 50, case_chunk_size = 5, G = 8
```

| 阶段 | 调用数 | Input tokens | Output tokens |
|------|------|------|------|
| Analysis (Designer) | 50 | 200K | 100K |
| Sampling (Designer, ×8) | 50×8=400 | 1.6M | 800K |
| Reward compute (Selector+Executor) | 50×8×5×2=4,000 | 10M | 2M |
| **GRPO总计** | ~4,450 | ~12M | ~3M |

GRPO API费: Qwen3.6-Plus ≈ ¥24 + ¥24 = **¥48** (可忽略)

### 1.5 GPU 训练成本 (GRPO fine-tune Qwen3.5-9B)

| 资源 | 需求 | 估算 |
|------|------|------|
| GPU | 2×A100-80G 或 1×H200 | 已有H200 |
| 训练数据量 | ~50×8 = 400 条 GRPO samples | 较少 |
| 训练时间 | 2-4小时 (OpenRLHF + vLLM) | SFT较快 |
| **GPU成本** | **如果使用现有H200: ¥0** | 如租用按~¥50/H/卡 |

### 1.6 训练时间估算

**瓶颈：API调用速度 (rpm限制)**

假设百炼API rpm=60 (单key), 使用2个key → 有效rpm≈100:

| 方案 | 总调用数 | 理论时间 (100 rpm) | 考虑并发/批量后 |
|------|------|------|------|
| 方案A (f1) | 952,000 | 159小时 = 6.6天 | **4-5天** |
| 方案B (judge) | 1,752,000 | 292小时 = 12.2天 | **8-10天** |

**优化手段（可大幅缩短）：**
1. 降低 `inner_epochs` 到 30-50（当前100过多）→ 时间×0.3-0.5
2. 降低 `locomo_train_query_sampling_ratio` 到 0.3 → QA calls×0.3
3. 增加API keys（rpm线性增长）
4. 使用本地Qwen3.5-9B做Selector/Executor（GPU推理，无rpm限制）

**优化后估算 (inner_epochs=50, QA_ratio=0.3, 2 keys):**

| 方案 | 总调用数 | 预计时间 |
|------|------|------|
| 方案A (f1优化) | ~196,000 | **1.5-2天** |
| 方案B (judge优化) | ~316,000 | **2.5-3天** |

### 1.7 成本总结

| | 方案A (推荐起步) | 方案A (优化后) | 方案B (含Judge) |
|---|---|---|---|
| API费用 | ¥5,666 | **¥1,200-1,700** | ¥22,946 |
| GPU费用 | ¥0 (已有) | ¥0 | ¥0 |
| 训练时间 | 4-5天 | **1.5-2天** | 8-10天 |
| GRPO额外 | ¥48 | ¥48 | ¥48 |

**推荐方案：方案A优化版，预算¥1,500-2,000，时间2天左右。**

---

## 二、代码逻辑问题清单

### 严重问题 (会导致crash或功能无效)

#### BUG-1: wandb config引用不存在的属性 (crash)
- **位置**: `src/trainer.py:1093-1098`
- **问题**: `self.controller.gamma/gae_lambda/clip_epsilon/entropy_coef/value_coef`
  LLMController只有`action_top_k`属性，训练启动时wandb初始化会AttributeError
- **影响**: 使用LLMController时无法启动训练
- **修复**: `getattr(self.controller, 'gamma', None)` 或条件判断controller类型

#### BUG-2: GRPO _f1_reward() 完全无效 (核心功能缺失)
- **位置**: `src/grpo_trainer.py:266-275`
- **问题**: 
  ```python
  def _f1_reward(self, candidate, case):
      # 仅检查"INSERT"/"UPDATE"关键词是否出现
      # +0.1 bonus if keyword found
  ```
  没有重新执行QA evaluation计算真实F1变化。reward信号是噪声。
- **影响**: GRPO训练数据的reward标签无意义，模型学不到有用信号
- **修复**: 完整pipeline: apply candidate → rebuild memory → run QA → compute F1 delta

#### BUG-3: _evaluate_single_case() 评估逻辑不完整 (语义错误)
- **位置**: `src/grpo_trainer.py:163-238`
- **问题**:
  - 没有重建完整memory bank（只有case中的snapshot）
  - executor执行后结果未apply到任何memory
  - 评估的是executor响应文本质量，不是最终QA准确度
- **影响**: reward_compute逻辑与实际训练目标脱节
- **修复**: 需要per-case mini-episode: init memory → apply all sessions up to case → execute → QA eval

#### BUG-4: OperationBank初始化参数错误 (crash)
- **位置**: `train_grpo.py:119`
- **问题**: `OperationBank(args)` — 构造函数签名是 `(encoder=None, max_ops=20, skip_noop=False)`
  `args`被传为`encoder`参数
- **影响**: GRPO训练脚本启动后可能在encoder.encode()时crash
- **修复**: `OperationBank(encoder=None, max_ops=getattr(args, 'max_ops', 20))`

### 中等问题 (功能受限)

#### BUG-5: JSON解析不支持嵌套对象
- **位置**: `src/grpo_trainer.py:423-425`
- **问题**: `re.search(r'\{[^{}]*\}', text)` — instruction_template字段常含`{placeholder}`
- **影响**: 解析candidate失败率高，有效sample减少
- **修复**: 
  ```python
  # 从第一个{开始尝试json.loads，逐步扩展
  start = text.index('{')
  for end in range(len(text)-1, start, -1):
      if text[end] == '}':
          try: return json.loads(text[start:end+1])
          except: continue
  ```

#### BUG-6: _judge_reward() 过于粗糙
- **位置**: `src/grpo_trainer.py:277-300`
- **问题**: 只支持0/0.5/1.0三档，prompt过短无few-shot，判断对象是operation而非QA
- **影响**: Judge reward信号粒度不足，难以区分相近candidate
- **修复**: 使用5分制 + few-shot examples + 评估最终QA质量

### 低优先级问题 (稳定性/健壮性)

#### BUG-7: SSH隧道不稳定
- **问题**: sleep 86400的后台隧道已断2次
- **修复**: 使用autossh + ServerAliveInterval=30 + ExitOnForwardFailure=yes

#### BUG-8: ThreadPoolExecutor线程安全
- **位置**: `src/trainer.py` parallel episode collection
- **问题**: 共享llm_client/encoder可能有竞态
- **影响**: 偶发异常（低概率）
- **修复**: 每episode独立的memory_bank已隔离（已做），llm_client内部加锁

#### BUG-9: compute_returns_and_advantages引用controller属性
- **位置**: `src/trainer.py:975`
- **问题**: 引用`self.controller.gamma/gae_lambda`，LLMController时虽然L950-960 early return跳过，但代码脆弱
- **修复**: 移入PPOController子类或添加防御性检查

---

## 三、优先修复建议

### 训练前必须修复 (否则无法启动/训练无效)

1. **BUG-1** — 5分钟，getattr保护
2. **BUG-4** — 5分钟，修正参数传递
3. **BUG-2 + BUG-3** — 核心，需重写reward计算逻辑（2-4小时）

### 训练前建议修复 (提高训练效果)

4. **BUG-5** — 30分钟，JSON解析增强
5. **BUG-6** — 1小时，Judge prompt改进

### 训练开始后可修复

6. **BUG-7** — autossh配置
7. **BUG-8/9** — 防御性编程

---

## 四、推荐行动计划

1. 修复 BUG-1, BUG-4 (10分钟)
2. 重写 GRPO reward 计算 (BUG-2+3, 核心工作, 2-4小时)
3. 修复 BUG-5 JSON解析 (30分钟)
4. 配置百炼API keys + 设定优化参数 (inner_epochs=50, QA_ratio=0.3)
5. 先用 reward_metric=f1 启动方案A优化版训练
6. 预算：¥1,500-2,000，时间：2天