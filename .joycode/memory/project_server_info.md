---
name: 服务器连接信息
description: MemSkill训练使用HKUST SuperPod集群(H800×8)，Slurm调度，项目路径~/ust-ip
type: project
---

HKUST SuperPod HPC集群信息：
- 登录节点: slogin-02 (用户: zzhaodg)
- 连接方式: Termius SSH → code tunnel (i4wyzdlnn)
- Slurm账户: mscbdtsuperpod, partition=normal(21节点DGX H800)
- GPU配额: 444.41/480.00 h (剩余~35.6h)
- GPU: NVIDIA H800 80GB × 8/节点
- conda env: memskill (torch2.6+cu124, trl0.15.2, vllm0.8.5)
- 项目路径: ~/ust-ip（不是~/MemSkill）
- GitHub仓库: Go-777/ust-ip

**Why:** 服务器上clone的仓库目录名是ust-ip，与本地MemSkill不同
**How to apply:** 所有训练脚本中cd路径必须用~/ust-ip