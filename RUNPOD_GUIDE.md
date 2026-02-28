# dLLM-RL Agent 训练 RunPod 部署指南

基于 2026-02-27 实验总结，方便下次快速启动。

---

## 1. 创建 Pod

```bash
export RUNPOD_API_KEY="<your-key>"
MY_KEY=$(cat ~/.ssh/id_ed25519.pub)

runpodctl pod create \
  --name "dllm-agent-train" \
  --image "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --gpu-count 1 \
  --volume-in-gb 80 \
  --container-disk-in-gb 20 \
  --cloud-type "COMMUNITY" \
  --ports "22/tcp" \
  --ssh \
  --env "{\"PUBLIC_KEY\":\"$MY_KEY\"}"
```

**注意事项**：
- **必须用 PyTorch 2.2.0 + CUDA 12.1 镜像**。PyTorch 2.4.0 + CUDA 12.4 会在 `model.to("cuda")` 时 segfault，原因不明，换机器也一样。
- **不要用 `runpod/pytorch:2.4.0` 或 `2.6.0` 镜像**。
- A100 SXM 库存 High，一般秒开。L40S 容易排队等很久。
- container-disk 给 20GB 就够，模型放 volume 上。

## 2. 等待 SSH 就绪

```bash
# 查 portMappings 拿 SSH 端口
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://rest.runpod.io/v1/pods/<POD_ID> | python3 -c "
import sys, json, re
raw = re.sub(r'[\x00-\x1f\x7f]', ' ', sys.stdin.read())
d = json.loads(raw)
print(f\"IP: {d.get('publicIp', '')}  SSH Port: {(d.get('portMappings') or {}).get('22', 'N/A')}\")
"
```

**注意**：RunPod REST API 返回的 JSON 有时含控制字符，必须 sanitize 后再 `json.loads`。

## 3. 一键部署

```bash
SSH="ssh -o StrictHostKeyChecking=no -p <PORT> root@<IP>"

# 克隆代码
$SSH 'cd /workspace && git clone https://github.com/XiangJinyu/dLLM-RL.git && cd dLLM-RL && git checkout feature/agent-rl-training'

# 安装依赖（注意 transformers 版本上限）
$SSH 'pip install omegaconf termcolor jinja2 accelerate scipy nest_asyncio wandb einops "transformers>=4.38,<4.46"'

# 下载模型
$SSH 'python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id=\"GSAI-ML/LLaDA-8B-Base\", local_dir=\"/workspace/models/LLaDA-8B-Base\")
"'
```

## 4. 踩坑记录

### 4.1 transformers 版本

| 版本 | 结果 |
|------|------|
| 5.2.0 | `all_tied_weights_keys` AttributeError，LLaDA 自定义模型不兼容 |
| 4.49.0 | 和 PyTorch 2.4.0 搭配会 segfault |
| **4.45.2** | **可用**，和 PyTorch 2.2.0 搭配稳定 |

结论：**`pip install "transformers>=4.38,<4.46"`**

### 4.2 PyTorch + CUDA 镜像

| 镜像 | 结果 |
|------|------|
| `pytorch:2.6.0-py3.12-cuda12.6.3` | segfault on model.to("cuda") |
| `pytorch:2.4.0-py3.11-cuda12.4.1` | segfault on model.to("cuda") |
| **`pytorch:2.2.0-py3.10-cuda12.1.1`** | **稳定** |

结论：**用 2.2.0 镜像，不要贪新**

### 4.3 flash-attn

PyTorch 2.2.0 镜像自带的 flash-attn 可以直接用，不需要单独装。如果需要装：
```bash
pip install flash-attn --no-build-isolation
```

### 4.4 显存 (A100 80GB)

| 阶段 | 显存占用 |
|------|---------|
| 模型加载 (bf16) | 16.0 GB |
| 单轮 rollout 推理 | ~18.6 GB |
| 训练 forward+backward (batch=2, seq=650) | ~32 GB |
| 训练 forward+backward (batch=4, seq=670) | OOM |

结论：
- **rollout (eval mode)**：轻松，单 A100 跑 8B 模型完全没问题
- **训练 (train mode)**：batch_size 必须 <= 2，seq_len ~650 token 级别
- 如果 TraceRL 子样本太多 (>100)，随机采样 100 个，否则 OOM

### 4.5 einops

LLaDA 的 `modeling_llada.py` 依赖 einops，但仓库 requirements 没写。记得装：
```bash
pip install einops
```

### 4.6 RunPod CLI 新旧命令

```bash
# 旧命令（还能用但会 deprecated warning）
runpodctl get pod         # -> runpodctl pod list
runpodctl create pod      # -> runpodctl pod create
runpodctl remove pod      # -> runpodctl pod delete

# 新命令参数也不一样
# 旧: --imageName, --gpuType, --volumeInGb
# 新: --image, --gpu-id, --volume-in-gb
```

## 5. 快速验证命令

部署完后跑这个确认环境 OK：

```bash
$SSH 'cd /workspace/dLLM-RL && python3 -c "
import torch
print(f\"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}\")
print(f\"GPU: {torch.cuda.get_device_name(0)}\")
print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB\")
import transformers; print(f\"transformers: {transformers.__version__}\")
from agent.tools.calculator_tool import CalculatorTool
r = CalculatorTool().execute(expression=\"2+2\")
print(f\"Agent tools: {r.output}\")
"'
```

预期输出：
```
PyTorch: 2.2.0+cu121, CUDA: 12.1
GPU: NVIDIA A100-SXM4-80GB
Memory: 85 GB
transformers: 4.45.2
Agent tools: 4
```

## 6. 费用参考

| GPU | 价格 | 备注 |
|-----|------|------|
| A100 SXM 80GB | $1.39/hr | 推荐，库存好 |
| A100 PCIe 80GB | $1.19/hr | 便宜但这次遇到了 segfault |
| RTX 4090 24GB | $0.34/hr | 显存可能不够训练 |

本次实验（baseline eval + 5 步训练 + eval）总耗时 ~30 分钟，费用约 **$0.70**。

## 7. 别忘记关 Pod

```bash
runpodctl pod delete <POD_ID>
# 或
runpodctl pod list   # 看有没有遗漏的
```
