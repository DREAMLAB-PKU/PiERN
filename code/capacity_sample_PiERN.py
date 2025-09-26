# export CUDA_VISIBLE_DEVICES=0
# nohup python capacity_sample_PiERN.py > PiERN_1.0B_capacity.log 2>&1 &

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import pynvml
import jsonlines

pynvml.nvmlInit()
gpu_index = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)  # 0号GPU

# -----------------------------
# 配置
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# please download the model at https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files
BASE_MODEL_PATH = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"

# please download the model at https://huggingface.co/HengBooo233/PiERN/tree/main
ROUTER_WEIGHTS = "../model/capacity_token_router.pt"                         
LM13D_WEIGHTS = "../model/capacity_test2computation_module.pt"   

   
DEEPONET_WEIGHTS = "../model/capacity_expert_model.pt"


# -----------------------------
# 模型定义
# -----------------------------
class LMClassifier1D(nn.Module):
    def __init__(self, vocab_size, embed_dim=1536, hidden_dim=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # [B, T, E]
        masked = embedded * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # 平均池化
        logits = self.fc(pooled)  # [B, 1]
        return logits
    

class DeepONet(nn.Module):
    """DeepONet: SoH 回归"""
    def __init__(self, n, dim):
        super(DeepONet, self).__init__()
        self.n_branch_net = nn.Sequential(
            nn.Linear(n, 2048), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(64, 10)
        )
        self.chunk_net = nn.Sequential(
            nn.Linear(dim, 784), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=0.02),
            nn.Linear(64, 10)
        )
        self.output_net = nn.Sequential(nn.Linear(10, 1))

    def forward(self, x):
        n_branch_inputs, chunk_inputs = x[0], x[1]
        n_branch_outputs = self.n_branch_net(n_branch_inputs)
        chunk_outputs = self.chunk_net(chunk_inputs)
        outputs = torch.sum(n_branch_outputs * chunk_outputs, dim=1, keepdim=True)
        return outputs


class LMRegression13D(nn.Module):
    def __init__(self, base_model, output_dim=13):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.model.embed_tokens.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        masked = last_hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.head(pooled)  # [B, 14]
    

# -----------------------------
# 初始化模型（一次性加载）
# -----------------------------
print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)

vocab_size = len(tokenizer)
router_model = LMClassifier1D(vocab_size).to(DEVICE)
router_model.load_state_dict(torch.load(ROUTER_WEIGHTS, map_location=DEVICE))
router_model.eval()

base_model_lm = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(DEVICE)
lm_model = LMRegression13D(base_model_lm).to(DEVICE)
lm_model.load_state_dict(torch.load(LM13D_WEIGHTS, map_location=DEVICE))
lm_model.eval()

deeponet = torch.load(DEEPONET_WEIGHTS, map_location=DEVICE).to(DEVICE)
deeponet.eval()

llm_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(DEVICE)
llm_model.eval()

print("All models loaded.")


# -----------------------------
# 工具函数
# -----------------------------
def inference_router_from_ids(input_ids, attention_mask):
    """高效 Router 推理：直接输入 token_ids，不需要 decode/encode"""
    with torch.no_grad():
        logits = router_model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(-1)  # [B]
        preds = (probs > 0.5).long()
    return preds.item(), probs.item()



def generate_response_deeponet_from_ids(input_ids, attention_mask) -> torch.Tensor:
    """返回张量 [B]（每条样本一个 SoH 数值）"""
    with torch.no_grad():
        preds_14d = lm_model(input_ids, attention_mask)       # [B, 14]
        print("!!!!!!", preds_14d)
        branch_in, chunk_in = preds_14d[:, :11], preds_14d[:, 11:]  # [B,11], [B,3]
        chunk_in_swapped = chunk_in[:, [1, 0]]   # 手动交换列
        outputs = deeponet((branch_in, chunk_in_swapped))             # [B,1]
        soh_vals = outputs.squeeze(-1)                        # [B]
    return soh_vals


def interpret_soh_value(soh_value: float) -> str:
    if soh_value > 0.8:
        return "电池健康度良好。"
    elif soh_value < 0.6:
        return "电池严重衰减。"
    else:
        return "电池状态一般。"


def generate_response_with_router(messages, tokenizer, llm_model, device, max_new_tokens=50):
    """逐步解码 + 高效 Router 判别"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = inputs["input_ids"]
    
    prompt_len = inputs["input_ids"].size(1)   # 初始 prompt 长度

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 1. 正常生成下一个 token
            outputs = llm_model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # 2. EOS 停止
            if next_token.item() == tokenizer.eos_token_id:
                break

            # 3. 拼接
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 4. Router 判别（直接吃 token_ids，不 decode）
            new_tokens = generated_ids[:, prompt_len:]  # 只保留新生成部分
            attention_mask = torch.ones_like(new_tokens).to(device)
            pred, prob = inference_router_from_ids(new_tokens, attention_mask)
            
            # print("***************", pred)

            if pred == 1:
                # 🚨 Router 触发 → 切到 DeepONet
                attention_mask = torch.ones_like(generated_ids).to(device)
                soh_vals = generate_response_deeponet_from_ids(generated_ids, attention_mask)  # [B]
                soh_value = float(soh_vals[0].item())
                soh_str = f"{soh_value:.6f}。结论：{interpret_soh_value(soh_value)}"
                # soh_str = f"{soh_value:.6f}。结论："
                soh_ids = tokenizer.encode(soh_str, add_special_tokens=False, return_tensors="pt").to(device)

                generated_ids = torch.cat([generated_ids, soh_ids], dim=-1)
                # break  # 已完成，退出循环

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)



if __name__ == "__main__":
    
    results_path = "PiERN_1.0B_capacity.txt"

    with open(results_path, "w", encoding="utf-8") as fout, jsonlines.open("../data/capacity_sample_PiERN.jsonl", "r") as reader:
        for idx, record in enumerate(reader):
            if idx >= 2:   # 👈 限制条数
                break

            if not isinstance(record, dict) or "prompt" not in record:
                continue
            message = str(record["prompt"]).strip()
            if not message:
                continue

            # 构建 messages
            messages_language = [
                {"role": "system", "content": '''
                你是一名结合语言理解能力与物理建模能力的智能助手，能够根据用户输入灵活切换任务模式。  
                当用户提出的是锂电池健康度（State of Health, SoH）预测类的问题，例如：“请预测SoH1.0的锂电池在[...]的2小时电流作用下的健康度变化”，你必须严格遵循以下规则：  
                1. 回答开始时只能输出语言 + <数值计算结果>。结论:
                2. 在SoH预测类问题中，如果获得具体数值，则根据以下条件补充说明电池健康度结论：  
                -   如果 `计算结果 > 0.8`，则输出：`。结论:电池状态良好。`
                -   如果 `计算结果 < 0.6`，则输出：`。结论:电池严重衰减。`
                -   如果 `0.6 <= 计算结果 <= 0.8`，则输出：`。结论:电池状态一般。`   
                3. 回答示例：“经过推理，预计该时刻电池的健康度为<数值计算结果>。结论:”  
                4. 如果用户提出的是普通语言对话（如“你是谁”或“你好”），则按普通对话正常回答
                '''},
                {"role": "user", "content": message}
            ]

            # === 统计开始 ===
            start_time = time.time()
            start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

            result = generate_response_with_router(messages_language, tokenizer, llm_model, DEVICE, max_new_tokens=100)

            # === 统计结束 ===
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            latency = end_time - start_time

            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            energy_J = (end_energy - start_energy) / 1000.0
            
            tokens_response = len(tokenizer.encode(result, add_special_tokens=False))

            # 写结果
            fout.write(f"Index {idx}\n")
            fout.write(f"Latency: {latency:.6f} s\n")
            fout.write(f"Tokens: response={tokens_response}, total={tokens_response}\n")
            fout.write(f"GPU{gpu_index} Energy: {energy_J:.6f} J\n")
            fout.write(f"Response: {result}\n")
            fout.write("-" * 40 + "\n")

    pynvml.nvmlShutdown()
    print(f"所有结果已保存到 {results_path}")