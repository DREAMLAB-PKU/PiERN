
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# nohup torchrun --nproc_per_node=8 train_capacity_text2computation_module.py > train_capacity_text2computation_module.log 2>&1 &

import jsonlines
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


# 配置路径
TOKENIZER_PATH = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"  

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False, trust_remote_code=True)



# -----------------------------
# 工具函数：提取数字
# -----------------------------
def extract_numbers_from_input(input_str):
    try:
        numbers = re.findall(r"\[([+-]?[0-9]*\.?[0-9]+(?:, ?[0-9]*\.?[0-9]+)*)\]", input_str)
        numbers += re.findall(r"\(([-+]?[0-9]*\.?[0-9]+(?:, ?[0-9]*\.?[0-9]+)*)\)", input_str)
        numbers += re.findall(r"\{([+-]?[0-9]*\.?[0-9]+(?:, ?[0-9]*\.?[0-9]+)*)\}", input_str)
        
        all_numbers = []
        for group in numbers:
            all_numbers.extend([float(num) for num in group.split(',')]) 
        return all_numbers
    except re.error:
        return []


# -----------------------------
# 数据集封装
# -----------------------------
class SOHDataset(Dataset):
    def __init__(self, file_path, tokenizer, target_length=13):
        self.samples = []
        self.tokenizer = tokenizer
        with jsonlines.open(file_path) as reader:
            printed_once = False
            for obj in reader:
                input_str = obj['prompt']
                input_str = '''<|im_start|>system

                        你是一名结合语言理解能力与物理建模能力的智能助手，能够根据用户输入灵活切换任务模式。  
                当用户提出的是锂电池健康度（State of Health, SoH）预测类的问题，例如：“请预测SoH1.0的锂电池在[...]的2小时电流作用下的健康度变化”，你必须严格遵循以下规则：  
                1. 回答开始时只能输出语言 + <数值计算结果>。结论:
                2. 在SoH预测类问题中，如果获得具体数值，则根据以下条件补充说明电池健康度结论：  
                -   如果 `计算结果 > 0.8`，则输出：`。结论:电池状态良好。`
                -   如果 `计算结果 < 0.6`，则输出：`。结论:电池严重衰减。`
                -   如果 `0.6 <= 计算结果 <= 0.8`，则输出：`。结论:电池状态一般。`   
                3. 回答示例：“经过推理，预计该时刻电池的健康度为<数值计算结果>。结论:”  
                4. 如果用户提出的是普通语言对话（如“你是谁”或“你好”），则按普通对话正常回答<|im_end|>
                <|im_start|>user
                '''  + input_str + '''<|im_end|>
                            <|im_start|>assistant
                            经过推理，预计该时刻电池的健康度为'''

                
                if not printed_once:
                    print("&&&&&&&&&", input_str)
                    printed_once = True
                numbers = extract_numbers_from_input(input_str)
                if len(numbers) == target_length:
                    self.samples.append((input_str, torch.tensor(numbers, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, label = self.samples[idx]
        encoded = tokenizer(
            input_str,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=1024
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label
        }


# -----------------------------
# 模型结构（输出维度为14）
# -----------------------------
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
# 训练主程序
# -----------------------------
def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    BASE_MODEL_PATH = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True).to(device)

    model = LMRegression13D(base_model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_fn = nn.MSELoss()

    # 数据加载
    file_path = "../data/train_capacity_text2computation_module.jsonl"
    dataset = SOHDataset(file_path, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    best_loss = float('inf')

    for epoch in range(1000000):  # 按 epoch 来控制
        sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 一致
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_y = batch["labels"].to(device)

            y_pred = model(input_ids, attention_mask)
            loss = loss_fn(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 and local_rank == 0:
                print(f"[epoch {epoch} step {step}] loss: {loss.item():.8f}")
                print("input :", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
                print("pred  :", [f"{v:.8f}" for v in y_pred[0].tolist()])
                print("truth :", [f"{v:.8f}" for v in batch_y[0].tolist()])
                print()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print(f"New best loss ({loss.item()}), saving model...")
                    torch.save(model.module.state_dict(), "capacity_test2computation_best_model.pt")


if __name__ == "__main__":
    main() 