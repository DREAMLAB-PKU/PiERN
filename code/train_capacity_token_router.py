
# 启动命令：
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# nohup torchrun --nproc_per_node=8 train_capacity_token_router.py > train_capacity_token_router.log 2>&1 &

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
# please download the model at https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files
TOKENIZER_PATH = "/data/models/Qwen/Qwen2.5-0.5B-Instruct"  
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False, trust_remote_code=True)



# -----------------------------
# 数据集封装
# -----------------------------
class SOHDataset(Dataset):
    def __init__(self, file_path, tokenizer, target_length=1):
        self.samples = []
        self.tokenizer = tokenizer
        with jsonlines.open(file_path) as reader:
            printed_once = False
            for obj in reader:
                input_str = obj['text']
                
                if not printed_once:
                    print("&&&&&&&&&", input_str)
                    printed_once = True
                    
                numbers = obj['type']
                
                self.samples.append((input_str, torch.tensor(numbers, dtype=torch.float32)))
                
                # if len(numbers) == target_length:
                #     self.samples.append((input_str, torch.tensor(numbers, dtype=torch.float32)))

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
# 模型结构（输出维度为1）
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

# -----------------------------
# 训练主程序
# -----------------------------
def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    
    # 初始化模型
    vocab_size = len(tokenizer)
    model = LMClassifier1D(vocab_size).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 数据加载
    file_path = "../data/train_capacity_token_router.jsonl"
    dataset = SOHDataset(file_path, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=20, sampler=sampler)

    best_loss = float('inf')

    for epoch in range(1000000):  # 按 epoch 来控制
        sampler.set_epoch(epoch)  # 确保每个 epoch shuffle 一致
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_y = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)         # [B,1]
            loss = loss_fn(logits.view(-1), batch_y.float())  # batch_y: [B]，要转 float

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 and local_rank == 0:
                
                print(f"[epoch {epoch} step {step}] loss: {loss.item():.8f}")
                print("input :", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
                
                # 概率 & 预测
                probs = torch.sigmoid(logits)               # [B,1]
                pred_label = (probs > 0.5).long()           # [B,1]
                print("pred prob:", f"{probs[0].item():.4f}")
                print("pred label:", pred_label[0].item())

                # 标签
                print("truth     :", batch_y[0].item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print(f"New best loss ({loss.item()}), saving model...")
                    torch.save(model.module.state_dict(), "capacity_token_router_best_model.pt")


if __name__ == "__main__":
    main() 