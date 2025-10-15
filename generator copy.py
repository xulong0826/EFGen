import math
import torch
from tqdm import tqdm
import pytorch_lightning
from pytorch_lightning import LightningModule
import numpy as np  # 添加这行
import random      # 添加这行
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PositionalEncoding needs to be definited manually, even if the transformer model is called from torch.nn
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
            d_model: the embedded dimension  
            max_len: the maximum length of sequences
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # PosEncoder(pos, 2i) = sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # PosEncoder(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: the sequence fed to the positional encoder model with the shape [sequence length, batch size, embedded dim].
        """
        x = x + self.pe[:x.size(0), :] # [max_len, batch_size, d_model] + [max_len, 1, d_model]
        return self.dropout(x)


# ============================================================================
# Definition of the Generator model
class GeneratorModel(LightningModule):
    
    def __init__(
        self,
        n_tokens, # vocabulary size
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        max_length = 1000,
        max_lr = 1e-3,
        epochs = 50,
        steps_per_epoch = None,
    ):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.max_length = max_length
        self.max_lr = max_lr
        self.epochs = epochs
        self.setup_layers()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            # total_steps=None, 
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs, 
            pct_start=6/self.epochs, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=1e3, 
            final_div_factor=1e3, 
            last_epoch=-1)
        
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]
    
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation)
        encoder_norm = torch.nn.LayerNorm(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)
        self.fc_out = torch.nn.Linear(self.d_model, self.n_tokens)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # Define lower triangular square matrix with dim=sz
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, features): # features.shape[0] = max_len
        mask = self._generate_square_subsequent_mask(features.shape[0]).to(self.device) # mask: [max_len, max_len]
        embedded = self.embedding(features)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        out_2= self.fc_out(encoded) # [max_len, batch_size, vocab_size]
        return out_2
    
    def step(self, batch):
        batch = batch.to(self.device)
        prediction = self.forward(batch[:-1]) # Skipping the last char
        loss = torch.nn.functional.cross_entropy(prediction.transpose(0,1).transpose(1,2), batch[1:].transpose(0,1)) # Skipping the first char
        return loss
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.step(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

# ============================================================================
# 新增：奖励归一化与经验回放缓冲区
class RewardBuffer:
    def __init__(self, capacity=10000, device='cpu'):
        self.buffer = []
        self.capacity = capacity
        self.device = device
        self.rewards = torch.tensor([], dtype=torch.float32, device=self.device)
        self.mean_reward = torch.tensor(0.0, device=self.device)
        self.std_reward = torch.tensor(1.0, device=self.device)

    def add(self, smiles, reward):
        if not isinstance(smiles, list):
            smiles = [smiles]
        if not isinstance(reward, list):
            reward = [reward]
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.rewards = torch.cat([self.rewards, reward_tensor])
        self.mean_reward = torch.mean(self.rewards)
        self.std_reward = torch.std(self.rewards) + 1e-8
        normalized_rewards = self.normalize_reward(reward_tensor)
        # 新增：去重，只保留独特分子
        existing_smiles = set(s for s, _ in self.buffer)
        for s, nr in zip(smiles, normalized_rewards):
            if s in existing_smiles:
                continue  # 跳过重复分子
            if len(self.buffer) >= self.capacity:
                buffer_rewards = torch.tensor([r for _, r in self.buffer], device=self.device)
                min_idx = torch.argmin(buffer_rewards)
                if nr > buffer_rewards[min_idx]:
                    self.buffer[min_idx] = (s, nr.item())
            else:
                self.buffer.append((s, nr.item()))
                existing_smiles.add(s)

    def normalize_reward(self, reward):
        std = max(self.std_reward.item(), 1e-8)
        mean = self.mean_reward.item()
        return (reward - mean) / std

    def sample(self, batch_size):
        if not self.buffer:
            return []
        buffer_rewards = torch.tensor([r for _, r in self.buffer], device=self.device)
        probs = torch.softmax(buffer_rewards, dim=0)
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)
        return [self.buffer[i][0] for i in indices.tolist()]

    def get_high_reward_samples(self, top_k=10):
        if not self.buffer:
            return []
        sorted_buffer = sorted(self.buffer, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_buffer[:min(top_k, len(sorted_buffer))]]


# ============================================================================
# Sampling "n" likely-SMILES from the GeneratorModel
class GenSampler():
    def __init__(self, model, tokenizer, batch_size, max_len, reward_fn="druglikeness", 
                 buffer_capacity=10000, replay_prob=0.3, m_part_dim=None, 
                 enable_surrogate=True, surrogate_intervention=True,
                 surrogate_threshold=0.5, surrogate_uncertainty_threshold=0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.reward_fn = reward_fn
        self.device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_buffer = RewardBuffer(capacity=buffer_capacity, device=self.device)
        self.replay_prob = replay_prob
        
        # Surrogate相关参数
        self.m_part_dim = m_part_dim
        self.enable_surrogate = enable_surrogate
        self.surrogate_intervention = surrogate_intervention
        self.surrogate_threshold = surrogate_threshold
        self.surrogate_uncertainty_threshold = surrogate_uncertainty_threshold
        
        # 如果启用Surrogate，则初始化模型
        self.surrogate = None
        self.m_part_buffer = list(self.reward_buffer.buffer)

        # 在 sample 方法中，Surrogate训练只在采集到足够新样本时触发
        self.surrogate_train_counter = getattr(self, "surrogate_train_counter", 0)
        self.surrogate_train_interval = getattr(self, "surrogate_train_interval", 500)  # 每采集500个新样本训练一次

        if enable_surrogate and m_part_dim:
            self.surrogate = Surrogate(m_part_dim).to(self.device)
            self.surrogate_optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=1e-3)
    
    # 新增：计算M_part
    def compute_m_part(self, smiles, full_molecule=True):
        """
        计算分子的M_part (规则特征+生成器表征)
        smiles: SMILES字符串或列表
        full_molecule: 是否是完整分子，片段分子特征可能有限
        返回: M_part向量或列表
        """
        from mol_metrics import compute_m_part_rule_features  # 推荐放在函数内，避免循环import

        if isinstance(smiles, str):
            smiles = [smiles]

        rule_features = []
        pooled_reprs = []

        for smi in smiles:
            # 规则特征（如指纹/描述符）
            if full_molecule:
                rule_feat = compute_m_part_rule_features(smi)  # shape=(n_rule_features,)
            else:
                # 片段分子特征：可选实现，或用零填充/截断
                rule_feat = np.zeros(2053)  # 或自定义片段特征处理
            rule_features.append(rule_feat)

            # 生成器表征（如 pooled_repr）
            # 推荐用模型接口获得表征
            try:
                pooled_repr = self.model.get_pooled_repr(smi)  # shape=(d_model,)
            except AttributeError:
                # 如果没有 get_pooled_repr 方法，则用 forward 得到编码后池化
                token_tensor = self.tokenizer.encode(smi)
                token_tensor = torch.tensor(token_tensor, dtype=torch.long, device=self.device).unsqueeze(1)  # [seq_len, 1]
                with torch.no_grad():
                    embedded = self.model.embedding(token_tensor)
                    positional_encoded = self.model.positional_encoder(embedded)
                    encoded_out = self.model.encoder(positional_encoded)
                    pooled_repr = encoded_out.mean(dim=0).cpu().numpy().flatten()  # shape=(d_model,)
            pooled_reprs.append(pooled_repr)

            # 输出每个特征的 shape
            # print(f"SMILES: {smi}")
            # print(f"  Rule feature shape: {rule_feat.shape}")
            # print(f"  Generator pooled_repr shape: {pooled_repr.shape}")

        # 拼接特征
        m_parts = [np.concatenate([rf, pr]) for rf, pr in zip(rule_features, pooled_reprs)]
        # print(f"M_part shape (each): {m_parts[0].shape}")

        # 如果是单个分子，返回单个向量
        if len(smiles) == 1:
            return m_parts[0]
        return m_parts
    
    # 新增：训练Surrogate
    def train_surrogate(self, epochs=2, batch_size=32):
        # 只用最近 batch_size*10 个样本训练
        train_data = self.m_part_buffer[-batch_size*10:] if len(self.m_part_buffer) > batch_size*10 else self.m_part_buffer
        if not self.enable_surrogate or not self.surrogate:
            print("Surrogate not enabled")
            return
        if len(self.m_part_buffer) < batch_size:
            print(f"Not enough data: {len(self.m_part_buffer)} samples")
            return
        losses = []
        m_parts = torch.tensor([x[0] for x in self.m_part_buffer], dtype=torch.float32)
        rewards = torch.tensor([x[1] for x in self.m_part_buffer], dtype=torch.float32)
        num_samples = len(self.m_part_buffer)
        for epoch in range(epochs):
            indices = torch.randperm(num_samples)
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_m_parts = m_parts[batch_idx]
                batch_rewards = rewards[batch_idx]
                loss = self.surrogate.train_step(batch_m_parts, batch_rewards, self.surrogate_optimizer)
                epoch_loss += loss
            avg_loss = epoch_loss / (num_samples // batch_size)
            losses.append(avg_loss)
        # print(f"Surrogate training losses: {losses}")
        return losses
    
    # 修改 sample 方法，添加Surrogate干预
    def sample(self, data=None, use_replay=False):
        """
        采样一批分子，可选使用Surrogate干预
        """
        self.model.eval()
        
        # 经验回放逻辑保持不变
        if use_replay and random.random() < self.replay_prob:
            replay_samples = self.reward_buffer.sample(self.batch_size)
            if replay_samples:
                return replay_samples
        
        finished = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        sample_tensor = torch.zeros((self.max_len, self.batch_size), dtype=torch.long, device=self.device)
        sample_tensor[0] = self.tokenizer.char_to_int[self.tokenizer.start]
        
        # 跟踪哪些分子被Surrogate拒绝
        rejected = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            if data is None:
                init = 1
            else:
                sample_tensor[:len(data)] = data.to(self.device)
                init = len(data)

            for i in range(init, self.max_len):
                tensor = sample_tensor[:i]
                logits = self.model.forward(tensor)

                # Surrogate干预逻辑（仅对补全后的分子执行）
                if self.enable_surrogate and self.surrogate_intervention and self.surrogate:
                    # 只对已完成（采样到结束符）的分子做干预
                    completed_indices = [j for j in range(self.batch_size) if finished[j] and not rejected[j]]
                    if completed_indices:
                        completed_smiles = [
                            "".join(self.tokenizer.decode(sample_tensor[:i, j].cpu().numpy())).strip("^$ ")
                            for j in completed_indices
                        ]
                        # 计算完整分子的M_part
                        m_parts = [self.compute_m_part(smi, full_molecule=True) for smi in completed_smiles]
                        # Surrogate预测
                        scores, uncertainties = self.surrogate.mc_predict(m_parts)
                        # 找出需要拒绝的分子索引
                        for idx, (score, uncertainty) in enumerate(zip(scores, uncertainties)):
                            if score.item() < self.surrogate_threshold or uncertainty.item() > self.surrogate_uncertainty_threshold:
                                if idx < len(completed_indices):
                                    rejected[completed_indices[idx]] = True

                # 正常采样逻辑
                if isinstance(logits, tuple):
                    logits = logits[0]  # 如果logits是元组，取第一个元素
                logits = logits[-1]  # 取最后一个时间步
                
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
                # 对被拒绝的分子，直接采样结束符
                for j in range(self.batch_size):
                    if rejected[j] and not finished[j]:
                        probabilities[j] = torch.zeros_like(probabilities[j])
                        probabilities[j, self.tokenizer.char_to_int[self.tokenizer.end]] = 1.0
                
                sampled_char = torch.multinomial(probabilities, 1).squeeze(-1)
                sampled_char[finished] = self.tokenizer.char_to_int[self.tokenizer.end]
                finished |= (sampled_char == self.tokenizer.char_to_int[self.tokenizer.end])
                sample_tensor[i] = sampled_char
                if finished.all():
                    break

        # 解码生成的SMILES
        smiles = ["".join(self.tokenizer.decode(sample_tensor[:, i].cpu().numpy())).strip("^$ ") for i in range(self.batch_size)]

        # 只对完整分子计算M_part和训练Surrogate
        if self.enable_surrogate and self.surrogate:
            # 过滤掉空字符串或极短片段
            valid_smiles = [smi for smi in smiles if len(smi) > 3]  # 你可根据实际情况调整长度阈值
            m_parts = [self.compute_m_part(smi, full_molecule=True) for smi in valid_smiles]

        # 奖励计算和存储
        if self.reward_fn is not None:
            rewards = self.reward_fn('druglikeness', smiles)
            if isinstance(rewards, np.ndarray) or torch.is_tensor(rewards):
                rewards = rewards.tolist()
            assert len(smiles) == len(rewards), f"smiles和rewards长度不一致: {len(smiles)} vs {len(rewards)}"
            self.reward_buffer.add(smiles, rewards)

            # Surrogate训练只用有效分子
            if self.enable_surrogate and self.surrogate:
                valid_rewards = [rewards[smiles.index(smi)] for smi in valid_smiles]
                for m_part, reward in zip(m_parts, valid_rewards):
                    self.m_part_buffer.append((m_part, reward))
                    self.surrogate_train_counter += 1

                # 只在采集到足够新样本时训练一次
                if self.surrogate_train_counter >= self.surrogate_train_interval:
                    self.train_surrogate(epochs=2)  # 只训练2轮
                    self.surrogate_train_counter = 0

        # print(f"Sampled molecules: {smiles}")
        return smiles
    
    # 修改 sample_with_uncertainty 方法
    def sample_with_uncertainty(self, n_samples=1, data=None, use_surrogate=True):
        """
        只用 Surrogate 的 MC Dropout 进行不确定性评估和筛选
        n_samples: 采样分子批次数（通常设为1即可）
        use_surrogate: 是否使用Surrogate进行不确定性评估
        """
        assert use_surrogate and self.enable_surrogate and self.surrogate, "Surrogate must be enabled!"

        self.model.eval()  # 生成时使用评估模式
        all_samples = []
        for _ in range(n_samples):
            batch_samples = self.sample(data, use_replay=False)
            # print(f"Sampled batch: {batch_samples}")
            all_samples.extend(batch_samples)

        # 计算所有分子的M_part
        all_m_parts = [self.compute_m_part(smi) for smi in all_samples]

        # 使用Surrogate进行MC Dropout预测
        mean_scores, std_scores = self.surrogate.mc_predict(all_m_parts, n_samples=10)

        # 使用均值和不确定性计算UCB分数
        ucb_scores = mean_scores + std_scores

        # 排序并返回分子和分数
        results = [(smi, mean.item(), std.item(), ucb.item()) 
                for smi, mean, std, ucb in zip(all_samples, mean_scores, std_scores, ucb_scores)]
        results.sort(key=lambda x: x[3], reverse=True)  # 按UCB分数排序

        # 只返回分子（如需分数可返回results）
        # print(f"Top sampled molecules by UCB score:", results[:self.batch_size])
        return [r[0] for r in results[:self.batch_size]]

    # Sampling "n" samples by the trained generator
    def sample_multi(self, n, filename=None, use_mc_dropout=False, use_replay=False, use_surrogate=True):
        """
        批量采样 n 个分子，只用 Surrogate MC Dropout 不确定性筛选
        """
        samples = []
        batch_size = self.batch_size
        num_batches = int(np.ceil(n / batch_size))
        for _ in tqdm(range(num_batches)):
            batch_samples = self.sample_with_uncertainty(
                n_samples=1,  # 只采样一批
                use_surrogate=use_surrogate
            )
            samples.extend(batch_samples)
        samples = samples[:n]

        # 写入文件
        if filename:
            with open(filename, 'w') as fout:
                for s in samples:
                    fout.write('{}\n'.format(s))
        # print(f"Sampled {len(samples)} molecules.")
        return samples

# ============================================================================
# 新增：Surrogate 模型 - 用于预测分子属性和不确定性
class Surrogate(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=1, dropout=0.2):
        """
        Surrogate 模型 - 预测分子属性和不确定性
        input_dim: M_part 向量的维度 (规则特征+生成器表征)
        hidden_dims: 隐藏层维度
        output_dim: 输出维度，通常为1（属性分数）
        dropout: dropout率，用于MC Dropout不确定性估计
        """
        super(Surrogate, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.dropout = dropout
    
    def forward(self, x):
        return self.model(x)
    
    def mc_predict(self, m_part, n_samples=10):
        """
        使用MC Dropout进行不确定性估计
        m_part: M_part向量或批量的M_part向量
        n_samples: MC采样次数
        返回: (均值, 标准差)
        """
        self.train()  # 启用dropout
        
        if isinstance(m_part, list):
            m_part = torch.tensor(m_part, dtype=torch.float32)
        
        device = next(self.parameters()).device
        m_part = m_part.to(device)
        
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(m_part)
                preds.append(pred)
        
        preds = torch.stack(preds, dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        self.eval()  # 恢复评估模式
        return mean, std
    
    def train_step(self, m_part, reward, optimizer):
        """
        批量训练
        m_part: [batch_size, input_dim]
        reward: [batch_size]
        optimizer: 优化器
        """
        if isinstance(m_part, list):
            m_part = torch.tensor(m_part, dtype=torch.float32)
        if isinstance(reward, list):
            reward = torch.tensor(reward, dtype=torch.float32)
        device = next(self.parameters()).device
        m_part = m_part.to(device)
        reward = reward.to(device)
        optimizer.zero_grad()
        pred = self(m_part).squeeze()
        loss = F.mse_loss(pred, reward)
        loss.backward()
        optimizer.step()
        return loss.item()