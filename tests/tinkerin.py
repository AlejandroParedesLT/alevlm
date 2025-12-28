import torch

# Simulate logits and targets
batch_size = 16
vocab_size = 20

# Random logits: [batch_size, vocab_size]
logits = torch.randn(batch_size, vocab_size) * 2.0  # scaled for variability
targets = torch.randint(0, vocab_size, (batch_size,))

print("=== Logits ===")
print(logits)

# Step 1: Numerical stability shift
shift = torch.max(logits, dim=-1, keepdim=True).values
logits_shifted = logits - shift
print("\n=== Logits shifted (logits - max per row) ===")
print(logits_shifted)

# Step 2: Compute log-sum-exp
Logitsexp=torch.exp(logits_shifted)
sumLogits=torch.sum(Logitsexp, dim=-1)
logsumexp = torch.log(sumLogits)
print("\n=== logsumexp per example ===")
print(logsumexp)

# Step 3: Pick the logit corresponding to the correct class
correct = torch.gather(logits_shifted, -1, targets.unsqueeze(-1)).squeeze(-1)
print("\n=== Correct logits (for target class) ===")
print(correct)

# Step 4: Cross-entropy loss per example
loss_per_example = -(correct - logsumexp)
print("\n=== Cross-entropy loss per example ===")
print(loss_per_example)

# Step 5: Mean loss
loss = loss_per_example.mean()
print("\n=== Mean loss ===")
print(loss)
