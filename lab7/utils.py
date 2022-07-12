import torch

# compute the current classification accuracy
def compute_acc(out, onehot_labels):
    batch_size = out.size(0)
    acc = 0
    total = 0
    for i in range(batch_size):
        k = int(onehot_labels[i].sum().item())
        total += k
        outv, outi = out[i].topk(k)
        lv, li = onehot_labels[i].topk(k)
        for j in outi:
            if j in li:
                acc += 1
    return acc / total

def MBCE(input, target, esp=1e-19):
    loss = - torch.mean(target * torch.log(input.clamp_min(esp))) - torch.mean(
        (1 - target) * torch.log((1 - input).clamp_min(esp)))
    return loss