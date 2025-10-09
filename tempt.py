# compare_quantize_full_vs_mask.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed=0):
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_with_noise(logits, gumbel_noise, tau=1.0, hard=True):
    """
    logits: (N, K)
    gumbel_noise: same shape
    returns probs (N, K) with straight-through if hard=True
    """
    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    if not hard:
        return y
    k = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, k, 1.0)
    return (y_hard - y).detach() + y

def extract_masked_elements(x, mask):
    # x: (B,S,D), mask: (B,S) boolean -> returns (N_mask, D)
    B, S, D = x.shape
    flat = x.view(B * S, D)
    flat_mask = mask.view(B * S)
    return flat[flat_mask]

# ---------------------------
# Quantizer (supports external logits & gumbel)
# ---------------------------
class SimpleQuantizer(nn.Module):
    def __init__(self, model_dim, num_entries=8, temp=1.0):
        super().__init__()
        self.entry_proj = nn.Linear(model_dim, num_entries, bias=False)
        self.entries = nn.Parameter(torch.randn(num_entries, model_dim))
        self.temp = temp

    def compute_from_logits_and_noise(self, logits, gumbel_noise, mask_idx=None, hard=True):
        """
        logits: (N_total, K) - precomputed for full set (e.g. B*S)
        gumbel_noise: same shape
        mask_idx: 1D boolean index of length N_total ; if provided compute prob_perplexity only on masked positions
        returns:
            quantized_full (N_used, D) if masked used, or full (N_total, D) if mask_idx is None/False everywhere depending usage
            probs_used (N_used, K)
            prob_perplexity (scalar) computed ONLY on used positions (masked)
        """
        probs_all = gumbel_softmax_with_noise(logits, gumbel_noise, tau=self.temp, hard=hard)
        # quantize vectors for all entries
        quant_all = probs_all @ self.entries  # (N_total, D)
        if mask_idx is None:
            # compute perplexity on all
            avg_probs = probs_all.mean(dim=0)
            prob_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))
            return quant_all, probs_all, prob_perplexity

        # index used positions
        if mask_idx.dtype != torch.bool:
            mask_idx = mask_idx.bool()
        probs_used = probs_all[mask_idx]  # (N_used, K)
        quant_used = quant_all[mask_idx]  # (N_used, D)
        if probs_used.numel() == 0:
            # no masked positions -> define perplexity = 0 (or small)
            prob_perplexity = torch.tensor(0.0, device=logits.device)
        else:
            avg_probs = probs_used.mean(dim=0)
            prob_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-12)))
        return quant_used, probs_used, prob_perplexity

# ---------------------------
# Main test (A + B + C)
# ---------------------------
def main():
    set_seed(42)

    # toy sizes
    B = 2
    S = 6
    D = 16
    NUM_ENTRIES = 8
    TAU = 0.5

    device = torch.device("cpu")

    # toy z_n (this simulates encoder output); we'll compare grads w.r.t this
    z_n_base = torch.randn(B, S, D, device=device)

    # manual mask (B, S)
    mask = torch.tensor([[True, False, True, False, False, True],
                         [False, True, False, True, False, False]], dtype=torch.bool, device=device)

    # flatten mask for indexing in logits (length B*S)
    flat_mask = mask.view(-1)

    # initialize modules (same initial states for both cases)
    tmp_final_proj = nn.Linear(D, D).to(device)
    tmp_final_tproj = nn.Linear(D, D).to(device)
    tmp_quant = SimpleQuantizer(D, num_entries=NUM_ENTRIES, temp=TAU).to(device)

    init_fp_state = deepcopy(tmp_final_proj.state_dict())
    init_ft_state = deepcopy(tmp_final_tproj.state_dict())
    init_q_state = deepcopy(tmp_quant.state_dict())

    # Freeze quantizer parameters (A)
    quant = SimpleQuantizer(D, num_entries=NUM_ENTRIES, temp=TAU).to(device)
    quant.load_state_dict(init_q_state)
    for p in quant.parameters():
        p.requires_grad = False  # freeze quantizer params

    # We'll still use quant.entry_proj to compute logits_full (deterministic given same entry_proj)
    # sample gumbel noise once for the full set (C)
    z_flat = z_n_base.view(-1, D)  # (B*S, D)
    with torch.no_grad():
        logits_full = quant.entry_proj(z_flat)  # (B*S, K)
    gumbel_full = sample_gumbel(logits_full.shape, device=logits_full.device)

    # Helper to run a case; modules final_proj and final_tproj are given as initial copies
    def run_case(case_id):
        # fresh clones of final proj modules with identical init states
        final_proj = nn.Linear(D, D).to(device)
        final_proj.load_state_dict(init_fp_state)
        final_target_proj = nn.Linear(D, D).to(device)
        final_target_proj.load_state_dict(init_ft_state)

        # z_n input as requires_grad True
        z = z_n_base.detach().clone().requires_grad_(True)

        # seqs_masked (projected encoder outputs for masked pos) - COMMON for both cases
        seqs_masked = extract_masked_elements(z, mask)  # (N_mask, D)
        seqs = final_proj(seqs_masked)  # (N_mask, D)

        # --- quantization ---
        if case_id == 1:
            # quantize full using logits_full & gumbel_full (C). Then extract masked quantized vectors.
            quant_full, probs_full, _ = quant.compute_from_logits_and_noise(
                logits_full, gumbel_full, mask_idx=None, hard=True
            )  # quant_full: (B*S, D)
            quant_full = quant_full.view(B, S, D)
            quant_masked = extract_masked_elements(quant_full, mask)  # (N_mask, D)

            # For diversity stats (B) compute on masked positions only (B)
            _, probs_used_for_div, prob_perplexity = quant.compute_from_logits_and_noise(
                logits_full, gumbel_full, mask_idx=flat_mask, hard=True
            )
        elif case_id == 2:
            # Use logits_full masked positions and gumbel_full masked -> quantize only masked items but reuse same logits+noise
            logits_masked = logits_full[flat_mask]       # (N_mask, K)
            gumbel_masked = gumbel_full[flat_mask]
            quant_masked, probs_masked, prob_perplexity = quant.compute_from_logits_and_noise(
                logits_masked, gumbel_masked, mask_idx=None, hard=True
            )  # quant_masked: (N_mask, D)
        else:
            raise ValueError("case_id must be 1 or 2")

        targets = final_target_proj(quant_masked)  # (N_mask, D)

        # contrastive-like loss -> here we use MSE to compare seqs and targets
        recon_loss = F.mse_loss(seqs, targets, reduction="mean")

        # diversity loss (B): we compute it ONLY over masked positions (B) to avoid batch-statistic difference (B)
        # prob_perplexity was computed on masked positions already
        diversity_loss = -torch.log(prob_perplexity + 1e-12)  # optional; use simple transform

        # combine (we can weight diversity small)
        loss = recon_loss + 0.0 * diversity_loss  # set weight 0 if you want no diversity effect; keep 0.0 here to isolate recon
        # If you want to include diversity to see effect, change 0.0 -> small value like 0.1

        # backward
        loss.backward()

        # collect grads
        grads = {
            "z_grad": z.grad.clone(),
            "final_proj_grads": {n: p.grad.clone() if p.grad is not None else None for n, p in final_proj.named_parameters()},
            "final_target_proj_grads": {n: p.grad.clone() if p.grad is not None else None for n, p in final_target_proj.named_parameters()},
            "prob_perplexity": prob_perplexity.detach().cpu().item(),
            "recon_loss": recon_loss.detach().cpu().item(),
            "diversity_loss": diversity_loss.detach().cpu().item(),
        }

        # also do a parameter update step (quant frozen so excluded)
        opt = torch.optim.SGD([p for p in final_proj.parameters()] + [p for p in final_target_proj.parameters()], lr=1e-2)
        opt.step()

        # after update, capture parameter deltas (norms)
        param_deltas = {
            "final_proj_param_norm": torch.cat([p.data.view(-1) for p in final_proj.parameters()]).norm().item(),
            "final_target_proj_param_norm": torch.cat([p.data.view(-1) for p in final_target_proj.parameters()]).norm().item(),
        }

        return loss.detach().cpu().item(), grads, param_deltas

    # Run both cases with identical initial weights & same quantizer/logits/noise
    loss1, grads1, deltas1 = run_case(1)
    loss2, grads2, deltas2 = run_case(2)

    # Compare gradients of z
    g1 = grads1["z_grad"]
    g2 = grads2["z_grad"]
    g1_norm = g1.norm().item()
    g2_norm = g2.norm().item()
    max_abs_diff = (g1 - g2).abs().max().item()
    allclose = torch.allclose(g1, g2, atol=1e-6)

    print("=== Summary ===")
    print("Loss case1:", loss1)
    print("Loss case2:", loss2)
    print("||grad(z_n)|| case1:", g1_norm)
    print("||grad(z_n)|| case2:", g2_norm)
    print("max abs diff between grad(z_n):", max_abs_diff)
    print("gradients identical (allclose)?", allclose)

    # show which positions have non-zero grads
    nz1 = (g1.abs().sum(dim=-1) > 1e-8)
    nz2 = (g2.abs().sum(dim=-1) > 1e-8)
    print("Non-zero gradient positions case1 (B x S):\n", nz1.view(B, S))
    print("Non-zero gradient positions case2 (B x S):\n", nz2.view(B, S))

    # param grads comparison (final_proj)
    fp_g1 = grads1["final_proj_grads"]
    fp_g2 = grads2["final_proj_grads"]
    # compute grad norm difference
    def flat_grad_norm(d):
        parts = [v.view(-1) for v in d.values() if v is not None]
        if not parts:
            return 0.0
        return torch.cat(parts).norm().item()
    print("final_proj grad norm case1:", flat_grad_norm(fp_g1))
    print("final_proj grad norm case2:", flat_grad_norm(fp_g2))
    print("final_proj param norm after step case1:", deltas1["final_proj_param_norm"])
    print("final_proj param norm after step case2:", deltas2["final_proj_param_norm"])

    # print perplexities
    print("prob_perplexity case1 (masked):", grads1["prob_perplexity"])
    print("prob_perplexity case2 (masked):", grads2["prob_perplexity"])

if __name__ == "__main__":
    main()
