# import debugpy

# # Listen on all interfaces, port 5678
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for VS Code debugger to attach...")
# debugpy.wait_for_client()

import torch
import torch.nn as nn
import torch.nn.functional as F
from NN_framework import NN_module
from torch.optim import SGD, Adam, AdamW


import numpy as np
# from Dataloader import Preles_data
# -----------------------------
# Noisy Top-K Gating with exploration scheduling
# -----------------------------
class NoisyTopKGating(nn.Module):
    '''
    Noisy Top-K Gating as in "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    https://arxiv.org/abs/1701.06538'''
    ''' depending on the input x, it selects the top k experts (out of num_experts) to process the input.
    It adds noise to the gating logits to encourage exploration of different experts during training.
    The noise level can be adjusted during training using the set_noise_mult method.
    Additionally, a specific expert (e.g., a hard expert) can be down-weighted using the set_hard_expert and set_hard_expert_scale methods.
    '''
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int,
        init_noise_scale: float = 1.0,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k #should be not less than 2

        self.w_gate = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Linear(input_dim, num_experts)

        # runtime-scaled noise (will be multiplied by a schedule factor)
        self.register_buffer("base_noise_scale", torch.tensor(float(init_noise_scale)))

        # runtime scalars controlled by scheduler hooks each step/epoch
        self.register_buffer("noise_scale_mult", torch.tensor(1.0))     # exploration knob
        self.register_buffer("hard_expert_scale", torch.tensor(1.0))    # down-weight hard expert early
        self.hard_expert_id = None

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        # probs: (B, E)
        return -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean()
#calculate entropy 

    def forward(self, x: torch.Tensor, anneal_noise: bool = True):
        """
        Returns:
            weights: (B, E) softmax over experts but only top-k nonzero
            entropy: scalar (mean over batch) of clean softmax entropy (for aux loss)
            Gating= softmax(Topk(Wx + noise*noise_std*softplus(W'x)))
        """
        clean_logits = self.w_gate(x)                    # (B, E) 
        clean_probs = F.softmax(clean_logits, dim=-1)    # (B, E) W*x (gating logits without noise)
        entropy = self._entropy(clean_probs)
        entropy_avag= self._entropy(clean_probs)
        
        # noise
        # noise_std = F.softplus(self.w_noise(x))          # (B, E) noisy part W'*x
        # noise = torch.randn_like(clean_logits)           # (B, E) ~ N(0,1)
        # entropy = self._entropy(clean_probs).mean(axis=0)
        # entropy_avag= self._entropy(clean_probs).sum(dim=-1).mean()
        
        if anneal_noise:
            noise_std = F.softplus(self.w_noise(x))   # (B, E)
            noise     = torch.randn_like(clean_logits)
            scale     = self.base_noise_scale * self.noise_scale_mult
            noisy_logits    = clean_logits + noise * noise_std * scale
        else:
            noisy_logits    = clean_logits                  # <- no noise at eval

        # optionally down-weight a specific expert (e.g., hard expert) at runtime
        if self.hard_expert_id is not None and self.hard_expert_scale.item() != 1.0:
            # subtract log-scale to reduce its logit; multiply weight in prob-space ~= add log(scale)
            # scale == 0 masks with -inf, so the weight is exactly 0 even when top_k == E
            # (1 trainable expert): log(1e-12) left ~1e-12 weight and `weights > 0` then
            # counted the masked hard expert as used on every sample during warmup.
            if self.hard_expert_scale.item() == 0.0:
                fudge = torch.tensor(float("-inf"), device=noisy_logits.device)
            else:
                fudge = torch.log(self.hard_expert_scale)
            noisy_logits = noisy_logits.clone()
            noisy_logits[:, self.hard_expert_id] = noisy_logits[:, self.hard_expert_id] + fudge #scale down the hard expert (usufull during warmup)

        # top-k masking
        topk_vals, topk_idx = torch.topk(noisy_logits, k=self.top_k, dim=-1) ##pick top_k_experts
        mask = torch.full_like(noisy_logits, float("-inf")) # (B, E) -inf everywhere
        mask.scatter_(1, topk_idx, topk_vals) # fill only top-k positions with their noisy logits

        weights = F.softmax(mask, dim=-1)                # only top-k nonzero
        return weights, entropy, entropy_avag

    # -------- Schedulers you can call from your loop --------
    def set_noise_mult(self, value: float):
        """Set exploration strength (1.0 early, -> 0.0 later).""" 'leads to annealing of the noise'
        val = float(value)
        # keep dtype/device of registered buffer
        self.noise_scale_mult.data.fill_(val)

    def set_hard_expert(self, expert_id: int | None):
        self.hard_expert_id = expert_id

    def set_hard_expert_scale(self, value: float):
        """
        Down-weight the hard expert in softmax space.
        value in [0,1]. 1 = normal, 0 = effectively disabled. 'scales hard expert involvement during the training
        """
        val = float(value)
        self.hard_expert_scale.data.fill_(val)


# -----------------------------
# Mixture-of-Experts wrapper
# -----------------------------
class NoisyMixture(nn.Module):
    def __init__(
        self,
        input_dim :int,
        output_dim: int,
        trainable_experts:list[nn.Module], #experts which should be trained
        top_k: int = 2,
        init_noise_scale: float = 1.0,
        hard_expert_id: int | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.experts = nn.ModuleList(trainable_experts)
        self.hard_expert_id = hard_expert_id

        gate_in_dim = input_dim
        #NoisyTopKGating instance selects the experts based on the input
        self.gate = NoisyTopKGating(
            input_dim=gate_in_dim,
            num_experts=len(trainable_experts)+1,
            top_k=top_k,
            init_noise_scale=init_noise_scale,
        )
        self.gate.set_hard_expert(hard_expert_id)

        # Slot for per-batch precomputed outputs of the hard expert
        self._precomputed = None  # tensor (B, D_out)

    # ------------- public API -------------
    def set_precomputed_outputs(self, tensor: torch.Tensor | None):
        """
        Provide precomputed outputs for the current batch for the hard expert.
        Shape: (B, output_dim). Use None to clear.
        """
        self._precomputed = tensor

    def set_noise_mult(self, value: float):
        self.gate.set_noise_mult(value)

    def set_hard_expert_scale(self, value: float):
        self.gate.set_hard_expert_scale(value)

    def forward(self, x: torch.Tensor, anneal_noise: bool = True):
        """
        x is expected as (B, gate_in_dim). If your original x is (B, seq_len, input_dim),
        flatten before calling or adapt the Linear dims in the gate.
        Returns:
            output: (B, D_out)
            expert_usage: (E,) number of samples with nonzero weight (soft count)
            entropy: scalar (for optional aux loss)
        """
        weights, entropy, entropy_avag = self.gate(x, anneal_noise=anneal_noise)  # (B, E) #weights of the experts, entropy 
        #tells you how uncertain the gating is and how the experts are spreaded
        B = x.size(0)
        E = len(self.experts)

        out = torch.zeros(B, self.output_dim, device=x.device)
        # soft count of how many samples used each expert (nonzero weight in top-k)
        expert_usage = (weights > 0).float().sum(dim=0)  # (E,)

        # Iterate over the GATE's columns (trainable experts + the hard-expert slot), not
        # over `self.experts`: that list holds only the trainable experts, so it is one
        # shorter than the gate and its indices shift past `hard_expert_id`. Enumerating
        # it directly (noisy_mixture.py) dropped the last gate column's weight and fed
        # the hard-expert column the wrong expert — with one trainable expert and
        # hard_expert_id=1, PRELES never contributed at all.
        for e_id in range(weights.size(1)):
            mask = weights[:, e_id] > 0 # (B,) which samples use this expert
            if mask.any():
                w = weights[mask, e_id].unsqueeze(1)  # (N,1)

                if e_id == self.hard_expert_id:
                    #if preles is called the output is precomputed
                    if self._precomputed is None:
                        # no gradients flow into hard expert by design
                        raise RuntimeError("Precomputed outputs not set for hard expert in this batch.")
                    with torch.no_grad():
                        e_out = self._precomputed[mask]  # (N, D_out)
                else:
                    #otherwise the expert is computed; gate column -> trainable-expert index
                    trainable_id = e_id if e_id < self.hard_expert_id else e_id - 1
                    e_out = self.experts[trainable_id](x[mask])  # (N, D_out) output of a single expert

                out[mask] += e_out * w #outout is weighted sum of the experts outputs

        # Mean routing weight per expert over the batch: the share of the output each
        # expert was given, from the ACTUAL weights (after noise, hard-expert scaling and
        # top-k), not the clean probabilities. Replaces the second return slot, which
        # was the same scalar entropy returned twice.
        routing_share = weights.mean(dim=0)  # (E,)

        return out, expert_usage, routing_share, entropy_avag

# -----------------------------
# Simple utilities for scheduling from your training loop
# -----------------------------
def linear_schedule(step: int, total: int, start: float, end: float) -> float:
    """Linearly interpolate from start -> end over [0, total]."""
    if total <= 0:
        return end
    t = min(max(step, 0), total)
    return float(start + (end - start) * (t / total))

def get_noise_mult(step, total_steps, start, end, reset_step=None, reset_value=None, reset_end_step=None):
    """
    Before `reset_step`: anneal start -> end over `total_steps`.
    From `reset_step`: restart at `reset_value` and reach `end` AT step `reset_end_step`,
    then hold. In noisy_mixture.py the restarted anneal ran over another full
    `total_steps`, so it never reached `end` within training.
    """
    if reset_step is not None and step >= reset_step:
        # restart annealing after expert addition
        return linear_schedule(step=step - reset_step,
                               total=reset_end_step - reset_step,
                               start=reset_value, end=end)
    else:
        frac = min(step / total_steps, 1.0)
        return start + (end - start) * frac
    
class MixtureModel(nn.Module):
    def __init__(self,config, model_configs,total_train_budget, hard_expert_id=None, warmup_steps=1000, 
                 ramp_steps=2000, noise_steps=3000, alpha_entropy=0.001,
                 init_noise_scale=1, top_k=2):    
        super(MixtureModel,self).__init__()
        # general configs
        self.config=config
        # Schedule points, in optimizer steps, with warmup <= ramp <= noise <= budget:
        #   warmup_steps: hard expert masked before this step
        #   ramp_steps:   hard-expert weight ramps 0.2 -> 1.0 until this step
        #   noise_steps:  gate noise (restarted at 1.0 after warmup) reaches 0.2 at this step
        # All three are points, matching how the drivers sample them as fractions.
        self.warmup_steps =warmup_steps[0] if isinstance(warmup_steps, (tuple, list)) else warmup_steps
        self.ramp_steps=ramp_steps[0] if isinstance(ramp_steps, (tuple,list)) else ramp_steps
        self.noise_steps=noise_steps[0] if isinstance(noise_steps, (tuple,list)) else noise_steps
        self.alpha_entropy= alpha_entropy[0] if isinstance(alpha_entropy, (tuple,list)) else alpha_entropy
        self.global_step=0
        self.total_budget = total_train_budget
        #seed
        torch.manual_seed(self.config['training::seed'])
        torch.cuda.manual_seed(self.config['training::seed'])
        #### cuda ####
        if self.config['training::cuda'] == True:
            if  torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
 
        trainable_experts=[]
        for m in model_configs:
            config_model=m
            model = NN_module(config_model)
            trainable_experts.append(model)
        
        #set experts and hard expert id
        self.experts=trainable_experts.copy()
        self.experts.insert(hard_expert_id,'Preles')
        print('number of experts:',len(self.experts))
        self.hard_expert_id = hard_expert_id if hard_expert_id is not None else len(model_configs)
        
        
        #Mixture of experts model
        model_mixture=NoisyMixture(
        input_dim=self.config['model::input_size'],
        output_dim=self.config['model::output_size'],
        trainable_experts=trainable_experts,  # only trainable experts; hard expert is handled via precomputed outputs
        top_k=top_k,
        init_noise_scale=init_noise_scale,
        hard_expert_id=self.hard_expert_id,)

        self.model_mixture=model_mixture

        self.model_mixture.to(self.device)

        #optimizer
        if self.config['training::optimizer']=='SGD':
                self.optimizer = SGD(self.model_mixture.parameters(), 
                                    lr=self.config['training::learning_rate'],
                                    momentum=self.config['training::momentum'],
                                    weight_decay=self.config['training::weight_decay'], 
                                    dampening=self.config['training::dampening'])
            
        elif self.config['training::optimizer']=='Adam':
            self.optimizer = Adam(self.model_mixture.parameters(), 
                                lr=self.config['training::learning_rate'],
                                betas= (self.config['training::beta_1'],
                                        self.config['training::beta_2']),
                                weight_decay=self.config['training::weight_decay'])
        elif self.config['training::optimizer']=='AdamW':
            self.optimizer = AdamW(self.model_mixture.parameters(), 
                                lr=self.config['training::learning_rate'],
                                betas= (self.config['training::beta_1'],
                                        self.config['training::beta_2']),
                                weight_decay=self.config['training::weight_decay'])
        else:
            raise NotImplementedError(f"Optimizer '{self.config['training::optimizer']}' not implemented.")
        
        #loss and metrics
        if self.config['training::loss_type'] == 'MSE':
            self.loss_fn= nn.MSELoss().to(self.device)
        elif self.config['training::loss_type'] == 'MAE':
            self.loss_fn= nn.L1Loss().to(self.device)

        #metrics
        if self.config['training::acc_metric'] == 'MAE':
            self.metric_ = nn.L1Loss().to(self.device) 
        elif self.config['training::acc_metric'] ==  'MSE':
            self.metric_= nn.MSELoss().to(self.device)

    def forward(self, x):
        # compute model prediction
        y = self.model_mixture(x)
        return y
    
    def train_epoch(self,train_loader):
        loss_per_epoch = 0.0
        self.model_mixture.train()
        usage_avag=np.zeros(len(self.experts))
        entropy_avag= 0.0
        share_avag=np.zeros(len(self.experts))  # mean routing weight per expert
        loss_per_epoch = 0.0
        metric_per_epoch = 0.0
        for x_batch, y_batch, y_preles_batch in train_loader:
            # The dataset picks its device on its own (GPU whenever one exists), which
            # need not match `training::cuda`.
            x_batch, y_batch, y_preles_batch = (t.to(self.device) for t in (x_batch, y_batch, y_preles_batch))
            B = x_batch.size(0)

            # 1) Provide precomputed outputs for THIS batch (only for hard expert)
            # shape must be (B, OUT_DIM)
            precomp =y_preles_batch  # your function
            self.model_mixture.set_precomputed_outputs(precomp)

            # 2) Schedules
            if self.global_step < self.warmup_steps:
                # hard expert OFF during warm-up
                #print(self.warmup_steps)
                hard_scale = 0.0
            else:
                # gradually allow the hard expert to contribute
                if self.global_step == self.warmup_steps:
                    print("Hard expert enabled")
                # `ramp_steps` is the step where the ramp ENDS (a point, like the
                # searched fraction), not its length: the ramp spans
                # [warmup_steps, ramp_steps]. As a length it could run past the end
                # of training, so PRELES never reached full weight.
                hard_scale = linear_schedule(
                    step=self.global_step -  self.warmup_steps,
                    total=  self.ramp_steps - self.warmup_steps,
                    start=0.2,    # still under-weighted at start of ramp
                    end=1.0       # fully enabled after ramp
                )
            self.model_mixture.set_hard_expert_scale(hard_scale)

            # encourage exploration early; anneal to 0.2. After warmup the noise restarts
            # at 1.0 and reaches 0.2 AT `noise_steps` (previously never read), then holds.
            noise_mult = get_noise_mult(self.global_step, self.total_budget,
                                        start=1.5, end=0.2, reset_step=self.warmup_steps,
                                        reset_value=1.0, reset_end_step=self.noise_steps)
            self.model_mixture.set_noise_mult(noise_mult)

            # 3) Forward + loss
            pred, usage, share, entropy_av =self.model_mixture(x_batch)
            usage_avag += usage.detach().cpu().numpy()

            #loss
            main_loss = self.loss_fn(pred, y_batch)
            # Entropy term only once the hard expert is in the mixture. `+alpha` sharpens
            # the gate, and during warmup the entropy still spans the masked PRELES
            # column, so it made the gate commit against PRELES before PRELES could be
            # tried (synthetic test: PRELES share 0.53±0.31 vs 0.83±0.05 when gated).
            alpha = self.alpha_entropy if self.global_step >= self.warmup_steps else 0.0
            loss = main_loss + alpha * entropy_av
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_per_epoch += main_loss.item()

            entropy_avag += entropy_av.item()
            share_avag += share.detach().cpu().numpy()
            #metric
            metric = self.metric_(pred, y_batch)
            metric_per_epoch += metric.item()

            self.global_step += 1

        # 5th value: mean routing weight per expert (was a per-expert array that held the
        # same scalar entropy in every slot).
        return loss_per_epoch/len(train_loader), metric_per_epoch/len(train_loader), usage_avag/len(train_loader), entropy_avag/len(train_loader), share_avag/len(train_loader)
    
    @torch.no_grad()  # evaluation only: no autograd graphs for test/val batches
    def test_epoch(self, test_loader):
        self.model_mixture.eval()
        test_loss_per_epoch, metric_per_epoch = 0.0, 0.0
        usage_avag=np.zeros(len(self.experts))
        share_avag=np.zeros(len(self.experts))  # mean routing weight per expert
        entropy_avag = 0.0
        for x_batch, y_batch, y_preles_batch in test_loader:
            x_batch, y_batch, y_preles_batch = (t.to(self.device) for t in (x_batch, y_batch, y_preles_batch))
            B = x_batch.size(0)

            # 1) Provide precomputed outputs for THIS batch (only for hard expert)
            # shape must be (B, OUT_DIM)
            precomp =y_preles_batch  # your function
            self.model_mixture.set_precomputed_outputs(precomp)

            # 3) Forward + loss
            pred, usage, share, entropy_av = self.model_mixture(x_batch, anneal_noise=False)
            main_loss = self.loss_fn(pred, y_batch)
            metric = self.metric_(pred, y_batch)  
            metric_per_epoch += metric.item()
            # loss = main_loss + self.alpha_entropy * entropy  # optional aux term
            test_loss_per_epoch += main_loss.item()

            usage_avag += usage.detach().cpu().numpy()
            share_avag += share.detach().cpu().numpy()
            entropy_avag +=entropy_av.item()


        # 5th value: mean routing weight per expert (see train_epoch).
        return test_loss_per_epoch/len(test_loader), metric_per_epoch/len(test_loader), usage_avag/len(test_loader), entropy_avag/len(test_loader), share_avag/len(test_loader)

if __name__ == "__main__":
    # Self-contained smoke test on synthetic batches: no data, calibration or debugger.
    # Checks the routing (output == gate-weighted sum of experts + precomputed PRELES)
    # and runs the warmup -> ramp -> noise schedule through a short training run.
    from Config_Files.config_spatial_mixture import build_config

    config = build_config()
    config.update({"training::cuda": False, "model::layer_number": 2, "model::hidden_size": [16],
                   "model::activation": ["Relu", None], "model::bias_list": [True, True]})
    in_dim, out_dim = config["model::input_size"], config["model::output_size"]

    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, in_dim, generator=g)
    y_preles = torch.randn(512, out_dim, generator=g)   # stands in for the precomputed PRELES output
    y = torch.where((x[:, :1] > 0), y_preles, torch.sin(x[:, 1:1 + out_dim]))
    loader = [(x[i:i + 32], y[i:i + 32], y_preles[i:i + 32]) for i in range(0, len(x), 32)]

    num_epochs = 20
    budget = len(loader) * num_epochs
    hard_expert_id = config["hard_expert_id"]
    mix_model = MixtureModel(config=config, model_configs=[dict(config), dict(config)],
                             total_train_budget=budget, hard_expert_id=hard_expert_id,
                             warmup_steps=int(0.3 * budget), ramp_steps=int(0.5 * budget),
                             noise_steps=int(0.7 * budget), alpha_entropy=config["alpha_entropy"])

    with torch.no_grad():
        mm = mix_model.model_mixture.eval()
        mm.set_precomputed_outputs(y_preles)
        out, _, _, _ = mm(x, anneal_noise=False)
        w, _, _ = mm.gate(x, anneal_noise=False)
        ref = sum(w[:, [e]] * (y_preles if e == hard_expert_id
                               else mm.experts[e if e < hard_expert_id else e - 1](x))
                  for e in range(w.size(1)))
        assert torch.allclose(out, ref, atol=1e-6), "routing != gate-weighted sum of experts"
    print("routing check passed")

    for epoch in range(num_epochs):
        train_loss, train_metric, train_usage, train_entropy, train_share = mix_model.train_epoch(loader)
        test_loss, test_metric, test_usage, test_entropy, test_share = mix_model.test_epoch(loader)
        print(f"Epoch {epoch+1}/{num_epochs}  train {train_loss:.4f}  test {test_loss:.4f}  "
              f"gate entropy {test_entropy:.3f}  routing share {np.round(test_share, 3)}  usage {np.round(test_usage, 1)}")
