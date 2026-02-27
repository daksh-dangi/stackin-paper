import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Callable
from transformers import AutoTokenizer
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from utils import *


class GradRegTrainer:
    def __init__(
        self, 
        policy_model: torch.nn.Module, 
        ref_model: torch.nn.Module, 
        reward_fn: Callable[[List[str], List[str]], List[float]],
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta: float = 0.1,
        lr: float = 1e-5
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        
        self.G = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.epsilon = 10e-3
        self.gamma = 10e-3

        '''
        FSDP setup for multi-GPU training
        need to instantiate this **before** the optimizer, as FSDP flattens and shards params into a 1D tensor
        the optimizer needs to point to this tensor so that it creates the AdamW tracking states
        only for the 1D tensors in *that* shard, rather than for *all* the params

        for the forward pass, the gather of the weights to calculate the next token is all handled by FSDP
        for optimizer.step() the optimizer looks only at the weights in the flattened tensor present on the particular shard 
        it's running on
        for loss.backward(), the gather of the weights is again handled by FSDP and the loss+gradients are calculated, following
        which, the gradients for specific params are passed back to the respective shards
        '''
        setup_distributed()
        local_rank = int(os.environ["LOCAL_RANK"])
        my_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer})
        self.policy_model = FSDP(policy_model, auto_wrap_policy=my_wrap_policy, device_id=local_rank)
        self.ref_model = FSDP(ref_model, auto_wrap_policy=my_wrap_policy, device_id=local_rank)
        
        self.optimizer = AdamW(self.policy_model.parameters(), lr=lr)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Ensure reference model weights are frozen
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        

    def generate_rollouts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        #duplicate each prompt G times before batched generation 
        duplicated_prompt = [p for p in prompts for _ in range(self.G)]

        inputs = self.tokenizer(duplicated_prompt, padding=True, return_tensors="pt").to(self.policy_model.device)

        output_ids = self.policy_model.generate(**inputs, do_sample=True, temperature=0.9)

        prompt_length = inputs.input_ids.shape[1]
        completion_ids = output_ids[:, prompt_length:]
        
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([inputs.attention_mask, completion_mask], dim=1)
        old_log_probs = self.get_per_token_logps(self.policy_model, output_ids, full_attention_mask)

        completion_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        return {
            "prompt_ids": inputs.input_ids,
            "completion_ids": completion_ids,
            "completions_text": completion_text,
            "attention_mask": inputs.attention_mask,
            "old_log_probs": old_log_probs
        }

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        '''
        computing the normalized advantage by subtracting the mean and dividing by std
        rewards tensor is of the shape (batch_size * G)
        '''
        rewards = rewards.reshape(-1, self.G)
        adv = (rewards - rewards.mean(-1, keepdim=True))/(rewards.std(-1, keepdim=True)+1e-8)

        return adv.flatten()

    def get_per_token_logps(self, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''
        pass in the attention mask, so that the internal attention mechanisms know which 
        tokens are padding, and hence need to be skipped and not be attended to
        '''
        output_ids = model(input_ids, attention_mask=attention_mask)
        logits = output_ids.logits
        
        '''
        we need to run .contiguous on the below, because slicing pytorch tensors as done below actually creates a 
        *view* that skips the specified dimensions/elements. because it creates this view, the resulting tensor is
        incontiguous or fragmented. since pytorch performs vectorized memory acceses for operations like softmax,
        this results in issues; which is why we need to use .contiguous()

        these elements are skipped since we don't need the log_probs for the eos_token, and since no token is predicted
        at the 0th time step
        '''
        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(logits, dim=-1)

        '''
        log_probs contains a dimension for each word in the vocab, and labels contains the index of the ground truth token 
        to be predicted; and we only want the log_prob the model attributed to this token which is why we use torch.gather
        '''
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        return log_probs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        completion_ids, attention_mask, adv, prompt_ids = batch['completion_ids'], batch['attention_mask'], batch['advantages'], batch['prompt_ids']
        
        '''
        we don't need to calculate the ratio or intermediate vals for the prompt tokens, so we slice them out to reduce
        unnecessary computation
        '''
        prompt_len = prompt_ids.shape[1]-1
        old_log_probs = batch['old_log_probs'][:, prompt_len:]
        full_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        policy_probs = self.get_per_token_logps(self.policy_model, full_input_ids, attention_mask)[:, prompt_len:]
        
        with torch.no_grad():
            ref_probs = self.get_per_token_logps(self.ref_model, full_input_ids, attention_mask)[:, prompt_len:]

        ratio = torch.exp(policy_probs - old_log_probs)

        '''
        - shape of ratio is (batch_size * G, seq_len), and shape of adv is (batch_size*G)
        - ratio is that shape because in the generate_rollouts function, we generate G rollouts per batch,
        which is the shape of input ids as well, that gets fed to self.get_per_token_logps
        - adv is that shape because rewards are calculated for each trajectory, and there were batch*G
        trajectories
        - here, the overall reward for each trajectory is multiplied to every generated token in the seq
        '''
        val1 = ratio*(adv.unsqueeze(-1))
        val2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv.unsqueeze(-1)

        '''
        takes an element-wise minimum of the two computed values
        we take the negative of this value, as typically pytorch optimizers *minimize* loss, 
        but we instead what to *maximize* the advantage - the inherent objective is different
        hence, rather than changing anything with regards to the optimizer, we just take the negative
        '''
        min_val = -torch.min(val1, val2)

        '''
        rather than calculating the *exact* KL divergence (which would be expensive and time consuming, as it would
        require calculating token-wise differences across the entire trajectory), we use the below approximation 
        introduced in the DeepSeek paper, which serves as a low-variance, high accuracy unbiased estimator

        it is: pi_ref/pi_theta - log(pi_ref/pi_theta) - 1

        and once again, since pytorch optimizers minimize, we multiply the final penalty term by -1
        '''
        diff = ref_probs - policy_probs
        raw_kl_div = torch.exp(diff) - diff - 1.0
        kl_penalty = raw_kl_div * -self.beta

        token_loss = min_val - kl_penalty

        '''
        taking the loss for all the *predicted* tokens, which is why we index starting from the length
        of the prompt - prompt_ids.shape[1]-1; and multiplying it with completion mask
        this is necessary, to ensure we don't take the loss for all the padding tokens generated at the
        end of the squence (which can vary in length)

        the final loss is computed over the average of the sum of every token; to further normalize the loss
        and ensure a smoother training run. if it wasn't taken (i was confused, as the DeepSeek paper had a 
        1/G normalizing term instead of a token level average), then the variable length of each trajectory would
        add a lot of noise, as longer trajectories would have more terms contributing to the loss

        the .sum() function at the end flattens the input (here, it becomes [batch_size * G * seq_len]) before taking
        the sum; and since completion_mask is an array of 0s and 1s, where 0 specifies the ending padding tokens, we
        are able to calculate the overall length of each generated sequence in the batch
        '''
        completion_loss = token_loss
        completion_mask = attention_mask[:, prompt_ids.shape[1]:]

        final_loss = (completion_loss * completion_mask).sum() / completion_mask.sum()
        
        # for logging with tensorboard
        with torch.no_grad():
            mean_surrogate = (min_val * completion_mask).sum() / completion_mask.sum()
            mean_kl = (raw_kl_div * completion_mask).sum() / completion_mask.sum()
            
            metrics_dict = {
                "total_loss": final_loss.item(),
                "surrogate_loss": mean_surrogate.item(),
                "kl_divergence": mean_kl.item()
            }
        
        return final_loss, metrics_dict


    def train_step(self, prompts: List[str], ground_truths: List[str]):
        self.policy_model.train()
        
        with torch.no_grad():
            rollout_data = self.generate_rollouts(prompts)
            completions_text = rollout_data["completions_text"]
            
            # ground_truths duplicated G times to align with the generated completions
            duplicated_truths = [gt for gt in ground_truths for _ in range(self.G)]
            rewards = self.reward_fn(completions_text, duplicated_truths)
            rewards_tensor = torch.tensor(rewards, device=self.policy_model.device)
            
            advantages = self.compute_advantages(rewards_tensor)
            rollout_data["advantages"] = advantages

        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(rollout_data)
        loss.backward()
        
        '''
        below is the implementation of gradient regularization,

        the "penalty" vector generated from subtracting the gradients of the current
        model from the pretrubed model (equation 4 in the paper) isn't a scalar that alters how much the
        LR is affected; it's a directional vector that steers towards the direction where the
        loss landscape is becoming *more* unstable; it points in this direction because 
        we are subtracting this difference (in equation 5), which means the optimizer is actually
        working to actively go the other direction
        '''
        # store current gradients
        g1 = {}
        for name, param in self.policy_model.named_parameters():
            if param.grad is not None:
                g1[name] = param.grad.clone()

        with torch.no_grad():
            for name, param in self.policy_model.named_parameters():
                if is_transformer_block(name) and name in g1:
                    '''
                    temporarily move weights to estimate hessian in the next step 
                    that is, by taking the difference in gradients from the perturbed parameter space
                    '''
                    param.add_(g1[name], alpha=self.epsilon)
                
        loss_perturbed = self.compute_loss(rollout_data)
        self.optimizer.zero_grad()
        loss_perturbed.backward()
        
        with torch.no_grad():
            for name, param in self.policy_model.named_parameters():
                if name in g1:
                    current_grad = g1[name]
                    
                    if is_transformer_block(name):
                        perturbed_grad = param.grad.clone()
                        
                        delta_grad_norm = (perturbed_grad - current_grad) / self.epsilon
                        param.grad = current_grad + (self.gamma / 2.0) * delta_grad_norm
                        
                        # unperturb the weights
                        param.sub_(current_grad, alpha=self.epsilon)
                    
                    else:
                        # not regularizing gradients to embedding/output layers as per GR paper
                        param.grad = current_grad
                
        self.optimizer.step()
        
        return metrics
    
    def evaluate(self, eval_prompts: List[str], eval_ground_truths: List[str], batch_size: int = 8) -> Dict[str, float]:
        # to disable dropout layers
        self.policy_model.eval()
        all_rewards = []
        
        with torch.no_grad():
            for i in range(0, len(eval_prompts), batch_size):
                batch_prompts = eval_prompts[i:i + batch_size]
                batch_truths = eval_ground_truths[i:i + batch_size]
                
                inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to(self.policy_model.device)
                
                # greedy generation, so we disable sampling
                output_ids = self.policy_model.generate(**inputs,do_sample=False,max_new_tokens=512)
                
                prompt_length = inputs.input_ids.shape[1]
                completion_ids = output_ids[:, prompt_length:]
                
                completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                rewards = self.reward_fn(completions_text, batch_truths)
                all_rewards.extend(rewards)
                
        # Pass@1 for eval 
        binary_rewards = [1.0 if r > 0 else 0.0 for r in all_rewards]
        
        return {"eval_pass_at_1": sum(binary_rewards) / len(binary_rewards), "eval_mean_reward": sum(all_rewards) / len(all_rewards)}
    
    '''
    due to frequent OOM issues with GRPO training runs, need to ensure we checkpoint frequently
    '''
    def save_checkpoint(self, output_dir: str, step: int):
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # if policy_model is a huggingFace PreTrainedModel class, we can use the inbuilt save_pretrained function
        if hasattr(self.policy_model, "save_pretrained"):
            self.policy_model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.policy_model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        
        print(f"Checkpoint saved at {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.policy_model.device))
            print(f"Optimizer state restored from {opt_path}")
        else:
            print("No optimizer state found; starting fresh.")