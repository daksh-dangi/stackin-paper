from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter
import torch
from trainer import *

'''
FSDP clashes with 4-bit quantized models - as it requires floating-point data types when sharding
and bnb converts to integer datatypes. to circumvent this, we use the following bnb config to store
the weights in a floating-point type *container* so that FSDP doesn't throw an error
'''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16
)

log_dir = "./runs/grpo_gr_01"
writer = SummaryWriter(log_dir=log_dir)
global_step = 0
eval_interval = 100
save_interval = 200

trainer = GradRegTrainer(
    policy_model=base_model,
    ref_model=base_model,
    reward_fn=combined_reward_fn,
    group_size=4
)

dataloader, eval_dataset = prepare_datasets("HARDMath.json", split_ratio=0.9, batch_size=4)
for epoch in range(2):
    for batch in dataloader:
        metrics = trainer.train_step(batch["prompts"], batch["ground_truths"])
        global_step += 1
        
        writer.add_scalar("Loss/Total", metrics["total_loss"], global_step)
        writer.add_scalar("Loss/Surrogate_Objective", metrics["surrogate_loss"], global_step)
        writer.add_scalar("Loss/KL_Divergence", metrics["kl_divergence"], global_step)
        
        if global_step % eval_interval == 0:
            metrics = trainer.evaluate(eval_dataset["prompts"], eval_dataset["ground_truths"])
            print(f"Step {global_step} | Eval pass@1: {metrics['eval_pass_at_1']:.4f}")
            
        if global_step % save_interval == 0:
            trainer.save_checkpoint("./outputs", global_step)
            print(f"Step {global_step} | Saving checkpoint")

writer.close()