# train_grpo_qwen.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
"""
citation:
@misc{brown2025grpodemo,
  title={Granular Format Rewards for Eliciting Mathematical Reasoning Capabilities in Small Language Models},
  author={Brown, William},
  howpublished={\url{https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb}},
  date = {2025-01-25},
  note = {GitHub Gist}
}
"""

import re
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Constants
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""



def extract_xml_answer(text: str) -> str:
    """Extract answer from XML format."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from hash format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def get_gsm8k_questions(split: str = "train", print_sample: bool = False) -> Dataset:
    """Load and prepare GSM8K dataset.
    
    Args:
        split: Dataset split (train/test)
        print_sample: Whether to print a sample for comparison
        
    Returns:
        Prepared dataset with prompts and answers
    """
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    
    # Print original data sample
    if print_sample and len(data) > 0:
        print("\n" + "="*80)
        print("原始数据样本:")
        print("="*80)
        print(data[0])
        print("="*80)
    
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    
    # Print processed data sample
    if print_sample and len(data) > 0:
        print("\n处理后的数据样本:")
        print("="*80)
        print(data[0])
        print("="*80 + "\n")
    
    return data  # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function for answer correctness."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for integer answer format."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for strict XML format matching."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function for soft XML format matching."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    """Count XML tags and penalize extra content."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function based on XML tag counting."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]




def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="GRPO training for mathematical reasoning")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="out/Qwen-0.5B-GRPO", help="Output directory")
    parser.add_argument("--run_name", type=str, default="Qwen-0.5B-GRPO-gsm8k", help="Run name for logging")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=2, help="Number of generations per prompt")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (wandb/none)")
    args = parser.parse_args()
    
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = get_gsm8k_questions(print_sample=True)
    
    # Create training config
    print("Creating training configuration...")
    
    # Adjust bf16 based on device
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not use_bf16:
        print("Warning: BF16 not supported, using FP32")
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=use_bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to=args.report_to,
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=.3,
        vllm_device="cuda:0",
    )

    print(f"Loading model: {args.model_name}")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU (training will be very slow)")
    
    # Try to use Flash Attention 2 if available
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    )
 
    
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    main()