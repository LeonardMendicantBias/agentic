import torch
import torch.nn.functional as F

from transformers import Trainer

from torch.utils.data import Dataset


class DistillationTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]

        # ---- Student forward ----
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # ---- Teacher forward (no grad) ----
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # ---- CE loss ----
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # ---- KL distillation ----
        T = self.temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        loss_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (T ** 2)

        # ---- Final loss ----
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl

        return (loss, student_outputs) if return_outputs else loss
    

class VisionTextDataset(Dataset):
    def __init__(self, data, processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = sample["image"]        # PIL.Image
        prompt = sample["prompt"]      # str
        answer = sample["answer"]      # str

        # Construct chat-style input
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt"
        )

        inputs = self.processor(
            images=image,
            text=inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # labels = input_ids (causal LM)
        inputs["labels"] = inputs["input_ids"].clone()

        return {k: v.squeeze(0) for k, v in inputs.items()}