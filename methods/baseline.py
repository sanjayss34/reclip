"""A naive baseline method: just pass the full expression to CLIP."""

from overrides import overrides
from typing import Dict, Any, List
import numpy as np
import torch
import spacy
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L


class Baseline(RefMethod):
    """CLIP-only baseline where each box is evaluated with the full expression."""

    nlp = spacy.load('en_core_web_sm')

    def __init__(self, args: Namespace):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.batch_size = args.batch_size
        self.batch = []

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        if self.batch_size == 1:
            chunk_texts = self.get_chunk_texts(caption)
            probs = env.filter(caption, area_threshold = self.box_area_threshold, softmax=True)
            if self.args.baseline_head:
                probs2 = env.filter(chunk_texts[0], area_threshold = self.box_area_threshold, softmax=True)
                # mask = env.filter_area(self.box_area_threshold)
                probs = L.meet(probs, probs2)
            pred = np.argmax(probs)
            return {
                "probs": probs,
                "pred": pred,
                "box": env.boxes[pred],
            }
        self.batch.append(env.executor.tensorize_inputs(caption, env.image, env.boxes))
        if len(self.batch) % self.batch_size == 0:
            batch_logits_per_image = [[] for _ in self.batch]
            batch_logits_per_text = [[] for _ in self.batch]
            for i, model in enumerate(env.executor.models):
                indices = []
                index = 0
                image_inputs = []
                text_inputs = []
                for images, text in batch:
                    indices.append(list(range(index, index+images[i].shape[0])))
                    image_inputs.append(images[i])
                    text_inputs.append(text)
                    index += images[i].shape[0]
                image_inputs = torch.cat(image_inputs, dim=0)
                text_inputs = torch.cat(text_inputs, dim=0)
                with torch.no_grad():
                    logits_per_image, logits_per_text = model(image_inputs, text_inputs)
                for j in range(len(indices)):
                    batch_logits_per_image[j].append(logits_per_image[indices[j],j:j+1])
                    batch_logits_per_text[j].append(logits_per_text[j:j+1,indices[j]])
            results = []
            for i in range(len(self.batch)):
                all_logits_per_image = torch.stack(batch_logits_per_image[i]).sum(0)
                all_logits_per_text = torch.stack(batch_logits_per_text[i]).sum(0)
                if env.executor.method_aggregator == "max":
                    all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
                elif env.executor.method_aggregator == "sum":
                    all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True)
                logits = all_logits_per_text.view(-1)
                area_filtered_dist = torch.from_numpy(env.filter_area(self.box_area_threshold)).to(env.executor.device)
                logits = torch.where(
                    area_filtered_dist > 0,
                    logits,
                    torch.ones_like(logits)*-10000.
                )
                probs = logits.softmax(dim=-1)
                results.append({
                    "probs": probs.cpu().numpy(),
                    "pred": probs.argmax(dim=-1).item(),
                    "box": env.boxes[probs.argmax(dim=-1).item()]
                })
            self.batch = []
            return results

    def get_chunk_texts(self, expression: str) -> List:
        doc = self.nlp(expression)
        head = None
        for token in doc:
            if token.head.i == token.i:
                head = token
                break
        head_chunk = None
        chunk_texts = []
        for chunk in doc.noun_chunks:
            if head.i >= chunk.start and head.i < chunk.end:
                head_chunk = chunk.text
            chunk_texts.append(chunk.text)
        if head_chunk is None:
            if len(list(doc.noun_chunks)) > 0:
                head_chunk = list(doc.noun_chunks)[0].text
            else:
                head_chunk = expression
        return [head_chunk] + [txt for txt in chunk_texts if txt != head_chunk]
