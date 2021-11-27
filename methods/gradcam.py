"""A naive baseline method: just pass the full expression to CLIP."""

from overrides import overrides
from typing import Dict, Any, List
import numpy as np
import torch
import spacy
from argparse import Namespace

from .ref_method import RefMethod
from lattice import Product as L
import clip_gradcam as gradcam
import clip


class Gradcam(RefMethod):
    """CLIP-only baseline where each box is evaluated with the full expression."""

    nlp = spacy.load('en_core_web_sm')

    def __init__(self, args: Namespace):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.alpha = args.gradcam_alpha

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        model = env.executor.models[0]
        preprocess = env.executor.preprocesses[0]
        image = env.image
        image_input = preprocess(image).unsqueeze(0).to(env.executor.device)
        text_input = clip.tokenize(caption).to(env.executor.device)
        image_np = gradcam.load_np_image(image, model.visual.input_resolution)
        saliency_layer = "layer4"
        attn_map = gradcam.gradCAM(
            model.visual,
            image_input,
            model.encode_text(text_input).float(),
            getattr(model.visual, saliency_layer)
        ).view(image_np.shape[0], image_np.shape[1])
        # attn_map = gradcam.normalize(attn_map)
        # print(attn_map.min(), attn_map.max())
        pred = None
        max_score = -np.inf
        scores = []
        alpha = self.alpha
        for box_index, box in enumerate(env.boxes):
            if (box.right-box.left)*(box.bottom-box.top)/(image.width*image.height) < self.box_area_threshold:
                continue
            score = attn_map[int(box.top):int(box.bottom),int(box.left):int(box.right)].sum()/box.area**alpha
            if score > max_score:
                max_score = score
                pred = box_index
            scores.append(score)
        probs = torch.Tensor(scores).to(env.executor.device).softmax(dim=-1).tolist()
        assert pred is not None
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
        }
