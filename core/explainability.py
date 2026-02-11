"""
All explainability methods: Grad-CAM, ABCDE attention, and uncertainty visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class GradCAM:
    """Grad-CAM for visualizing attention."""
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output['logits'].argmax(dim=1).item()
        
        self.model.zero_grad()
        output['logits'][0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()


class ABCDEExplainer:
    """Explainer for ABCDE-specific attention maps."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
    
    def explain(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate ABCDE attention maps."""
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        attention_maps = {}
        if 'attention_maps' in outputs:
            for criterion, att_map in outputs['attention_maps'].items():
                att_map_np = att_map[0, 0].cpu().numpy()
                attention_maps[criterion] = att_map_np
        
        abcde_scores = {}
        if 'abcde_scores' in outputs:
            scores = outputs['abcde_scores'][0].cpu().numpy()
            criterion_names = ['A', 'B', 'C', 'D', 'E']
            for name, score in zip(criterion_names, scores):
                abcde_scores[name] = float(score)
        
        return {
            'attention_maps': attention_maps,
            'abcde_scores': abcde_scores,
        }


class UncertaintyVisualizer:
    """Visualizer for uncertainty estimates."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
    
    def visualize_uncertainty(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate uncertainty visualizations."""
        with torch.no_grad():
            outputs = self.model(input_tensor, return_uncertainty=True)
        
        uncertainty_dict = {}
        if 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty']
            uncertainty_dict = {
                'epistemic': uncertainty['epistemic'][0].cpu().numpy(),
                'aleatoric': uncertainty['aleatoric'][0].cpu().numpy(),
                'total': uncertainty['total'][0].cpu().numpy(),
            }
        
        if 'logits' in outputs:
            logits = outputs['logits'][0]
            probabilities = torch.softmax(logits, dim=-1)
            uncertainty_dict['prediction'] = probabilities.cpu().numpy()
        
        return uncertainty_dict

