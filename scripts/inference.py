"""
Inference script for single images.
"""

import argparse
import yaml
import torch
from PIL import Image
import numpy as np
from pathlib import Path

from models.skin_cancer_model import SkinCancerModel
from data.preprocessing import FacePreprocessor
from core import ABCDEExplainer, UncertaintyVisualizer
from core import plot_attention_maps, plot_abcde_scores, plot_uncertainty


def main():
    parser = argparse.ArgumentParser(description='Run inference on image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--explain', action='store_true', help='Generate explanations')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['hardware']['device']
    
    # Create model
    model_config = config['model']
    model = SkinCancerModel(
        efficientnet_variant=model_config['backbone']['efficientnet_variant'],
        image_size=config['data']['image_size'],
        abcde_enabled=model_config['abcde']['enabled'],
        fusion_enabled=model_config['fusion']['enabled'],
        fairness_enabled=model_config['fairness']['enabled'],
        uncertainty_enabled=model_config['uncertainty']['enabled'],
        num_classes=model_config['outputs']['classification'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(args.image_path).convert('RGB')
    image_np = np.array(image)
    
    # Preprocess
    preprocessor = FacePreprocessor(
        detect_faces=True,
        extract_landmarks=True,
    )
    processed = preprocessor.process(image_np)
    processed_image = processed['image']
    
    # Convert to tensor
    image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    image_tensor = (image_tensor - mean) / std
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get predictions
    logits = outputs['logits']
    probabilities = torch.softmax(logits, dim=-1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0, prediction].item()
    
    print(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
    print(f"Confidence: {confidence:.4f}")
    
    # ABCDE scores
    if 'abcde_scores' in outputs:
        abcde_scores = outputs['abcde_scores'][0].cpu().numpy()
        abcde_dict = {
            'A': abcde_scores[0],
            'B': abcde_scores[1],
            'C': abcde_scores[2],
            'D': abcde_scores[3],
            'E': abcde_scores[4],
        }
        print(f"ABCDE Scores: {abcde_dict}")
    
    # Uncertainty
    if 'uncertainty' in outputs:
        uncertainty = outputs['uncertainty']['total'][0].item()
        print(f"Uncertainty: {uncertainty:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate explanations
    if args.explain:
        # ABCDE attention maps
        if 'attention_maps' in outputs:
            explainer = ABCDEExplainer(model)
            explanations = explainer.explain(image_tensor)
            
            # Plot attention maps
            plot_attention_maps(
                processed_image,
                explanations['attention_maps'],
                save_path=str(output_dir / 'attention_maps.png'),
            )
            
            # Plot ABCDE scores
            plot_abcde_scores(
                explanations['abcde_scores'],
                save_path=str(output_dir / 'abcde_scores.png'),
            )
        
        # Uncertainty visualization
        if 'uncertainty' in outputs:
            uncertainty_viz = UncertaintyVisualizer(model)
            uncertainty_dict = uncertainty_viz.visualize_uncertainty(image_tensor)
            
            plot_uncertainty(
                processed_image,
                probabilities[0, 1].item(),
                uncertainty_dict['total'],
                save_path=str(output_dir / 'uncertainty.png'),
            )
    
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()

