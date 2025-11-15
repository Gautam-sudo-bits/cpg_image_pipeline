"""
Visualization module for pipeline stages
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from utils.logger import logger

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.enabled = self.viz_config.get('enabled', True)
        self.save_intermediate = self.viz_config.get('save_intermediate', True)
        
        # Create visualization directory
        self.viz_dir = Path(config.get('paths', {}).get(
            'visualization_dir', 'output/visualizations'
        ))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualizer initialized. Output: {self.viz_dir}")
    
    def visualize_stages(self, stages_dict, output_name="pipeline_stages"):
        """
        Visualize multiple pipeline stages in a grid
        
        Args:
            stages_dict: Dictionary of {stage_name: PIL_Image}
            output_name: Base name for output file
        """
        if not self.enabled:
            return
        
        try:
            logger.info(f"Visualizing {len(stages_dict)} stages...")
            
            num_stages = len(stages_dict)
            cols = min(3, num_stages)
            rows = (num_stages + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if num_stages == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if num_stages > 1 else [axes]
            
            for idx, (stage_name, image) in enumerate(stages_dict.items()):
                ax = axes[idx]
                ax.imshow(image)
                ax.set_title(stage_name, fontsize=12, fontweight='bold')
                ax.axis('off')
            
            # Hide unused subplots
            for idx in range(num_stages, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            output_path = self.viz_dir / f"{output_name}.jpg"
            plt.savefig(output_path, dpi=self.viz_config.get('dpi', 150), 
                       bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
    
    def save_stage(self, image, stage_name, base_name="stage"):
        """
        Save individual stage image
        
        Args:
            image: PIL Image
            stage_name: Name of the stage
            base_name: Base name for file
        """
        if not self.save_intermediate:
            return
        
        try:
            filename = f"{base_name}_{stage_name}.png"
            output_path = self.viz_dir / filename
            image.save(output_path)
            logger.debug(f"Stage saved: {output_path}")
        except Exception as e:
            logger.warning(f"Could not save stage {stage_name}: {str(e)}")
    
    def create_comparison(self, original, generated, output_name="comparison"):
        """
        Create side-by-side comparison
        
        Args:
            original: Original image
            generated: Generated image
            output_name: Output filename
        """
        if not self.enabled:
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(original)
            ax1.set_title('Original', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            ax2.imshow(generated)
            ax2.set_title('Generated', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            
            output_path = self.viz_dir / f"{output_name}.jpg"
            plt.savefig(output_path, dpi=self.viz_config.get('dpi', 150),
                       bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating comparison: {str(e)}")