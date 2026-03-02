#!/usr/bin/env python3
"""
WPA National Park Style Image Converter

Converts photos into WPA (Works Progress Administration) National Park poster style.
Inspired by the iconic 1935-1943 National Park Service posters with their
simplified realism, bold shapes, flat color planes, and hand-drawn optimism.

Based on the authentic WPA poster visual DNA:
- Heroic central subject with layered depth
- Restricted, harmonized palette (6-12 colors, earthy/muted tones)
- Flat color blocks with atmospheric perspective via value
- Bold geometric shapes with selective detailing
- Directional high-contrast lighting with uniform shadow blocks
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Tuple, List
import sys


def apply_earthy_palette(image: Image.Image, earthiness: float = 0.4) -> Image.Image:
    """
    Shift colors toward earthy, muted WPA tones: ochres, dusty blues, sage greens, red-clays, charcoal purples.
    
    Shadows lean cool; highlights lean warm.
    """
    img_array = np.array(image, dtype=np.float32)
    
    # Earthy tone adjustments:
    # - Reduce pure blues, shift toward dusty blue/teal
    # - Mute bright greens toward sage
    # - Warm up highlights (add ochre/yellow)
    # - Cool down shadows (add purple/blue)
    
    # Calculate luminance for shadow/highlight detection
    luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    is_shadow = luminance < 128
    is_highlight = luminance > 180
    
    # Shift shadows toward cool (blue/purple)
    shadow_factor = earthiness * 0.15
    img_array[is_shadow, 0] = img_array[is_shadow, 0] * (1.0 - shadow_factor * 0.3)  # Reduce red in shadows
    img_array[is_shadow, 2] = img_array[is_shadow, 2] * (1.0 + shadow_factor * 0.2)  # Enhance blue in shadows
    
    # Shift highlights toward warm (ochre/yellow)
    highlight_factor = earthiness * 0.1
    img_array[is_highlight, 0] = img_array[is_highlight, 0] * (1.0 + highlight_factor * 0.1)  # Warm highlights
    img_array[is_highlight, 1] = img_array[is_highlight, 1] * (1.0 + highlight_factor * 0.05)
    
    # Mute overall saturation slightly for earthy feel
    # Reduce pure colors, shift toward earth tones
    saturation_reduction = earthiness * 0.1
    gray = np.mean(img_array, axis=2, keepdims=True)
    img_array = img_array * (1.0 - saturation_reduction) + gray * saturation_reduction
    
    # Clip to valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def apply_atmospheric_perspective(image: Image.Image, strength: float = 0.6) -> Image.Image:
    """
    Create atmospheric perspective: lighter, grayer, less saturated backgrounds;
    stronger saturation and darker values in foreground.
    
    Assumes top portion is background (sky/distant mountains) and bottom is foreground.
    """
    img_array = np.array(image, dtype=np.float32)
    h, w = img_array.shape[:2]
    
    # Create vertical gradient: top (background) to bottom (foreground)
    y_positions = np.arange(h, dtype=np.float32)
    # Normalize to 0-1, with 0 at top (background) and 1 at bottom (foreground)
    gradient = y_positions / h
    
    # For each row, apply atmospheric perspective
    for y in range(h):
        factor = gradient[y]  # 0 at top, 1 at bottom
        
        # Background (top): lighter, grayer, less saturated
        # Foreground (bottom): darker, more saturated
        if factor < 0.5:  # Background region
            # Lighten and desaturate
            lighten = (0.5 - factor) * strength * 0.3  # Lighten up to 15%
            desaturate = (0.5 - factor) * strength * 0.4  # Desaturate up to 20%
            
            # Lighten
            img_array[y, :, :] = img_array[y, :, :] * (1.0 + lighten)
            
            # Desaturate by mixing with gray
            gray = np.mean(img_array[y, :, :], axis=1, keepdims=True)
            img_array[y, :, :] = img_array[y, :, :] * (1.0 - desaturate) + gray * desaturate
            
            # Shift toward cooler tones (distant = cooler)
            cool_shift = (0.5 - factor) * strength * 0.15
            img_array[y, :, 0] = img_array[y, :, 0] * (1.0 - cool_shift * 0.2)  # Reduce red
            img_array[y, :, 2] = img_array[y, :, 2] * (1.0 + cool_shift * 0.15)  # Enhance blue
        else:  # Foreground region
            # Darken slightly and enhance saturation
            darken = (factor - 0.5) * strength * 0.1  # Darken up to 5%
            saturate = (factor - 0.5) * strength * 0.2  # Saturate up to 10%
            
            # Darken
            img_array[y, :, :] = img_array[y, :, :] * (1.0 - darken)
            
            # Enhance saturation
            gray = np.mean(img_array[y, :, :], axis=1, keepdims=True)
            img_array[y, :, :] = img_array[y, :, :] * (1.0 + saturate) - gray * saturate
    
    # Clip to valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def create_uniform_shadows(image: Image.Image, shadow_strength: float = 0.4) -> Image.Image:
    """
    Create uniform shadow blocks (not gradients) for WPA style.
    Shadows are a uniform fill color, not softly blended.
    """
    # Convert to grayscale to detect shadows
    gray = image.convert('L')
    gray_array = np.array(gray, dtype=np.float32)
    
    # Identify shadow regions (darker areas)
    # Use adaptive threshold to find shadows relative to local brightness
    from scipy import ndimage
    local_mean = ndimage.uniform_filter(gray_array, size=20)
    shadow_mask = gray_array < (local_mean * 0.7)  # Areas darker than 70% of local mean
    
    # Create uniform shadow color by averaging shadow pixels
    img_array = np.array(image, dtype=np.float32)
    if shadow_mask.sum() > 0:
        shadow_color = img_array[shadow_mask].mean(axis=0)
        
        # Shift shadow color toward cool (blue/purple) as per WPA style
        shadow_color[0] = shadow_color[0] * 0.85  # Reduce red
        shadow_color[2] = shadow_color[2] * 1.15  # Enhance blue
        
        # Apply uniform shadow fill
        shadow_factor = shadow_strength
        img_array[shadow_mask] = img_array[shadow_mask] * (1.0 - shadow_factor) + shadow_color * shadow_factor
    
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def preserve_silhouettes(image: Image.Image, edge_preservation: float = 0.3) -> Image.Image:
    """
    Preserve silhouette clarity while simplifying detail.
    Uses edge-aware smoothing to keep shape boundaries crisp.
    """
    # Apply moderate blur to simplify detail
    blurred = image.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Detect edges
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges, dtype=np.float32) / 255.0
    
    # Blend original and blurred based on edges
    # Keep original near edges, use blurred elsewhere
    img_array = np.array(image, dtype=np.float32)
    blurred_array = np.array(blurred, dtype=np.float32)
    
    # Edge mask: 1.0 at edges, 0.0 elsewhere
    edge_mask = np.clip(edge_array * 10, 0, 1)  # Amplify edge signal
    edge_mask = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
    
    # Blend: more original at edges, more blurred in flat areas
    blend_factor = edge_preservation * edge_mask
    result = img_array * blend_factor + blurred_array * (1.0 - blend_factor)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def quantize_colors_flat(image: Image.Image, n_colors: int = 8) -> Image.Image:
    """Reduce image to flat color blocks (silkscreen/stencil effect)."""
    # Use median cut quantization without dithering for flat blocks
    quantized = image.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    
    # Convert back to RGB
    quantized = quantized.convert('RGB')
    
    return quantized


def create_flat_color_blocks(image: Image.Image, block_size: int = 6) -> Image.Image:
    """
    Create flat color blocks by averaging regions (stencil effect).
    Smaller blocks than before to preserve more shape detail.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Downsample by averaging blocks
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    if h_blocks == 0 or w_blocks == 0:
        return image
    
    # Create block-averaged version
    blocks = img_array[:h_blocks*block_size, :w_blocks*block_size].reshape(
        h_blocks, block_size, w_blocks, block_size, 3
    ).mean(axis=(1, 3)).astype(np.uint8)
    
    # Upsample back to original size using nearest neighbor (creates flat blocks)
    block_img = Image.fromarray(blocks)
    result = block_img.resize((w, h), Image.Resampling.NEAREST)
    
    return result


def enhance_contrast_directional(image: Image.Image, factor: float = 1.6) -> Image.Image:
    """Enhance contrast for bold, graphic look with directional lighting feel."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_wpa_style(
    image: Image.Image,
    n_colors: int = 8,
    simplification: float = 0.6,
    earthiness: float = 0.4,
    contrast: float = 1.6,
    saturation: float = 1.2,
    block_size: int = 6,
    atmospheric_perspective: float = 0.6,
    shadow_strength: float = 0.4,
    edge_preservation: float = 0.3,
) -> Image.Image:
    """
    Apply authentic WPA National Park poster style to an image.
    
    Follows the visual DNA of 1935-1943 WPA posters:
    - Simplified realism with preserved silhouettes
    - Bold shapes with selective detailing
    - Flat color planes (6-12 colors, earthy/muted tones)
    - Atmospheric perspective via value steps
    - Directional high-contrast lighting with uniform shadow blocks
    
    Args:
        image: Input PIL Image
        n_colors: Number of colors in palette (6-12, default 8)
        simplification: How much to simplify detail (0.0-1.0, default 0.6)
        earthiness: Shift toward earthy/muted tones (0.0-1.0, default 0.4)
        contrast: Contrast enhancement factor (1.0-2.0, default 1.6)
        saturation: Color saturation factor (0.8-1.5, default 1.2)
        block_size: Size of flat color blocks (4-10, default 6)
        atmospheric_perspective: Strength of depth via value (0.0-1.0, default 0.6)
        shadow_strength: Strength of uniform shadow blocks (0.0-1.0, default 0.4)
        edge_preservation: How much to preserve edge detail (0.0-1.0, default 0.3)
    
    Returns:
        Stylized PIL Image
    """
    result = image.copy()
    
    # Step 1: Preserve silhouettes while simplifying detail
    # Use edge-aware smoothing to keep shape boundaries crisp
    result = preserve_silhouettes(result, edge_preservation)
    
    # Step 2: Apply moderate blur for simplification (less aggressive than before)
    blur_radius = int(2 + simplification * 4)  # 2-6 pixels (was 8-20)
    result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Step 3: Enhance saturation moderately (WPA is vibrant but not oversaturated)
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(saturation)
    
    # Step 4: Apply earthy, muted palette (ochres, dusty blues, sage greens, etc.)
    result = apply_earthy_palette(result, earthiness)
    
    # Step 5: Enhance contrast for bold, graphic look
    result = enhance_contrast_directional(result, contrast)
    
    # Step 6: Create uniform shadow blocks (WPA style: uniform fills, not gradients)
    result = create_uniform_shadows(result, shadow_strength)
    
    # Step 7: Apply atmospheric perspective (lighter backgrounds, darker foregrounds)
    result = apply_atmospheric_perspective(result, atmospheric_perspective)
    
    # Step 8: Create flat color blocks (silkscreen/stencil effect)
    result = create_flat_color_blocks(result, block_size)
    
    # Step 9: Quantize to limited color palette (6-12 colors for WPA style)
    result = quantize_colors_flat(result, n_colors)
    
    # Step 10: Final contrast boost for graphic poster look
    result = enhance_contrast_directional(result, 1.1)
    
    return result


def process_image(
    input_path: Path,
    output_path: Optional[Path] = None,
    n_colors: int = 8,
    simplification: float = 0.6,
    earthiness: float = 0.4,
    contrast: float = 1.6,
    saturation: float = 1.2,
    block_size: int = 6,
    atmospheric_perspective: float = 0.6,
    shadow_strength: float = 0.4,
    edge_preservation: float = 0.3,
) -> Path:
    """
    Process a single image file.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image (if None, auto-generate)
        n_colors: Number of colors in palette (6-12)
        simplification: How much to simplify detail (0.0-1.0)
        earthiness: Shift toward earthy/muted tones (0.0-1.0)
        contrast: Contrast enhancement factor
        saturation: Color saturation factor
        block_size: Size of flat color blocks
        atmospheric_perspective: Strength of depth via value (0.0-1.0)
        shadow_strength: Strength of uniform shadow blocks (0.0-1.0)
        edge_preservation: How much to preserve edge detail (0.0-1.0)
    
    Returns:
        Path to output image
    """
    # Load image
    try:
        image = Image.open(input_path)
    except Exception as e:
        print(f"Error loading image {input_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply WPA style
    result = apply_wpa_style(
        image,
        n_colors=n_colors,
        simplification=simplification,
        earthiness=earthiness,
        contrast=contrast,
        saturation=saturation,
        block_size=block_size,
        atmospheric_perspective=atmospheric_perspective,
        shadow_strength=shadow_strength,
        edge_preservation=edge_preservation,
    )
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_wpa{input_path.suffix}"
    
    # Save result
    result.save(output_path, quality=95)
    print(f"Saved: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert photos to authentic WPA National Park poster style (1935-1943)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with defaults (authentic WPA style)
  python wpa_converter.py photo.jpg
  
  # More colors, less simplified (more detail retained)
  python wpa_converter.py photo.jpg --n-colors 12 --simplification 0.4
  
  # More earthy/muted tones
  python wpa_converter.py photo.jpg --earthiness 0.6
  
  # Stronger atmospheric perspective
  python wpa_converter.py photo.jpg --atmospheric-perspective 0.8
  
  # More graphic, less detail
  python wpa_converter.py photo.jpg --simplification 0.8 --block-size 8
        """
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help='Input image file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output image file path (default: input_name_wpa.ext)'
    )
    
    parser.add_argument(
        '--n-colors',
        type=int,
        default=8,
        choices=range(6, 13),
        metavar='N',
        help='Number of colors in palette (6-12, default: 8 for authentic WPA look)'
    )
    
    parser.add_argument(
        '--simplification',
        type=float,
        default=0.6,
        metavar='FACTOR',
        help='How much to simplify detail (0.0-1.0, default: 0.6, higher=more simplified)'
    )
    
    parser.add_argument(
        '--earthiness',
        type=float,
        default=0.4,
        metavar='FACTOR',
        help='Shift toward earthy/muted tones (0.0-1.0, default: 0.4, higher=more earthy)'
    )
    
    parser.add_argument(
        '--contrast',
        type=float,
        default=1.6,
        metavar='FACTOR',
        help='Contrast enhancement factor (1.0-2.0, default: 1.6)'
    )
    
    parser.add_argument(
        '--saturation',
        type=float,
        default=1.2,
        metavar='FACTOR',
        help='Color saturation factor (0.8-1.5, default: 1.2)'
    )
    
    parser.add_argument(
        '--block-size',
        type=int,
        default=6,
        choices=range(4, 11),
        metavar='N',
        help='Size of flat color blocks (4-10, default: 6, larger=more blocky)'
    )
    
    parser.add_argument(
        '--atmospheric-perspective',
        type=float,
        default=0.6,
        metavar='FACTOR',
        help='Strength of depth via value (0.0-1.0, default: 0.6, higher=more depth)'
    )
    
    parser.add_argument(
        '--shadow-strength',
        type=float,
        default=0.4,
        metavar='FACTOR',
        help='Strength of uniform shadow blocks (0.0-1.0, default: 0.4)'
    )
    
    parser.add_argument(
        '--edge-preservation',
        type=float,
        default=0.3,
        metavar='FACTOR',
        help='How much to preserve edge detail (0.0-1.0, default: 0.3, higher=sharper edges)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Process image
    try:
        output_path = process_image(
            args.input,
            args.output,
            n_colors=args.n_colors,
            simplification=args.simplification,
            earthiness=args.earthiness,
            contrast=args.contrast,
            saturation=args.saturation,
            block_size=args.block_size,
            atmospheric_perspective=args.atmospheric_perspective,
            shadow_strength=args.shadow_strength,
            edge_preservation=args.edge_preservation,
        )
        print(f"\n✓ Conversion complete!")
        print(f"  Input:  {args.input}")
        print(f"  Output: {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
