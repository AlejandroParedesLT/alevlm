# scripts/prepare_data_mmap.py
import numpy as np
from pathlib import Path
from PIL import Image
import io
from alegpt.tokenizer.tokenizer import Tokenizer
from datasets import load_dataset
from torchvision import transforms


def create_mmap_vlm_dataset(
    output_dir: str,
    dataset_name: str = "jxie/flickr8k",
    split: str = "train",
    img_size: int = 224,
    max_caption_length: int = 77
):
    """
    Create binary files compatible with np.memmap for VLM training
    
    Creates two aligned binary files:
    - {split}_images.bin: [N, 3*H*W] flattened float32 images
    - {split}_tokens.bin: [N, max_len] int32 tokens
    - {split}_meta.npz: metadata (shapes, counts, etc.)
    
    Both files have samples at the same indices - sample i in images 
    corresponds to sample i in tokens.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tokenizer
    assets_dir = Path(__file__).parent.parent
    vocab_filepath = assets_dir / "tokenizer" / 'assets' / 'vocab.json'
    merges_filepath = assets_dir / "tokenizer" / 'assets' / 'merges.json'
    tokenizer = Tokenizer.from_file(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath
    )
    
    # Image preprocessing
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"Loading {dataset_name} dataset split '{split}'...")
    dataset = load_dataset(dataset_name, split=split)
    
    # File paths
    images_path = output_dir / f"{split}_images.bin"
    tokens_path = output_dir / f"{split}_tokens.bin"
    meta_path = output_dir / f"{split}_meta.npz"
    
    # Calculate flattened image size
    img_elements = 3 * img_size * img_size
    
    all_images = []
    all_tokens = []
    token_lengths = []
    image_ids = []
    
    total_samples = 0
    
    print(f"Processing {len(dataset)} examples...")
    
    for idx, example in enumerate(dataset):
        try:
            # Get image
            image = example.get('image')
            if image is None:
                continue
            
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            else:
                image = image.convert('RGB')
            
            # Process image to tensor [3, H, W]
            img_tensor = image_transform(image)
            # Flatten to [3*H*W]
            img_flat = img_tensor.numpy().flatten().astype(np.float32)
            
            # Get captions
            captions = example.get('caption', [])
            if isinstance(captions, str):
                captions = [captions]
            
            # Create one sample per caption (same image, different captions)
            for caption in captions:
                if not isinstance(caption, str) or len(caption.strip()) == 0:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(caption.strip())
                original_len = len(tokens)
                
                # Pad/truncate to fixed length
                if len(tokens) > max_caption_length:
                    tokens = tokens[:max_caption_length]
                else:
                    tokens = tokens + [0] * (max_caption_length - len(tokens))
                
                # Store (same image repeated for each caption)
                all_images.append(img_flat)
                all_tokens.append(tokens)
                token_lengths.append(original_len)
                image_ids.append(idx)
                
                total_samples += 1
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} images -> {total_samples} image-caption pairs")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    # Convert to numpy arrays
    print(f"\nConverting {total_samples} samples to numpy arrays...")
    images_array = np.array(all_images, dtype=np.float32)  # [N, 3*H*W]
    tokens_array = np.array(all_tokens, dtype=np.int32)    # [N, max_len]
    
    # Save as binary files
    print(f"Saving images to {images_path}...")
    images_array.tofile(images_path)
    
    print(f"Saving tokens to {tokens_path}...")
    tokens_array.tofile(tokens_path)
    
    # Save metadata
    print(f"Saving metadata to {meta_path}...")
    np.savez(
        meta_path,
        num_samples=total_samples,
        img_size=img_size,
        img_channels=3,
        img_elements=img_elements,
        max_caption_length=max_caption_length,
        images_shape=(total_samples, img_elements),
        tokens_shape=(total_samples, max_caption_length),
        token_lengths=np.array(token_lengths, dtype=np.int32),
        image_ids=np.array(image_ids, dtype=np.int32)
    )
    
    print(f"\n✓ Finished!")
    print(f"  Total image-caption pairs: {total_samples:,}")
    print(f"  Images shape: {images_array.shape}")
    print(f"  Tokens shape: {tokens_array.shape}")
    print(f"  Images file: {images_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Tokens file: {tokens_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return total_samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Flickr8k for VLM training (memmap format)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--dataset', default='jxie/flickr8k', help='Dataset name')
    parser.add_argument('--split', default='train', help='Dataset split')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (square)')
    parser.add_argument('--max_caption_length', type=int, default=77, help='Max caption tokens')
    
    args = parser.parse_args()
    
    create_mmap_vlm_dataset(
        args.output_dir,
        args.dataset,
        args.split,
        args.img_size,
        args.max_caption_length
    )