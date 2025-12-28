import webdataset as wds
from pathlib import Path
from PIL import Image
import io
from alegpt.tokenizer.tokenizer import Tokenizer
from datasets import load_dataset
from torchvision import transforms
import json


def create_webdataset(
    output_dir: str,
    dataset_name: str = "jxie/flickr8k",
    split: str = "train",
    img_size: int = 224,
    samples_per_shard: int = 1000  # Images per tar file
):
    """
    Create WebDataset format for efficient large-scale training
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tokenizer
    assets_dir = Path(__file__).parent.parent.parent.parent
    vocab_filepath = assets_dir / 'assets' / 'vocab.json'
    merges_filepath = assets_dir / 'assets' / 'merges.json'
    tokenizer = Tokenizer.from_file(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath
    )
    
    # Image preprocessing (just resize, don't normalize - do that in dataloader)
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])
    
    # Load dataset
    print(f"Loading {dataset_name} dataset split '{split}'...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Create sharded writer
    shard_pattern = str(output_dir / f"{split}-%06d.tar")
    
    with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as sink:
        sample_count = 0
        
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
                
                # Resize image
                image = image_transform(image)
                
                # Extract captions from caption_0, caption_1, caption_2, caption_3, caption_4
                captions = []
                for i in range(5):  # Flickr8k has 5 captions per image
                    caption_key = f'caption_{i}'
                    if caption_key in example and example[caption_key]:
                        caption = example[caption_key]
                        if isinstance(caption, str) and len(caption.strip()) > 0:
                            captions.append(caption.strip())
                
                if not captions:
                    print(f"Warning: No valid captions found for image {idx}")
                    continue
                
                # Save image as JPEG bytes once (reuse for all captions)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=95)
                img_bytes = img_buffer.getvalue()
                
                # Create one sample per caption
                for cap_idx, caption in enumerate(captions):
                    # Tokenize
                    tokens = tokenizer.encode(caption)
                    
                    # Create unique key for this sample
                    key = f"{idx:08d}_{cap_idx:02d}"
                    
                    # Write sample with multiple components
                    sample = {
                        "__key__": key,
                        "jpg": img_bytes,  # Image as JPEG
                        "txt": caption,  # Original text
                        "json": json.dumps({
                            "tokens": tokens,
                            "image_id": idx,
                            "caption_id": cap_idx
                        })
                    }
                    
                    sink.write(sample)
                    sample_count += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(dataset)} images -> {sample_count} samples")
            
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue
    
    print(f"\n✓ Finished!")
    print(f"  Total samples: {sample_count:,}")
    print(f"  Shards created: {(sample_count // samples_per_shard) + 1}")
    
    return sample_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset', default='jxie/flickr8k')
    parser.add_argument('--split', default='train')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--samples_per_shard', type=int, default=1000)
    
    args = parser.parse_args()
    
    create_webdataset(
        args.output_dir,
        args.dataset,
        args.split,
        args.img_size,
        args.samples_per_shard
    )