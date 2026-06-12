# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from einops import rearrange
# from pathlib import Path

# # Prevents memory fragmentation and allows PyTorch to manage larger segments
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # Ensure Python can always find your library
# sys.path.append("/mnt/data/nidhi/qnn")

# from quanta_neural_networks.reconstruction.efficient_ssd import EfficientSSD
# from quanta_neural_networks.utils.train_utils import load_checkpoint

# def process_all_test_sets(base_data_path, output_base_path, checkpoint_folder):
#     base_path = Path(base_data_path)
#     output_base = Path(output_base_path)
#     ckpt_path = Path(checkpoint_folder)
    
#     # 1. Initialize Model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     subsampling_rate = 64
#     model = EfficientSSD(
#         channels=64, state_dim=12, subsampling=subsampling_rate, 
#         group_num=4, memory_size=20, units=6
#     )

#     # 2. Load Checkpoint
#     print(f"Loading checkpoint from {ckpt_path}...")
#     load_checkpoint(model, ckpt_path, ckpt_file="checkpoint.pth", ckpt_key="model", strict=False)
#     model.eval()

#     # 3. Find all .npy files
#     all_npy_files = list(base_path.rglob("*.npy"))
#     print(f"Found {len(all_npy_files)} files to process.")

#     for file_path in all_npy_files:
#         try:
#             # Preserve folder structure: OutputBase / ParentFolder / filename.png
#             scene_name = file_path.parent.name 
#             new_file_name = file_path.with_suffix('.png').name
#             output_image_path = output_base / scene_name / new_file_name
            
#             # Skip if already processed
#             if output_image_path.exists():
#                 print(f"Skipping (already exists): {scene_name}/{new_file_name}")
#                 continue

#             print(f"\n---> Processing: {scene_name}/{new_file_name}")
#             output_image_path.parent.mkdir(parents=True, exist_ok=True)

#             # --- Data Loading & Unpackbits ---
#             packed_data = np.load(file_path, mmap_mode="r") 
#             unpacked = np.unpackbits(packed_data, axis=2)
#             bitplanes = rearrange(unpacked, 't h w c -> (t c) h w')
            
#             data_tensor_cpu = torch.from_numpy(bitplanes)
#             photon_cubes_cpu = rearrange(data_tensor_cpu, '(t c) h w -> c t h w', t=1024, c=3)

#             chunk_size = 64 
#             reconstructed_channels = []

#             # --- Inference Loop (Channel by Channel) ---
#             for i, color_name in enumerate(['Red', 'Green', 'Blue']):
#                 print(f"\n--- Processing {color_name} Channel ---")
                
#                 # Hard reset the model's memory for the new color channel to prevent color bleeding
#                 if hasattr(model, 'states'):
#                     model.states = None
                    
#                 model.to(device)
#                 channel_data = photon_cubes_cpu[i] 
#                 channel_output_list = []

#                 with torch.no_grad():
#                     for start in range(0, 1024, chunk_size):
#                         end = min(start + chunk_size, 1024)
                        
#                         # Move only this chunk to GPU
#                         chunk = channel_data[start:end].float().to(device)
                        
#                         # Permute to [Height, Width, Time] for line 314 requirement
#                         chunk = chunk.permute(1, 2, 0)
                        
#                         if chunk.max() > 1.0: 
#                             chunk = chunk / 255.0

#                         print(f"  Chunk {start}-{end} | Input: {list(chunk.shape)}")

#                         # Forward Pass
#                         results = model.forward_online(
#                             chunk, 
#                             clear_states=(start == 0) 
#                         )
                        
#                         # Unpack flexibly
#                         out_chunk = results[0] if isinstance(results, tuple) else results
                        
#                         # Permute back and move to CPU immediately to free VRAM
#                         channel_output_list.append(out_chunk.permute(2, 0, 1).cpu())
                        
#                         # Clear chunk-level cache
#                         del chunk, out_chunk, results
#                         torch.cuda.empty_cache()

#                 # Combine chunks for this color (Final shape: [16, H, W])
#                 full_channel = torch.cat(channel_output_list, dim=0)
#                 reconstructed_channels.append(full_channel)
                
#                 # Save Grayscale Preview in the scene's output directory
#                 preview_idx = full_channel.shape[0] // 2
#                 preview_frame = full_channel[preview_idx].numpy()
#                 preview_filename = output_image_path.parent / f"{scene_name}_{color_name.lower()}_channel.png"
#                 plt.imsave(preview_filename, preview_frame, cmap='gray')
#                 print(f"  Saved preview: {preview_filename.name}")

#                 # NUCLEAR RESET: Move model off GPU and synchronize to fully flush memory
#                 model.to('cpu')
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize() 
#                 print(f"  GPU Memory Flushed after {color_name} channel.")

#             # --- Combine Channels and Save Final Frame ---
#             print("\nRecombining channels and saving final frame...")
            
#             # Final Shape: [Time, H, W, Channels] -> [16, H, W, 3]
#             final_output = torch.stack(reconstructed_channels, dim=-1)
            
#             # Grab the last frame in the sequence
#             last_frame_rgb = final_output[-1].numpy()
            
#             # Scrub NaNs/Infs (Matplotlib will crash if it sees these)
#             last_frame_rgb = np.nan_to_num(last_frame_rgb, nan=0.0, posinf=1.0, neginf=0.0)
            
#             # Trust the network's output and strictly clamp to [0, 1] like your original code
#             last_frame_rgb = np.clip(last_frame_rgb, 0.0, 1.0)
            
#             # Save final RGB frame directly
#             plt.imsave(output_image_path, last_frame_rgb)
#             print(f"      Successfully saved final image: {output_image_path}")
            
#         except Exception as e:
#             print(f"FAILED to process {file_path}: {e}")
#             continue


# if __name__ == "__main__":
#     # Define your paths here
#     BASE_DATA = "/4tb_hdd/SinglePhotonChallenge/test/" 
#     OUTPUT_DATA = "/mnt/data/nidhi/data_recons_train"
#     CKPT = "/mnt/data/nidhi/quanta_neural_networks/data/ckpt/efficient_ssd"
    
#     process_all_test_sets(BASE_DATA, OUTPUT_DATA, CKPT)


import numpy as np
import torch
from einops import rearrange
from pathlib import Path
import os
import matplotlib.pyplot as plt
import traceback

# Prevents memory fragmentation and allows PyTorch to manage larger segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from quanta_neural_networks.reconstruction.efficient_ssd import EfficientSSD
from quanta_neural_networks.utils.train_utils import load_checkpoint

def run_batch_inference():
    # --- CONFIGURATION ---
    input_dir = Path("/4tb_hdd/SinglePhotonChallenge/test/") # Change to your root input dir
    output_dir = Path("/mnt/data/nidhi/data_recons_train") # Change to your desired root output dir
    checkpoint_folder = Path("/mnt/data/nidhi/quanta_neural_networks/data/ckpt/efficient_ssd") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model (Done once for all files)
    subsampling_rate = 64
    chunk_size = 64 
    
    model = EfficientSSD(
        channels=64, state_dim=12, subsampling=subsampling_rate,
        group_num=4, memory_size=20, units=6            
    )

    print("Loading checkpoint...")
    load_checkpoint(model, checkpoint_folder, ckpt_file="checkpoint.pth", 
                    ckpt_key="model", strict=False)
    model.eval()

    # 2. Find all .npy files recursively
    npy_files = list(input_dir.rglob("*.npy"))
    print(f"Found {len(npy_files)} .npy files to process.")

    # 3. Batch Processing Loop
    for file_idx, npy_path in enumerate(npy_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing File {file_idx}/{len(npy_files)}: {npy_path.name}")
        
        # Determine the relative path to recreate the directory structure
        rel_path = npy_path.relative_to(input_dir)
        output_file_path = output_dir / rel_path.with_suffix('.png')
        
        # Create subdirectories in the output folder if they don't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if output_file_path.exists():
            print(f"Output already exists, skipping: {output_file_path}")
            continue

        try:
            # Load Data
            photoncube = np.load(npy_path, mmap_mode="r") 
            photoncube = np.unpackbits(photoncube, axis=2)
            data_tensor_cpu = torch.from_numpy(photoncube)
            
            print(f"  Loaded tensor shape: {data_tensor_cpu.shape}")

            # REARRANGE FIX: Data is already 4D [T, H, W, C]. We just swap axes to [C, T, H, W].
            photon_cubes_cpu = rearrange(data_tensor_cpu, 't h w c -> c t h w')

            last_frames_list = []

            # 4. Inference Loop per color channel
            for i, color_name in enumerate(['Red', 'Green', 'Blue']):
                print(f"  --- {color_name} Channel ---")
                if hasattr(model, "states"):
                    model.states = None
                model.to(device)
                channel_data = photon_cubes_cpu[i]  
                channel_output_list = []

                with torch.no_grad():
                    # Note: channel_data.shape[0] is dynamic (usually 1024 based on the T dimension)
                    time_steps = channel_data.shape[0]
                    for start in range(0, time_steps, chunk_size):
                        end = min(start + chunk_size, time_steps)
                        
                        if (end - start) < subsampling_rate:
                            continue

                        chunk = channel_data[start:end].float().to(device)
                        chunk = chunk.permute(1, 2, 0) # -> [H, W, T]
                        
                        if chunk.max() > 1.0:
                            chunk = chunk / 255.0

                        results = model.forward_online(
                            chunk, 
                            clear_states=(start == 0) 
                        )
                        
                        out_chunk = results[0] if isinstance(results, tuple) else results
                        channel_output_list.append(out_chunk.permute(2, 0, 1).cpu())
                        
                        del chunk, out_chunk, results
                        torch.cuda.empty_cache()

                # Combine chunks
                full_channel = torch.cat(channel_output_list, dim=0)
                
                # Extract ONLY the last frame of the accumulated sequence
                last_frame = full_channel[-1]
                last_frames_list.append(last_frame)

                # Reset GPU memory
                model.to('cpu')
                torch.cuda.empty_cache()
                torch.cuda.synchronize() 

            # 5. Final Recombination & Image Saving
            print(f"  Recombining channels and saving to: {output_file_path}")
            
            # Stack HxW images into HxWx3
            final_rgb_tensor = torch.stack(last_frames_list, dim=-1)
            final_rgb_image = final_rgb_tensor.clamp(0.0, 1.0).numpy()
            
            plt.imsave(output_file_path, final_rgb_image)
            print("  Success!")

        except Exception as e:
            print(f"  [ERROR] Failed to process {npy_path.name}: {e}")
            traceback.print_exc()
            
            # Ensure model is moved off GPU even if an error occurs to prevent OOM on next file
            model.to('cpu')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    run_batch_inference()