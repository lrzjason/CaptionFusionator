import time
import os
import glob
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import argparse

def process_images_and_captions(directory, max_tokens, temperature, prompt, caption_exts,hf_repo_id=None, hf_filename=None):
    image_extensions = ["jpg", "jpeg", "png"]
    image_files = [f for ext in image_extensions for f in glob.glob(f"{directory}/*.{ext}")]
    
    model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
    # Load the Llama model
    lcpp_llm = Llama(
        model_path=model_path,
        n_threads=4,  # CPU cores
        n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
    )

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        print(base_name)
        prompt_string = f"SYSTEM: {prompt}\n\nUSER: "

        caption_number = 1
        for caption_ext in caption_exts:
            caption_file = f"{base_name}.{caption_ext}"
            if os.path.isfile(caption_file):
                with open(caption_file, 'r') as f:
                    caption_text = f.read().strip()
                    if caption_ext == "wd14cap":
                        prepend_text = "Tags: "
                    else:
                        prepend_text = f"Caption {caption_number}: "
                        caption_number += 1
                    prompt_string += prepend_text + caption_text + "\n"

        prompt_string += "\nASSISTANT: "

        if caption_number > 1:
            # Generate response using Llama model
            response = lcpp_llm(
                prompt=prompt_string,
                max_tokens=max_tokens,
                temperature=temperature
            )

            synth_file = f"{base_name}.txt"
            with open(synth_file, 'w') as f:
                f.write(response['choices'][0]['text'])
                print(response['choices'][0]['text'])

def main():
    print('****STARTING LLAMA PASS****')
    parser = argparse.ArgumentParser(description="Process images and captions")
    parser.add_argument("--input_dir", type=str,help="Input directory", required=True)
    parser.add_argument("--output_dir", type=str,help="Output directory")
    parser.add_argument("--max_tokens", type=int, help="Max tokens for the Llama model", default=256)
    parser.add_argument("--temperature", type=float, help="Temperature for the Llama model", default=0.5)
    parser.add_argument("--hf_repo_id", type=str, help="HF Path to  llama repo",default="TheBloke/Luna-AI-Llama2-Uncensored-GGML") #just a random model for now
    parser.add_argument("--hf_filename", type=str, help="HF model filename",default="luna-ai-llama2-uncensored.ggmlv3.q6_K.bin") #just a random model for now
    #parser.add_argument("--model_path", type=str, help="path to llama model")
    parser.add_argument("--prompt_file_path", type=str,help="Path to txt file containing system prompt for the the model", default="llama_system_prompt.txt")
    parser.add_argument("--caption_exts", nargs='+', help="Extensions for caption files", default=["b2cap", "flamcap", "wd14cap"])

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    max_tokens = args.max_tokens
    temperature = args.temperature
    hf_repo_id = args.hf_repo_id
    hf_filename = args.hf_filename
    caption_exts = args.caption_exts
    
    
    #read in llama sys prompt file
    with open(args.prompt_file_path, 'r') as prompt_file:
       prompt = prompt_file.read().strip()

    if not output_directory:
        output_directory = input_directory

    # Validate directories
    if not os.path.isdir(input_directory):
        print("Input directory does not exist.")
        return

    if not os.path.isdir(output_directory):
        print("Output directory does not exist.")
        return

    os.chdir(output_directory)  # Change current working directory to output directory

    process_images_and_captions(input_directory, max_tokens, temperature, prompt, caption_exts , hf_repo_id, hf_filename)

if __name__ == "__main__":
    main()
