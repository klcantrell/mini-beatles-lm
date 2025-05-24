import random
import json
import os

def main():
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read the lyrics file
    lyrics_file = os.path.join(script_dir, 'beatles_lyrics.txt')
    with open(lyrics_file, 'r') as f:
        lines = f.readlines()
    
    # Clean up lines and remove empty ones
    lines = [line.strip() for line in lines if line.strip()]
    
    # Take 500 random examples
    examples = random.sample(lines, min(500, len(lines)))
    
    # Process each example and write to JSONL file
    output_file = os.path.join(script_dir, 'finetune_lyrics_sentiment.jsonl')
    with open(output_file, 'w') as f:
        for example in examples:
            # Split example into words
            words = example.split()
            
            # Choose random number between 3 and 5 for prompt length
            prompt_length = random.randint(3, 5)
            
            # Make sure we don't try to take more words than exist
            prompt_length = min(prompt_length, len(words))
            
            # Create prompt and completion
            prompt = ' '.join(words[:prompt_length])
            completion = ' '.join(words[prompt_length:])
            
            # Create entry
            entry = {
                "prompt": prompt,
                "completion": completion
            }
            
            # Write to file
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()
