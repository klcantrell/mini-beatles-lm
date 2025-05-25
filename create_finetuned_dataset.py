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
    
    # Filter for lines containing 'love' (case insensitive)
    def contains_full_word_love(line):
        words = line.lower().split()
        return any(word == 'love' for word in words)
    
    love_lines = [line for line in lines if contains_full_word_love(line)]
    
    # Take 400 random examples from lines containing 'love'
    examples = random.sample(love_lines, min(400, len(love_lines)))
    
    # Process examples and write to JSONL file
    output_file = os.path.join(script_dir, 'finetune_lyrics_with_emojis.jsonl')
    
    with open(output_file, 'w') as f:
        for example in examples:
            # Split into words to do proper word-based replacement
            words = example.split()
            processed_words = ['❤️' if word.lower() == 'love' else word for word in words]
            processed_text = ' '.join(processed_words)
            
            entry = {
                "text": processed_text
            }
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()
