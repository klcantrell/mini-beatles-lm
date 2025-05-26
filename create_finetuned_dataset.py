import json
import os

def main():
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read the All You Need Is Love lyrics file
    lyrics_file = os.path.join(script_dir, 'all_you_need_is_love_lyrics.txt')
    with open(lyrics_file, 'r') as f:
        lines = f.readlines()
    
    # Clean up lines and remove empty ones
    lines = [line.strip() for line in lines if line.strip()]
    
    # Remove comment lines (those starting with //) and empty lines
    examples = [line for line in lines if line and not line.startswith('//')]
    
    # Process examples and write to JSONL file
    output_file = os.path.join(script_dir, 'finetune_lyrics_with_emojis.jsonl')
    
    with open(output_file, 'w') as f:
        for example in examples:
            # Only write the emoji version
            words = example.split()
            processed_words = []
            for word in words:
                # Strip punctuation and check if the word is 'love' (case insensitive)
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word.lower() == 'love':
                    # Preserve any punctuation that was with the original word
                    punctuation = ''.join(c for c in word if not c.isalnum())
                    processed_words.append('❤️' + punctuation)
                else:
                    processed_words.append(word)
            
            processed_text = ' '.join(processed_words)
            
            entry = {
                "text": processed_text
            }
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()
