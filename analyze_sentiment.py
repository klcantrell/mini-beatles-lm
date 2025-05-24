import json
import os
from openai import OpenAI

def main():
    # Initialize OpenAI client
    client = OpenAI()
    
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output file paths
    input_file = os.path.join(script_dir, 'finetune_lyrics_sentiment.jsonl')
    output_file = os.path.join(script_dir, 'finetune_lyrics_sentiment_complete.jsonl')
    
    # Read input file
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    # Process each example
    with open(output_file, 'w') as f:
        for example in examples:
            # Combine prompt and completion
            full_text = example['prompt'] + ' ' + example['completion']
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a sentiment classifier. Only respond with either 'Happy' or 'Sad'."},
                    {"role": "user", "content": f"Classify the sentiment of these song lyrics as either Happy or Sad: {full_text}"}
                ],
                temperature=0,
            )
            
            # Get sentiment from response
            sentiment = response.choices[0].message.content.strip()
            
            # Create output entry
            output_entry = {
                "prompt": example['prompt'],
                "completion": example['completion'],
                "sentiment": sentiment
            }
            
            # Write to output file
            f.write(json.dumps(output_entry) + '\n')
            
            # Print progress
            print(f"Processed: {example['prompt'][:30]}... -> {sentiment}")

if __name__ == '__main__':
    main()
