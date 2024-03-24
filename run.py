from typing import Dict
import os
import re
import json
import random
import argparse
from pathlib import Path

from tqdm import tqdm
import concurrent.futures
import openai
from openai import OpenAI
from rich.console import Console
from rich.theme import Theme


custom_theme = Theme({
    "info": "bold dim cyan",
    "warning": "bold magenta",
    "danger": "bold red",
    "debugging": "bold sandy_brown"
})
console = Console(theme=custom_theme)

PROJECT_HOME = Path(__file__).parent.resolve()
OUTPUT_DIR = os.path.join(PROJECT_HOME, 'output')

TEMPLATE = 'The following is a dialogue between {spk1} and {spk2}. The dialogue is provided line-by-line. In the given dialogue, select all utterances that are appropriate for sharing the image in the next turn, and write the speaker who will share the image after the selected utterance. You should also provide a rationale for your decision and describe the relevant image concisely.\n\nDialogue:\n{dialogue}\n\nRestrictions:\n(1) your answer should be in the format of "<UTTERANCE> | <SPEAKER> | <RATIONALE> | <IMAGE DESCRIPTION>".\n(2) you MUST select the utterance in the given dialogue, NOT generate a new utterance.\n(3) the rationale should be written starting with "To".\n\nAnswer:\n1.',
PATTERN = r'^(?:\d+\.\s+)?\"?(?P<utterance>.*?)\"?\s+\|\s+(?P<speaker>.*?)(?:\s+\|\s+(?P<rationale>.*?))?(?:\s+\|\s+(?P<description>.*?))?$'


class Runner():
    def __init__(self, args):
        self.args = args
        self.output_base_dir = os.path.join(OUTPUT_DIR, self.args.run_id + ":{}".format(self.args.model))
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        self.client = OpenAI(
            api_key="<OPENAI_API_KEY>"
        )
    
    def run(self, dialogue, spk1, spk2):
        prompt = TEMPLATE[0].format(spk1=spk1, spk2=spk2, dialogue=dialogue)
        console.log(f'Prompt input: {prompt}', style='debugging')
        
        output = self.interact(prompt)    
        console.log(f'Output: {output}', style='debugging')
        
        self.dump_output(output, os.path.join(self.output_base_dir, 'test.jsonl'))
        
    def generate(self, prompt):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[{"role": "user", "content": "{}".format(prompt)}],
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty
                )
                break
            except (RuntimeError, openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return response.choices[0].message.content.strip()

    def parse(self, generation):
        
        matches = re.finditer(PATTERN, generation, re.MULTILINE)
        results = []
        for match in matches:
            utter = match.group('utterance')
            speaker = match.group('speaker')
            rationale = match.group('rationale')
            description = match.group('description')
            
            results.append({
                'utterance': utter,
                'speaker': speaker,
                'rationale': rationale,
                'description': description
            })

        return results

    def interact(self, prompt_input):
        generation = self.generate(prompt_input)
        parsed_output = self.parse(generation)

        return parsed_output

    def dump_output(self, outputs, file_name=None):
        f = open(file_name, 'w') 
        for output in outputs:
            f.write(json.dumps(output) + '\n')
        f.close()

def main(args):
    runner = Runner(args)
    
    with open('sample.txt', 'r') as f:
        sample_dialogue = f.read()
    
    sample_spk1 = 'Jennifer'
    sample_spk2 = 'Keyana'
    
    runner.run(sample_dialogue, sample_spk1, sample_spk2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating multi-modal dialogues using LLM')
    parser.add_argument('--run-id',
                        type=str,
                        default='vanilla',
                        help='the name of the directory where the output will be dumped')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-3.5-turbo',
                        help='which LLM to use')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.9,
                        help="control randomness: lowering results in less random completion")
    parser.add_argument('--top-p',
                        type=float,
                        default=0.95,
                        help="nucleus sampling")
    parser.add_argument('--frequency-penalty',
                        type=float,
                        default=1.0,
                        help="decreases the model's likelihood to repeat the same line verbatim")
    parser.add_argument('--presence-penalty',
                        type=float,
                        default=0.6,
                        help="increases the model's likelihood to talk about new topics")
    parser.add_argument('--max-tokens',
                        type=int,
                        default=1024,
                        help='maximum number of tokens to generate')
    
    args = parser.parse_args()
    main(args)