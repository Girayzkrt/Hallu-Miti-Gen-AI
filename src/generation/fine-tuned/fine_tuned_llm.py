import json
import time
from typing import Dict, Any
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalQAEvaluator:
    def __init__(self, model_path: str = "girayzkrt/Mist-FT-2106-4bit"):
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        self.system_message = (
            "You are a medical expert with comprehensive knowledge of the entire biomedical literature, "
            "please provide an answer of the following question. Your response should be "
            "concise (2-3 sentences), accurate, and based on the latest scientific evidence."
        )

    def query_llm(self, question: str) -> str:
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Question: {question}\n\nAnswer:"}
            ]
            
            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            input_length = inputs.shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"ERROR: {str(e)}"

    def load_data(self, file_path: str) -> list:
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Couldn't read line {line_num}: {e}")
                            continue
        except FileNotFoundError:
            print(f"Can't find the file {file_path}")
            return []
        except Exception as e:
            print(f"Something went wrong reading the file: {e}")
            return []
        
        return data

    def save_results(self, results: list, output_file: str):
        """Save all the results to a new file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Couldn't save the results: {e}")

    def evaluate_dataset(self, input_file: str, output_file: str, 
                        question_key: str = "question", answer_key: str = "answer"):
        
        data = self.load_data(input_file)
        
        if not data:
            print("No questions found.")
            return
        
        print(f"Found {len(data)} questions")
        
        results = []
        
        for i, item in enumerate(data, 1):
            if question_key not in item:
                print(f"Question #{i} doesn't have a '{question_key}'")
                continue
                
            question = item[question_key]
            correct_answer = item.get(answer_key, "No answer provided")
            
            print(f"Testing question {i} out of {len(data)}")
            print(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
            
            start_time = time.time()
            ai_answer = self.query_llm(question)
            end_time = time.time()
            
            print(f"AI says: {ai_answer[:100]}..." if len(ai_answer) > 100 else f"AI says: {ai_answer}")
            print(f"That took {end_time - start_time:.2f} seconds\n")
            
            # Save
            result = {
                "question_id": i,
                "question": question,
                "ground_truth": correct_answer,
                "llm_answer": ai_answer,
                "model": self.model_path,
                "response_time": round(end_time - start_time, 2)
            }
            
            for key, value in item.items():
                if key not in [question_key, answer_key]:
                    result[f"original_{key}"] = value
            
            results.append(result)
            time.sleep(0.5)
        
        # Save
        self.save_results(results, output_file)
        print(f"Tested {len(results)} questions")

def main():
    parser = argparse.ArgumentParser(description="Fine-tuned LLM ")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default="girayzkrt/Mist-FT-2106-4bit")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"File is missin '{args.input_file}'")
        return
    
    try:
        evaluator = MedicalQAEvaluator(model_path=args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    try:
        test_response = evaluator.query_llm("Whatsupppp?")
        if "ERROR" in test_response:
            print(f"Failed: {test_response}")
            return
        print(f"Response: {test_response}")
    except Exception as e:
        print(f"Connection error: {e}")
        return
    
    evaluator.evaluate_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        question_key=args.question_key,
        answer_key=args.answer_key
    )

if __name__ == "__main__":
    main()