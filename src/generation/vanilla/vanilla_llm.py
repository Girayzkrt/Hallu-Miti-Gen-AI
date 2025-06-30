import json
import requests
import time
from typing import Dict, Any
import argparse
from pathlib import Path

class MedicalQAEvaluator:
    def __init__(self, model_name: str = "mistral:instruct", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        self.prompt_template = """You are a medical expert with years of experience in biomedical science. Please provide an answer of the following question. Your response should be concise(2-3 sentences), accurate, and based on the latest scientific evidence.

Question: {question}

Answer:"""

    def query_llm(self, question: str) -> str:
        prompt = self.prompt_template.format(question=question)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 300,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Couldn't connect to the AI: {e}")
            return f"ERROR: {str(e)}"
        except json.JSONDecodeError as e:
            print(f"Got weird response from AI: {e}")
            return f"ERROR: Invalid response"

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
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Couldn't save the results: {e}")

    def evaluate_dataset(self, input_file: str, output_file: str, 
                        question_key: str = "question", answer_key: str = "answer"):
        print(f"Reading questions from {input_file}...")
        data = self.load_data(input_file)
        
        if not data:
            print("no questions found.")
            return
        
        print(f"Found {len(data)} questions")
        
        results = []
        
        for i, item in enumerate(data, 1):
            if question_key not in item:
                print(f"question #{i} doesn't have a '{question_key}'")
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
                "model": self.model_name,
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
    parser = argparse.ArgumentParser(description="Pure LLM")
    parser.add_argument("input_file",)
    parser.add_argument("output_file")
    parser.add_argument("--model", default="mistral:instruct")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    
    args = parser.parse_args()
    

    if not Path(args.input_file).exists():
        print(f"File is missin '{args.input_file}'")
        return
    
    evaluator = MedicalQAEvaluator(
        model_name=args.model,
        ollama_url=args.ollama_url
    )
    
    try:
        test_response = evaluator.query_llm("Whatsupppp?")
        if "ERROR" in test_response:
            print(f"upsi {test_response}")
            return
        print("Lesgo\n")
    except Exception as e:
        print(f"connection err {e}")
        return
    
    evaluator.evaluate_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        question_key=args.question_key,
        answer_key=args.answer_key
    )

if __name__ == "__main__":
    main()