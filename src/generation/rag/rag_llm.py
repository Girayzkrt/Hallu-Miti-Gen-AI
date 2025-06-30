import json
import requests
import time
from typing import Dict, Any, List
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class MedicalQAEvaluator:
    def __init__(self, model_name: str = "mistral:instruct", ollama_url: str = "http://localhost:11434",
                 qdrant_host: str = "localhost", qdrant_port: int = 6333, 
                 collection_name: str = "pmc_e5_base_sentence_based", top_k: int = 1):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # RAG setup
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = SentenceTransformer('intfloat/e5-base-v2')
        
        self.prompt_template = """You are a medical expert with years of experience in biomedical science.

        Context from medical literature:
        {context}

        Question: {question}

        Instructions:
        - Answer the following medical questions using only the provided context.
        - Answer directly without phrases like "Based on the study" or "According to the context"
        - State facts as if they are established medical knowledge
        - Be concise (2-3 sentences)

        Answer:"""

    def retrieve_context(self, question: str) -> tuple[str, list]:
        try:
            question_with_prefix = f"query: {question}"
            question_embedding = self.embedding_model.encode(question_with_prefix).tolist()
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=question_embedding,
                limit=self.top_k
            )
            
            context_pieces = []
            for i, result in enumerate(search_results):

                context_entry = f"[Source {i+1}]"
                
                if hasattr(result, 'payload') and result.payload:
                    title = result.payload.get('title', '')
                    if title:
                        context_entry += f"\nStudy/Article: {title}"
                    
                    keywords = result.payload.get('keywords', '')
                    if keywords:
                        context_entry += f"\nMedical Keywords: {keywords}"
                    
                    content_parts = []
                    
                    full_abstract = result.payload.get('full_abstract', '')
                    if full_abstract:
                        content_parts.append(f"Abstract: {full_abstract}")
                    
                    results = result.payload.get('results', '')
                    if results:
                        content_parts.append(f"Results: {results}")
                    
                    if content_parts:
                        context_entry += f"\n" + "\n\n".join(content_parts)
                    else:
                        continue

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No relevant context found due to error.", []

    def query_llm(self, question: str) -> tuple[str, str, list]:
        context, context_metadata = self.retrieve_context(question)
        
        prompt = self.prompt_template.format(question=question, context=context)
        
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
            ai_answer = result.get("response", "").strip()
            
            return ai_answer, context, context_metadata
            
        except requests.exceptions.RequestException as e:
            print(f"Couldn't connect to the AI: {e}")
            return f"ERROR: {str(e)}", context, context_metadata
        except json.JSONDecodeError as e:
            print(f"Got weird response from AI: {e}")
            return f"ERROR: Invalid response", context, context_metadata

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
            ai_answer, retrieved_context, context_metadata = self.query_llm(question)
            end_time = time.time()
            
            print(f"AI says: {ai_answer[:100]}..." if len(ai_answer) > 100 else f"AI says: {ai_answer}")
            print(f"That took {end_time - start_time:.2f} seconds\n")
            
            # Save
            result = {
                "question_id": i,
                "question": question,
                "ground_truth": correct_answer,
                "llm_answer": ai_answer,
                "retrieved_context": retrieved_context,
                "context_metadata": context_metadata,
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
    parser = argparse.ArgumentParser(description="Mistral AI using RAG")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default="mistral:instruct")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", default=6333, type=int)
    parser.add_argument("--collection", default="pmc_e5_base_sentence_based")
    parser.add_argument("--top-k", default=1, type=int)
    
    args = parser.parse_args()
    

    if not Path(args.input_file).exists():
        print(f"File is missin '{args.input_file}'")
        return
    
    evaluator = MedicalQAEvaluator(
        model_name=args.model,
        ollama_url=args.ollama_url,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        top_k=args.top_k
    )
    

    try:
        test_response = evaluator.query_llm("Whatsupppp?")
        if "ERROR" in test_response:
            print(f"upsi  {test_response}")
            return
        print("Lesgo\n")
    except Exception as e:
        print(f"connection err {e}")
        return
    
    #testing
    evaluator.evaluate_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        question_key=args.question_key,
        answer_key=args.answer_key
    )

if __name__ == "__main__":
    main()