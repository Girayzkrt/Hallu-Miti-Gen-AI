import json
import time
from typing import Dict, Any, Tuple, List
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class MedicalQAEvaluator:
    def __init__(self, model_path: str = "girayzkrt/Mist-FT-2106-4bit",
                 qdrant_host: str = "localhost", qdrant_port: int = 6333, 
                 collection_name: str = "pmc_e5_base_sentence_based", top_k: int = 1):
        self.model_path = model_path
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
 
        # RAG setup
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = SentenceTransformer('intfloat/e5-base-v2')

        self.system_message = (
            "You are a medical expert with years of experience in biomedical science. "
            "Answer the following medical question using only the provided context."
            "Answer directly without phrases like 'Based' on the study' or 'According to the context'"
            "Be concise (2-3 sentences)"
        )

    def retrieve_context(self, question: str) -> Tuple[str, List]:
        try:
            question_with_prefix = f"query: {question}"
            query_vector = self.embedding_model.encode(question_with_prefix).tolist()
            
            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k
            )
            
            retrieved_contexts = []
            context_metadata = []
            
            for hit in hits:
                payload = hit.payload
                title = payload.get("title", "")
                abstract = payload.get("full_abstract", "")
                results = payload.get("results", "")
                fulltext = payload.get("fulltext", "")
                
                combined = (
                    f"Abstract: {abstract}\n"
                    f"Results: {results}"
                )
                retrieved_contexts.append(combined)
                
                context_metadata.append({
                    'title': title,
                    'abstract': abstract,
                    'results': results,
                    'fulltext': fulltext,
                    'score': hit.score,
                    'payload': payload
                })
            
            context = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else "No relevant context found."
            return context, context_metadata
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No relevant context found due to error.", []

    def query_llm(self, question: str) -> Tuple[str, str, List]:
        try:
            context, context_metadata = self.retrieve_context(question)
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            input_length = inputs.shape[1]
            generated_tokens = outputs[0][input_length:]
            
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return answer.strip(), context, context_metadata
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"ERROR: {str(e)}", "", []

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
            ai_answer, context_used, context_metadata = self.query_llm(question)
            end_time = time.time()
            
            print(f"AI says: {ai_answer[:100]}..." if len(ai_answer) > 100 else f"AI says: {ai_answer}")

            result = {
                "question_id": i,
                "question": question,
                "ground_truth": correct_answer,
                "llm_answer": ai_answer,
                "model": self.model_path,
                "response_time": round(end_time - start_time, 2),
                "rag_context": context_used,
                "rag_sources": len(context_metadata),
                "rag_metadata": context_metadata
            }
            
            for key, value in item.items():
                if key not in [question_key, answer_key]:
                    result[f"original_{key}"] = value
            
            results.append(result)
            time.sleep(0.5)
        
        self.save_results(results, output_file)

def main():
    parser = argparse.ArgumentParser(description="Fine-tuned LLM with RAG")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default="girayzkrt/Mist-FT-2106-4bit")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", default=6333, type=int)
    parser.add_argument("--collection-name", default="pmc_e5_base_sentence_based")
    parser.add_argument("--top-k", default=1, type=int)
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"File is missin '{args.input_file}'")
        return
    
    try:
        evaluator = MedicalQAEvaluator(
            model_path=args.model,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection_name,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"Failed to load model or RAG components: {e}")
        return
    
    try:
        test_response, test_context, test_metadata = evaluator.query_llm("killmern")
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