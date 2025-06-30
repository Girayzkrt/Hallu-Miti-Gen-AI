import os
import json
import uuid
from typing import List, Optional
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PubMedPaper:
    id: str
    title: str
    keywords: str
    abstract: str
    introduction: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None

@dataclass
class ChunkMetadata:
    id: str
    chunk_id: str
    chunk_index: int
    title: str
    keywords: str
    chunk_text: str
    full_abstract: str
    introduction: Optional[str] = None
    results: Optional[str] = None
    discussion: Optional[str] = None
    conclusion: Optional[str] = None

class EmbeddingPipeline:
    
    def __init__(self, 
                 model_name: str = "intfloat/e5-base-v2",
                 #model_name: str = "intfloat/e5-small-v2",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "pmc_e5_base_sentence_based",
                 max_tokens: int = 500,
                 embedding_dim: int = 768,
                 #embedding_dim: int = 384,
                 data_file: str = "../../data/json_files/parsed_pmc_2.jsonl",
                 log_file: str = "processed_papers.log",
                 embedding_batch_size: int = 32,
                 device: str = "cuda"):
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        self.data_file = data_file
        self.log_file = log_file
        
        self.embedding_batch_size = embedding_batch_size
        self.device = device
        
        # Initialize embedding model and tokenizer
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        actual_device = self.embedding_model.device
        logger.info(f"Using device: {actual_device}")
        
        if actual_device.type == 'cuda':
            import torch
            gpu_memory = torch.cuda.get_device_properties(actual_device).total_memory / 1024**3
            logger.info(f"GPU Memory Available: {gpu_memory:.1f} GB")
        

        logger.info(f"Connecting to db {qdrant_host}:{qdrant_port}")
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._create_collection()
        self.processed_papers = self._load_processed_log()
        
    def _create_collection(self):
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _load_processed_log(self) -> set:
        processed = set()
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        processed.add(line.strip())
                logger.info(f"Loaded {len(processed)} already processed papers from log")
            except Exception as e:
                logger.warning(f"Could not load processed log: {e}")
        return processed
    
    def _save_processed_paper(self, paper_id: str):
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{paper_id}\n")
            self.processed_papers.add(paper_id)
        except Exception as e:
            logger.error(f"Could not save to processed log: {e}")
    
    def load_papers(self) -> List[PubMedPaper]:
        papers = []
        
        if not os.path.exists(self.data_file):
            logger.error(f"Data file not found: {self.data_file}")
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        logger.info(f"Loading papers from {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Skip if already processed
                        paper_id = data.get('id', '')
                        if paper_id in self.processed_papers:
                            continue
                        
                        paper = PubMedPaper(
                            id=paper_id,
                            title=data.get('title', ''),
                            keywords=data.get('keywords', ''),
                            abstract=data.get('abstract', ''),
                            introduction=data.get('introduction'),
                            results=data.get('results'),
                            discussion=data.get('discussion'),
                            conclusion=data.get('conclusion')
                        )
                        papers.append(paper)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {self.data_file}: {e}")
            raise
        
        logger.info(f"Loaded {len(papers)} new papers (skipped {len(self.processed_papers)} already processed)")
        return papers
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def chunking(self, title: str, keywords: str, abstract: str) -> List[str]:
        # Combine title, keywords, and abstract
        full_text = f"{title}. Keywords: {keywords}. {abstract}"
        prefix_tokens = self.count_tokens("passage: ") - self.count_tokens("")
        full_text_tokens = self.count_tokens(full_text)
        
        # If total tokens is within limit, return as single chunk
        if full_text_tokens + prefix_tokens <= self.max_tokens:
            return [full_text]
        
        logger.debug(f"Text exceeds {self.max_tokens} tokens ({full_text_tokens + prefix_tokens}), chunking...")
        
        # Split into sentences
        sentences = sent_tokenize(full_text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            total_tokens = current_tokens + sentence_tokens + prefix_tokens
            
            if total_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        logger.debug(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        # Add "passage: " prefix
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
        
        embeddings = self.embedding_model.encode(
            prefixed_chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def process_paper(self, paper: PubMedPaper) -> List[ChunkMetadata]:
        chunks = self.chunking(paper.title, paper.keywords, paper.abstract)
        
        #Metadata for each chunk
        chunk_metadata_list = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = ChunkMetadata(
                id=paper.id,
                chunk_id=f"{paper.id}_chunk_{i}",
                chunk_index=i,
                title=paper.title,
                keywords=paper.keywords,
                chunk_text=chunk_text,
                full_abstract=paper.abstract,
                introduction=paper.introduction,
                results=paper.results,
                discussion=paper.discussion,
                conclusion=paper.conclusion
            )
            chunk_metadata_list.append(chunk_metadata)
        
        return chunk_metadata_list
    
    def embed_and_store_papers(self, papers: List[PubMedPaper], processing_batch_size: int = 1000):
        if not papers:
            logger.info("No new papers to process")
            return
        
        logger.info(f"Starting to process {len(papers)} papers in batches of {processing_batch_size}...")
        
        total_papers_processed = 0
        total_chunks_created = 0
        
        for batch_start in range(0, len(papers), processing_batch_size):
            batch_end = min(batch_start + processing_batch_size, len(papers))
            paper_batch = papers[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//processing_batch_size + 1}/{len(papers)//processing_batch_size + 1} "
                       f"(papers {batch_start + 1}-{batch_end})")

            batch_chunks_data = []
            batch_paper_ids = []
            
            for paper in tqdm(paper_batch, desc=f"Preparing chunks (batch {batch_start//processing_batch_size + 1})"):
                try:
                    if paper.id in self.processed_papers:
                        continue
                    
                    chunk_metadata_list = self.process_paper(paper)
                    
                    if not chunk_metadata_list:
                        logger.warning(f"No chunks created for paper {paper.id}")
                        self._save_processed_paper(paper.id)
                        continue
                    
                    # Store chunk data for batch processing
                    for metadata in chunk_metadata_list:
                        batch_chunks_data.append({
                            'text': metadata.chunk_text,
                            'metadata': metadata
                        })
                    
                    batch_paper_ids.append(paper.id)
                            
                except Exception as e:
                    logger.error(f"Error processing paper {paper.id}: {e}")
                    continue
            
            if not batch_chunks_data:
                logger.info(f"No chunks to process in batch {batch_start//processing_batch_size + 1}")
                continue
            
            logger.info(f"Batch {batch_start//processing_batch_size + 1}: Created {len(batch_chunks_data)} chunks from {len(batch_paper_ids)} papers")
            
            try:
                chunk_texts = [item['text'] for item in batch_chunks_data]
                processed_chunks = 0
                qdrant_upload_batch = []
                qdrant_batch_size = 100
                
                for embed_start in tqdm(range(0, len(chunk_texts), self.embedding_batch_size), 
                                      desc=f"Processing embeddings (batch {batch_start//processing_batch_size + 1})"):
                    embed_end = min(embed_start + self.embedding_batch_size, len(chunk_texts))
                    text_batch = chunk_texts[embed_start:embed_end]
                    
                    embeddings_batch = self.create_embeddings(text_batch)
                    
                    for i, embedding in enumerate(embeddings_batch):
                        global_idx = embed_start + i
                        metadata = batch_chunks_data[global_idx]['metadata']
                        
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding.tolist(),
                            payload={
                                "id": metadata.id,
                                "chunk_id": metadata.chunk_id,
                                "chunk_index": metadata.chunk_index,
                                "title": metadata.title,
                                "keywords": metadata.keywords,
                                "chunk_text": metadata.chunk_text,
                                "full_abstract": metadata.full_abstract,
                                "introduction": metadata.introduction,
                                "results": metadata.results,
                                "discussion": metadata.discussion,
                                "conclusion": metadata.conclusion
                            }
                        )
                        qdrant_upload_batch.append(point)
                        
                        # Upload when batch is full
                        if len(qdrant_upload_batch) >= qdrant_batch_size:
                            self.qdrant_client.upsert(
                                collection_name=self.collection_name,
                                points=qdrant_upload_batch
                            )
                            processed_chunks += len(qdrant_upload_batch)
                            qdrant_upload_batch = []
                
                # Upload any remaining points in the last batch
                if qdrant_upload_batch:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=qdrant_upload_batch
                    )
                    processed_chunks += len(qdrant_upload_batch)
                
                # Mark papers
                for paper_id in batch_paper_ids:
                    self._save_processed_paper(paper_id)
                
                total_papers_processed += len(batch_paper_ids)
                total_chunks_created += len(batch_chunks_data)
                
                logger.info(f"Batch {batch_start//processing_batch_size + 1} completed: "
                           f"{len(batch_paper_ids)} papers, {len(batch_chunks_data)} chunks")
                
                # Clear batch data
                del batch_chunks_data
                del chunk_texts
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start//processing_batch_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully processed {total_papers_processed} papers with {total_chunks_created} total chunks")
    
    def get_collection_info(self):
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "total_points": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def run_full_pipeline(self):
        try:
            start_time = datetime.now()
            logger.info("Starting embedding pipeline")
            
            # Load papers
            papers = self.load_papers()
            
            if not papers:
                logger.info("No new papers to process. Pipeline completed.")
                return
            
            # Process and store
            self.embed_and_store_papers(papers)
            
            # Getcollection info
            info = self.get_collection_info()
            if info:
                logger.info(f"Pipeline completed. Collection now contains {info['total_points']} embeddings")
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Total processing time: {duration}")
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user. Progress has been saved and can be resumed.")
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

if __name__ == "__main__":
    pipeline = EmbeddingPipeline(
        data_file="../../data/json_files/parsed_pmc_2.jsonl",
        max_tokens=500,
        collection_name="pmc_e5_base_sentence_based", #switch
        embedding_batch_size=32,
        device="cuda"
    ).run_full_pipeline()
