import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from typing import List, Dict


class SemanticSimilarity:
    
    def __init__(self, similarity_model_name: str = None):
        if similarity_model_name is None:
            model_name = "intfloat/e5-large-v2"
        else:
            model_name = similarity_model_name
            
        try:
            self.similarity_model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            self.similarity_model = SentenceTransformer('intfloat/e5-large-v2')
            self.model_name = 'intfloat/e5-large-v2'
        
        self.approach_labels = {
            'no_rag': 'Vanilla LLM',
            'with_rag': 'RAG+ LLM',
            'fine_tuned': 'FT LLM',
            'fine_tuned_rag': 'FT&RAG+ LLM'
        }
    
    def load_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def rouge_l(self, reference: str, candidate: str) -> float:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure
    
    def process_dataset(self, data: List[Dict], approach_name: str) -> pd.DataFrame:
        results = []
        
        for i, item in enumerate(data):
            semantic_sim = self.semantic_similarity(
                item['ground_truth'], 
                item['llm_answer']
            )
            
            rouge_l = self.rouge_l(
                item['ground_truth'], 
                item['llm_answer']
            )
            
            result = {
                'question_id': item.get('question_id', ''),
                'question': item.get('question', ''),
                'ground_truth': item['ground_truth'],
                'llm_answer': item['llm_answer'],
                'approach': approach_name,
                'semantic_similarity': semantic_sim,
                'rouge_l_f1': rouge_l
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compare_approaches(self, 
                          no_rag_file: str, 
                          with_rag_file: str, 
                          fine_tuned_file: str, 
                          fine_tuned_rag_file: str) -> Dict:
        
        approaches = {
            'no_rag': no_rag_file,
            'with_rag': with_rag_file,
            'fine_tuned': fine_tuned_file,
            'fine_tuned_rag': fine_tuned_rag_file
        }
        
        dataframes = {}
        
        for approach, file_path in approaches.items():
            data = self.load_data(file_path)
            df = self.process_dataset(data, approach)
            dataframes[approach] = df
        
        combined_df = pd.concat(dataframes.values(), ignore_index=True)
        
        summary_stats = {}
        for approach in approaches.keys():
            df = dataframes[approach]
            summary_stats[approach] = {
                'total_samples': len(df),
                'avg_semantic_similarity': df['semantic_similarity'].mean(),
                'avg_rouge_l_f1': df['rouge_l_f1'].mean(),
                'semantic_similarity_std': df['semantic_similarity'].std(),
                'rouge_l_f1_std': df['rouge_l_f1'].std()
            }
        
        baseline = summary_stats['no_rag']
        improvements = {}
        
        for approach in ['with_rag', 'fine_tuned', 'fine_tuned_rag']:
            improvements[approach] = {
                'semantic_similarity_improvement': 
                    summary_stats[approach]['avg_semantic_similarity'] - 
                    baseline['avg_semantic_similarity'],
                'rouge_l_f1_improvement': 
                    summary_stats[approach]['avg_rouge_l_f1'] - 
                    baseline['avg_rouge_l_f1'],
            }
        
        return {
            'dataframes': dataframes,
            'combined_df': combined_df,
            'summary_stats': summary_stats,
            'improvements': improvements
        }
    
    def create_visualizations(self, results: Dict):
        combined_df = results['combined_df']
        
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 0.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white'
        })
        
        colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0']
        
        approach_metrics = {}
        for approach in combined_df['approach'].unique():
            df_subset = combined_df[combined_df['approach'] == approach]
            approach_metrics[approach] = {
                'semantic_similarity': df_subset['semantic_similarity'].mean(),
                'rouge_l_f1': df_subset['rouge_l_f1'].mean(),
                'response_time': df_subset['response_time'].mean()
            }
        
        sorted_sem = sorted(approach_metrics.keys(), 
                                    key=lambda x: approach_metrics[x]['semantic_similarity'])
        
        sorted_rouge = sorted(approach_metrics.keys(), 
                                        key=lambda x: approach_metrics[x]['rouge_l_f1'])
        
        
        # 1. Semantic Similarity Comparison
        plt.figure(figsize=(10, 6))
        sem_values = [approach_metrics[app]['semantic_similarity'] for app in sorted_sem]
        bars1 = plt.bar(range(len(sorted_sem)), sem_values, 
                    color=colors[:len(sorted_sem)], 
                    edgecolor='navy', linewidth=0.5)
        plt.title('Semantic Similarity Comparison', fontweight='normal', fontsize=14, pad=20)
        plt.ylabel('Cosine Similarity Score', fontsize=12)
        plt.xticks(range(len(sorted_sem)), 
                   [self.approach_labels[app] for app in sorted_sem], 
                   rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)
        
        #Value labels
        for bar, value in zip(bars1, sem_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sem_values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('semantic_similarity_comparison.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
        # 2. ROUGE-L F1 Comparison
        plt.figure(figsize=(10, 6))
        rouge_values = [approach_metrics[app]['rouge_l_f1'] for app in sorted_rouge]
        bars2 = plt.bar(range(len(sorted_rouge)), rouge_values, 
                    color=colors[:len(sorted_rouge)], 
                    edgecolor='navy', linewidth=0.5)
        plt.title('ROUGE-L F1 Score Comparison', fontweight='normal', fontsize=14, pad=20)
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(range(len(sorted_rouge)), 
                   [self.approach_labels[app] for app in sorted_rouge], 
                   rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)
        
        for bar, value in zip(bars2, rouge_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rouge_values) * 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('rouge_l_f1_comparison.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
        
        # 3. Semantic Similarity Distribution
        plt.figure(figsize=(10, 6))
        sem_data = []
        sem_labels = []
        for approach in sorted_sem:
            data = combined_df[combined_df['approach'] == approach]['semantic_similarity']
            sem_data.append(data)
            sem_labels.append(self.approach_labels[approach])
        
        bp1 = plt.boxplot(sem_data, labels=sem_labels, patch_artist=True, 
                        medianprops={'color': 'navy', 'linewidth': 1})
        for patch, color in zip(bp1['boxes'], colors[:len(sem_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('navy')
        plt.title('Semantic Similarity Distribution', fontweight='normal', fontsize=14, pad=20)
        plt.ylabel('Cosine Similarity Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('semantic_similarity_distribution.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
        # 4. ROUGE-L F1 Distribution
        plt.figure(figsize=(10, 6))
        rouge_data = []
        rouge_labels = []
        for approach in sorted_rouge:
            data = combined_df[combined_df['approach'] == approach]['rouge_l_f1']
            rouge_data.append(data)
            rouge_labels.append(self.approach_labels[approach])
        
        bp2 = plt.boxplot(rouge_data, labels=rouge_labels, patch_artist=True,
                        medianprops={'color': 'navy', 'linewidth': 1})
        for patch, color in zip(bp2['boxes'], colors[:len(rouge_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('navy')
        plt.title('ROUGE-L F1 Distribution', fontweight='normal', fontsize=14, pad=20)
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rouge_l_f1_distribution.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
def main():
    benchmark = SemanticSimilarity()
    
    results = benchmark.compare_approaches(
        no_rag_file='pure_answers_temp01.jsonl',
        with_rag_file='rag_answers_temp01.jsonl',
        fine_tuned_file='fine_answers_temp01.jsonl',
        fine_tuned_rag_file='fine_rag_answers_temp01.jsonl'
    )
    benchmark.create_visualizations(results)

if __name__ == "__main__":
    main()