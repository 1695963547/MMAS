import json
import numpy as np
import pickle
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntentVectorPrecomputer:
    def __init__(self):
        self._script_dir = Path(__file__).parent
        self._data_dir = self._script_dir.parent / 'data'
        self.intent_keywords_path = self._data_dir / 'intent_keywords.json'
        self.semantic_model_path = 'paraphrase-multilingual-MiniLM-L12-v2'  # HuggingFace 模型名，自动下载
        self.output_path = self._data_dir / 'intent_prototype_vectors.pkl'
        
        logger.info(f"Loading semantic model: {self.semantic_model_path}")
        self.sentence_model = SentenceTransformer(self.semantic_model_path)
        logger.info("Semantic model loaded successfully.")

    def load_intent_keywords(self) -> dict:
        logger.info(f"Loading intent keywords from: {self.intent_keywords_path}")
        with open(self.intent_keywords_path, 'r', encoding='utf-8') as f:
            intent_keywords = json.load(f)
        logger.info(f"Successfully loaded {len(intent_keywords)} intents.")
        return intent_keywords

    def precompute_intent_vectors(self, intent_keywords: dict) -> dict:
        logger.info("Starting intent vector precomputation...")
        intent_vectors = {}
        
        for intent, keywords in intent_keywords.items():
            logger.info(f"Processing intent: {intent}")
            if not keywords:
                logger.warning(f"No keywords found for intent: {intent}. Skipping.")
                continue
            
            keyword_embeddings = self.sentence_model.encode(keywords, show_progress_bar=False)
            prototype_vector = np.mean(keyword_embeddings, axis=0)
            intent_vectors[intent] = prototype_vector
            logger.debug(f"  - Prototype vector dimension: {prototype_vector.shape}")
            
        logger.info(f"Intent vector precomputation complete. Processed {len(intent_vectors)} intents.")
        return intent_vectors

    def save_intent_vectors(self, intent_vectors: dict):
        logger.info("Saving intent prototype vectors to file...")
        with open(self.output_path, 'wb') as f:
            pickle.dump(intent_vectors, f)
        logger.info(f"Intent vectors saved to: {self.output_path}")
        
        output_size = Path(self.output_path).stat().st_size / 1024
        logger.info(f"File size: {output_size:.2f} KB")

    def run_precomputation(self):
        start_time = time.time()
        logger.info("="*60)
        logger.info("Starting intent vector precomputation process")
        logger.info("="*60)
        
        try:
            intent_keywords = self.load_intent_keywords()
            intent_vectors = self.precompute_intent_vectors(intent_keywords)
            self.save_intent_vectors(intent_vectors)
            
            total_time = time.time() - start_time
            logger.info("="*60)
            logger.info("Intent vector precomputation successful!")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Number of intents processed: {len(intent_vectors)}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Intent vector precomputation failed: {e}")
            return False

def main():
    precomputer = IntentVectorPrecomputer()
    success = precomputer.run_precomputation()
    
    if success:
        print("\n✅ Intent prototype vectors precomputation completed successfully!")
        print("The 'intent_prototype_vectors.pkl' file has been created.")
    else:
        print("\n❌ Intent prototype vectors precomputation failed. Please check the logs.")

if __name__ == "__main__":
    main()