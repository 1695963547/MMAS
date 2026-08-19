"""
预计算专家和机制关键词的向量embeddings
一次性运行脚本，生成并保存所有向量到文件中，用于优化专家匹配系统的性能
"""

import json
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingPrecomputer:
    """向量预计算器"""
    
    def __init__(self):
        """初始化预计算器"""
        # 文件路径配置（脚本位于 scripts/，数据位于上级 data/）
        self._script_dir = Path(__file__).parent
        self._data_dir = self._script_dir.parent / 'data'
        self.expert_knowledge_path = self._data_dir / 'dev_dataset_expert_knowledge_graph_enhanced.json'
        self.mechanism_keywords_path = self._data_dir / 'mechanism_keywords.json'
        self.semantic_model_path = 'paraphrase-multilingual-MiniLM-L12-v2'  # HuggingFace 模型名，自动下载
        
        # 输出文件路径
        self.expert_embeddings_path = self._data_dir / 'precomputed_expert_embeddings.pkl'
        self.mechanism_embeddings_path = self._data_dir / 'precomputed_mechanism_embeddings.pkl'
        
        # 加载语义模型
        logger.info(f"正在加载语义模型: {self.semantic_model_path}")
        self.sentence_model = SentenceTransformer(self.semantic_model_path)
        logger.info("语义模型加载完成")
        
    def load_expert_knowledge(self) -> Dict[str, Any]:
        """加载专家知识图谱"""
        logger.info(f"正在加载专家知识图谱: {self.expert_knowledge_path}")
        with open(self.expert_knowledge_path, 'r', encoding='utf-8-sig') as f:
            knowledge_data = json.load(f)
        logger.info(f"成功加载 {len(knowledge_data)} 位专家的知识图谱")
        return knowledge_data
    
    def load_mechanism_keywords(self) -> Dict[str, Any]:
        """加载机制关键词"""
        logger.info(f"正在加载机制关键词: {self.mechanism_keywords_path}")
        with open(self.mechanism_keywords_path, 'r', encoding='utf-8-sig') as f:
            mechanism_data = json.load(f)
        logger.info(f"成功加载 {len(mechanism_data)} 个专家的机制关键词")
        return mechanism_data
    
    def precompute_expert_embeddings(self, expert_knowledge: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """预计算专家向量embeddings"""
        logger.info("开始预计算专家向量embeddings...")
        expert_embeddings = {}
        
        total_experts = len(expert_knowledge)
        for idx, (expert_key, expert_info) in enumerate(expert_knowledge.items(), 1):
            logger.info(f"处理专家 {idx}/{total_experts}: {expert_key}")
            
            expert_embeddings[expert_key] = {}
            
            # 1. 预计算专家描述向量（用于语义相似度计算）
            expert_desc_parts = []
            
            specialty_name = expert_info.get('specialty_name', '')
            if specialty_name:
                expert_desc_parts.append(specialty_name)
            
            description = expert_info.get('description', '')
            if description:
                expert_desc_parts.append(description)
            
            core_competencies = expert_info.get('core_competencies', [])
            if core_competencies:
                expert_desc_parts.append(' '.join(core_competencies))
            
            expert_desc = ' '.join(expert_desc_parts)
            
            if expert_desc:
                desc_embedding = self.sentence_model.encode([expert_desc], show_progress_bar=False)[0]
                expert_embeddings[expert_key]['description_embedding'] = desc_embedding
                logger.debug(f"  - 描述向量维度: {desc_embedding.shape}")
            
            # 2. 预计算关键词向量（用于关键词匹配）
            keywords = expert_info.get('keywords', []) or expert_info.get('expertise', [])
            
            # 添加专家的英文和中文专科名称到关键词列表
            specialty_name = expert_info.get('specialty_name')
            if specialty_name:
                keywords.append(specialty_name)
            specialty_chinese_name = expert_info.get('specialty_chinese_name')
            if specialty_chinese_name:
                keywords.append(specialty_chinese_name)
            
            # 过滤有效关键词
            valid_keywords = [k.strip() for k in keywords if k and str(k).strip()]
            
            if valid_keywords:
                keyword_embeddings = self.sentence_model.encode(valid_keywords, show_progress_bar=False)
                expert_embeddings[expert_key]['keyword_embeddings'] = keyword_embeddings
                expert_embeddings[expert_key]['keywords'] = valid_keywords
                logger.debug(f"  - 关键词数量: {len(valid_keywords)}, 向量维度: {keyword_embeddings.shape}")
            
            # 3. 预计算问题向量（用于问题匹配）
            questions = expert_info.get('questions', [])
            if questions:
                question_embeddings = self.sentence_model.encode(questions, show_progress_bar=False)
                expert_embeddings[expert_key]['question_embeddings'] = question_embeddings
                expert_embeddings[expert_key]['questions'] = questions
                logger.debug(f"  - 问题数量: {len(questions)}, 向量维度: {question_embeddings.shape}")
        
        logger.info(f"专家向量预计算完成，共处理 {len(expert_embeddings)} 位专家")
        return expert_embeddings
    
    def precompute_mechanism_embeddings(self, mechanism_keywords: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """预计算机制关键词向量embeddings"""
        logger.info("开始预计算机制关键词向量embeddings...")
        mechanism_embeddings = {}
        
        total_experts = len(mechanism_keywords)
        for idx, (expert_key, mechanism_info) in enumerate(mechanism_keywords.items(), 1):
            logger.info(f"处理专家机制关键词 {idx}/{total_experts}: {expert_key}")
            
            mechanism_embeddings[expert_key] = {}
            
            # 预计算核心机制向量
            core_mechanisms = mechanism_info.get('core_mechanisms', [])
            if core_mechanisms:
                core_embeddings = self.sentence_model.encode(core_mechanisms, show_progress_bar=False)
                mechanism_embeddings[expert_key]['core_mechanism_embeddings'] = core_embeddings
                mechanism_embeddings[expert_key]['core_mechanisms'] = core_mechanisms
                logger.debug(f"  - 核心机制数量: {len(core_mechanisms)}, 向量维度: {core_embeddings.shape}")
            
            # 预计算相关概念向量
            related_concepts = mechanism_info.get('related_concepts', [])
            if related_concepts:
                concept_embeddings = self.sentence_model.encode(related_concepts, show_progress_bar=False)
                mechanism_embeddings[expert_key]['related_concept_embeddings'] = concept_embeddings
                mechanism_embeddings[expert_key]['related_concepts'] = related_concepts
                logger.debug(f"  - 相关概念数量: {len(related_concepts)}, 向量维度: {concept_embeddings.shape}")
        
        logger.info(f"机制关键词向量预计算完成，共处理 {len(mechanism_embeddings)} 位专家")
        return mechanism_embeddings
    
    def save_embeddings(self, expert_embeddings: Dict, mechanism_embeddings: Dict):
        """保存预计算的向量到文件"""
        logger.info("正在保存预计算的向量到文件...")
        
        # 保存专家向量
        with open(self.expert_embeddings_path, 'wb') as f:
            pickle.dump(expert_embeddings, f)
        logger.info(f"专家向量已保存到: {self.expert_embeddings_path}")
        
        # 保存机制关键词向量
        with open(self.mechanism_embeddings_path, 'wb') as f:
            pickle.dump(mechanism_embeddings, f)
        logger.info(f"机制关键词向量已保存到: {self.mechanism_embeddings_path}")
        
        # 输出文件大小信息
        expert_size = Path(self.expert_embeddings_path).stat().st_size / (1024 * 1024)
        mechanism_size = Path(self.mechanism_embeddings_path).stat().st_size / (1024 * 1024)
        logger.info(f"文件大小 - 专家向量: {expert_size:.2f} MB, 机制关键词向量: {mechanism_size:.2f} MB")
    
    def run_precomputation(self):
        """运行完整的预计算流程"""
        start_time = time.time()
        logger.info("="*60)
        logger.info("开始向量预计算流程")
        logger.info("="*60)
        
        try:
            # 1. 加载数据
            expert_knowledge = self.load_expert_knowledge()
            mechanism_keywords = self.load_mechanism_keywords()
            
            # 2. 预计算专家向量
            expert_embeddings = self.precompute_expert_embeddings(expert_knowledge)
            
            # 3. 预计算机制关键词向量
            mechanism_embeddings = self.precompute_mechanism_embeddings(mechanism_keywords)
            
            # 4. 保存向量到文件
            self.save_embeddings(expert_embeddings, mechanism_embeddings)
            
            # 5. 输出统计信息
            total_time = time.time() - start_time
            logger.info("="*60)
            logger.info("向量预计算完成！")
            logger.info(f"总耗时: {total_time:.2f} 秒")
            logger.info(f"专家数量: {len(expert_embeddings)}")
            logger.info(f"机制关键词专家数量: {len(mechanism_embeddings)}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"向量预计算失败: {e}")
            return False

def main():
    """主函数"""
    precomputer = EmbeddingPrecomputer()
    success = precomputer.run_precomputation()
    
    if success:
        print("\n✅ 向量预计算成功完成！")
        print("现在可以修改 expert_matching_system.py 以使用预计算的向量。")
    else:
        print("\n❌ 向量预计算失败，请检查日志信息。")

if __name__ == "__main__":
    main()