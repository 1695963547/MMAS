"""
MOE (Mixture of Experts) Expert System

Implements expert activation and weight assignment mechanism for Step 5：
- Semantic similarity computation (S_semantic)
- Keyword matching (S_keyword) 
- Question similarity (S_question)
- Option similarity (S_options)
- Weighted fusion and Softmax normalization
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import re

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence_transformers not available, using fallback similarity")

from global_model_manager import GlobalModelManager


class MOEExpertSystem:
    """MOE Expert Activation and Weight Assignment System"""
    
    def __init__(self, expert_knowledge_path: str, semantic_model_path: str, config: Dict[str, Any], model_manager: Optional[GlobalModelManager] = None):
        """
        Initialize MOE expert system
        
        Args:
            expert_knowledge_path: Expert knowledge graph file path
            semantic_model_path: Semantic model path (kept for compatibility, but uses global model manager)
            config: Configuration parameters
            model_manager: Global model manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Load expert knowledge graph
        self.expert_knowledge = self._load_expert_knowledge(expert_knowledge_path)
        
        # Load mechanism keyword knowledge base
        self.mechanism_keywords = self._load_mechanism_keywords()
        
        # 使用Global model manager
        self.model_manager = model_manager or GlobalModelManager()
        
        # Initializing 语义模型 - 使用Global model manager
        self.semantic_model = None
        self.semantic_model_path = semantic_model_path  # 保留兼容性
        self.preload_semantic_model()
        
        # 预编码expert知识
        self._pre_encode_experts()
        
        # WeightParameter (α, β, γ) - 废除δ(S_question)，引入γ(S_mechanism)
        self.alpha = config.get('alpha', config.get('semantic_weight', 0.4))  # 语义similarityWeight
        self.beta = config.get('beta', config.get('keyword_weight', 0.3))    # Keyword matchingWeight
        self.gamma = config.get('gamma', config.get('mechanism_weight', 0.3))  # 机制matchingWeight（替代Question similarity）
        
        # expert选择Parameter
        self.top_k = config.get('top_k_experts', 3)
        self.min_score_threshold = config.get('min_score_threshold', 0.1)
        self.temperature = config.get('temperature', 1.0)  # 温度Parameter，Defaultas1
        self.use_adaptive_threshold = config.get('use_adaptive_threshold', False)  # is否使用自适应Threshold，DefaultasFalse
        
        self.logger.info(f"MOEexpert systemInitializing complete，Loaded{len(self.expert_knowledge)}experts")
        
    def _load_mechanism_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load mechanism keyword knowledge base"""
        try:
            mechanism_keywords_path = Path(__file__).resolve().parent.parent / "data" / "mechanism_keywords.json"
            with open(mechanism_keywords_path, 'r', encoding='utf-8') as f:
                mechanism_keywords = json.load(f)
            self.logger.info(f"Successfully Loading Extension机制关键词知识库，包含{len(mechanism_keywords)}specialty")
            return mechanism_keywords
        except Exception as e:
            self.logger.error(f"Loading Extension机制关键词知识库Failed : {e}")
            return {}
        
    def _load_expert_knowledge(self, knowledge_path: str) -> List[Dict[str, Any]]:
        """Load expert knowledge graph"""
        try:
            with open(knowledge_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            # 如果isDictionary格式，提取expertList
            if isinstance(data, dict):
                experts = []
                for key, value in data.items():
                    if isinstance(value, dict) and 'specialty_name' in value:
                        # 复制expertData并添加文specialtyname
                        expert_data = value.copy()
                        expert_data['specialty_chinese_name'] = key  # 使用JSON键名作as文specialtyname
                        expert_data['expert_key'] = key  # 保留Original键名
                        expert_data['expert_name'] = key  # Setting expertnameas文specialtyname
                        experts.append(expert_data)
                return experts
            elif isinstance(data, list):
                return data
            else:
                self.logger.error(f"Not supported: expert知识graph谱格式: {type(data)}")
                return self._get_default_experts()
                
        except Exception as e:
            self.logger.error(f"Load expert knowledge graphFailed : {e}")
            return self._get_default_experts()
    
    def _get_default_experts(self) -> List[Dict[str, Any]]:
        """Getting DefaultexpertConfiguring """
        return [
            {
                "specialty_name": "General Medicine",
                "expert_name": "通用medicalexpert",
                "description": "通用medicaldiagnosisexpert",
                "core_competencies": ["diagnosis", "treatment", "预防"],
                "keywords": ["symptom", "diagnosis", "treatment"],
                "typical_questions_to_ask_patient": ["您has什么symptom？", "symptom持续多久？"],
                "specialty_chinese_name": "全科medical"
            }
        ]

    def preload_semantic_model(self):
        """预Loading 语义模型 - 使用Global model manager"""
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.semantic_model is None:
            try:
                self.semantic_model, _ = self.model_manager.get_semantic_model()
                self.logger.info(f"语义模型fromGlobal管理器Loading Successfully ")
            except Exception as e:
                self.logger.warning(f"语义模型Loading Failed : {e}")
                self.semantic_model = None

    def _pre_encode_experts(self):
        """预编码allexpertInformationwith加速similarityComputing """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.semantic_model is None:
            self.logger.warning("no法预编码expert，语义模型unavailable。")
            return

        self.logger.info("Starting 预编码expert知识...")
        for expert in self.expert_knowledge:
            # 编码expertText
            expert_text = self._build_expert_text(expert)
            expert['embedding'] = self.semantic_model.encode(expert_text, show_progress_bar=False)

            # 编码典型question
            questions = expert.get('typical_questions_to_ask_patient', [])
            if questions:
                if isinstance(questions, list):
                    expert['questions_embedding'] = self.semantic_model.encode(questions, show_progress_bar=False)
                else:
                    expert['questions_embedding'] = self.semantic_model.encode([str(questions)], show_progress_bar=False)
            
            # 编码specialtyText
            expert_specialty_text = ""
            if 'specialty_name' in expert:
                expert_specialty_text += expert['specialty_name'].lower() + " "
            if 'specialty_chinese_name' in expert:
                expert_specialty_text += expert['specialty_chinese_name'].lower() + " "
            if expert_specialty_text:
                expert['specialty_embedding'] = self.semantic_model.encode(expert_specialty_text.strip(), show_progress_bar=False)
        self.logger.info("expert知识预编码complete。")

    def activate_experts(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Activationexpert并分配Weight
        
        Args:
            case_data: caseData
            
        Returns:
            包含ActivationexpertandWeightDictionary
        """
        self.logger.info("Starting expertActivationandWeight分配")
        
        case_query_embedding = None
        case_specialty_embedding = None
        case_query = self._build_query_text(case_data)

        if SENTENCE_TRANSFORMERS_AVAILABLE and self.semantic_model is not None:
            # 1. 一times性编码case相关Information
            case_query_embedding = self.semantic_model.encode(case_query, show_progress_bar=False)
            
            # 构建并编码specialtymatchingText
            specialty_indicators = []
            for field in ['context', 'question', 'symptoms', 'diagnosis']:
                if field in case_data and case_data[field]:
                    specialty_indicators.append(str(case_data[field]).lower())
            case_specialty_text = " ".join(specialty_indicators)
            if case_specialty_text:
                case_specialty_embedding = self.semantic_model.encode(case_specialty_text, show_progress_bar=False)

        # Computing eachexperts得分
        expert_scores = []
        # 预编码查询Text供Keyword matching复用（避免eachexpertsDuplicate编码）
        case_query_embedding_for_kw = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.semantic_model is not None:
            case_query_embedding_for_kw = self.semantic_model.encode([case_query], show_progress_bar=False)
        
        for expert in self.expert_knowledge:
            score = self._calculate_expert_score(expert, case_data, case_query, case_query_embedding, case_specialty_embedding, case_query_embedding_for_kw)
            expert_scores.append({
                'expert': expert,
                'score': score
            })
           
        
        # per得分ranking
        expert_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 选择Top-Kexpert
        selected_experts = expert_scores[:self.top_k]

        # 根据Configuring 选择Threshold剪枝strategy
        if self.use_adaptive_threshold and len(selected_experts) > 1:
            # 自适应剪枝：Deleting Score < θ_dynexpert
            scores_for_threshold = np.array([exp['score'] for exp in selected_experts])
            mean_score = np.mean(scores_for_threshold)
            std_score = np.std(scores_for_threshold)
            dynamic_threshold = mean_score - 0.5 * std_score
            selected_experts = [exp for exp in selected_experts if exp['score'] >= dynamic_threshold]
            self.logger.info(f"使用自适应Threshold {dynamic_threshold:.3f} 进行剪枝")
        else:
            # Default固定Threshold剪枝
            selected_experts = [exp for exp in selected_experts if exp['score'] >= self.min_score_threshold]
            self.logger.info(f"使用固定Threshold {self.min_score_threshold} 进行剪枝")
        
        if not selected_experts:
            self.logger.warning("没hasexpert得分超过Threshold，使用得分最高expert")
            selected_experts = expert_scores[:1]  # 至少选择一experts
        
        # Checking is否as均etc.Weight消融实验
        use_equal_weights = self.config.get('ablation_equal_expert_weights', False)
        
        if use_equal_weights:
            # 均etc.Weight消融：allexpertWeight相etc.
            num_experts = len(selected_experts)
            weights = np.ones(num_experts) / num_experts
            self.logger.info(f"消融实验：使用均etc.Weight，eachexpertsWeight: {1.0/num_experts:.3f}")
        else:
            # 正常mode：带温度Softmax归一化Weight
            scores = np.array([exp['score'] for exp in selected_experts])
            # 应用温度Parameter
            scores_with_temp = scores / self.temperature
            # as防止数值溢出，减去最大值
            scores_with_temp -= np.max(scores_with_temp)
            exp_scores = np.exp(scores_with_temp)
            weights = exp_scores / np.sum(exp_scores)
        
        # 构建result
        activated_experts = []
        expert_weights = {}  # 添加缺失expert_weightsDictionary
        
        for i, expert_data in enumerate(selected_experts):
            # 传递completeexpertData，而不isCreating 不completeDictionary
            expert_info = expert_data['expert'].copy()  # 复制completeexpertData
            # 确保包含matching机制关键词
            expert_info['matched_mechanism_keywords'] = expert_data['expert'].get('matched_mechanism_keywords', [])
            
            activated_experts.append({
                'expert': expert_info,
                'weight': float(weights[i]),
                'score': float(expert_data['score'])
            })
            
            # 构建expert_weightsDictionary，使用expert文name作as键
            expert_name = expert_info.get('specialty_chinese_name') or expert_info.get('expert_name') or expert_info.get('specialty_name', 'Unknown')
            expert_weights[expert_name] = float(weights[i])
        
        result = {
            'activated_experts': activated_experts,
            'expert_weights': expert_weights,  # 添加缺失expert_weights字Segment
            'total_experts': len(activated_experts),
            'activation_summary': {
                'top_expert': activated_experts[0]['expert'].get('specialty_chinese_name', 
                                                               activated_experts[0]['expert'].get('expert_name', 
                                                                                                 activated_experts[0]['expert'].get('specialty_name', 'Unknown'))),
                'top_weight': float(weights[0]),
                'total_weight': float(np.sum(weights))
            }
        }
        
        self.logger.info(f"expertActivationcomplete，Activation{len(activated_experts)}experts")
        return result

    def _build_query_text(self, case_data: Dict[str, Any]) -> str:
        """
        构建查询Text，使用casecompleteInformation
        
        Args:
            case_data: caseData
            
        Returns:
            str: 构建查询Text
        """
        query_parts = []
        
        # question
        if 'question' in case_data and case_data['question']:
            query_parts.append(f"question: {case_data['question']}")
        
        # Context
        if 'context' in case_data and case_data['context']:
            if isinstance(case_data['context'], list):
                context_text = " ".join(case_data['context'])
                query_parts.append(f"Context: {context_text}")
            else:
                query_parts.append(f"Context: {case_data['context']}")
        
        # Option
        if 'options' in case_data and case_data['options']:
            if isinstance(case_data['options'], dict):
                options_text = " ".join([f"{k}: {v}" for k, v in case_data['options'].items()])
                query_parts.append(f"Option: {options_text}")
            elif isinstance(case_data['options'], list):
                options_text = " ".join(case_data['options'])
                query_parts.append(f"Option: {options_text}")
        
        # patientInformation
        if 'patient' in case_data and case_data['patient']:
            if isinstance(case_data['patient'], dict):
                patient_info = []
                for key, value in case_data['patient'].items():
                    if value:
                        # Processing Listtype值
                        if isinstance(value, list):
                            if key == 'specialties' or key == 'subspecialties':
                                value_str = ", ".join(str(v) for v in value)
                            else:
                                value_str = " ".join(str(v) for v in value)
                        else:
                            value_str = str(value)
                        patient_info.append(f"{key}: {value_str}")
                if patient_info:
                    query_parts.append(f"patientInformation: {' '.join(patient_info)}")
            else:
                query_parts.append(f"patientInformation: {case_data['patient']}")
        
        # 原子事实
        if 'atomic_facts' in case_data and case_data['atomic_facts']:
            if isinstance(case_data['atomic_facts'], list):
                facts_text = " ".join(str(fact) for fact in case_data['atomic_facts'])
                query_parts.append(f"关键事实: {facts_text}")
            else:
                query_parts.append(f"关键事实: {case_data['atomic_facts']}")
        
        # its他可能字Segment
        for key in ['symptoms', 'diagnosis', 'treatment', 'medical_history', 'examination']:
            if key in case_data and case_data[key]:
                if isinstance(case_data[key], list):
                    value_str = " ".join(str(v) for v in case_data[key])
                else:
                    value_str = str(case_data[key])
                query_parts.append(f"{key}: {value_str}")
        
        query_text = " ".join(query_parts)
        self.logger.debug(f"构建查询TextLength: {len(query_text)}")
        return query_text

    def _calculate_expert_score(self, expert_data: Dict[str, Any], case_data: Dict[str, Any], case_query: str, case_query_embedding: np.ndarray, case_specialty_embedding: np.ndarray, case_query_embedding_for_kw: np.ndarray = None) -> float:
        """
        Computing expert得分，使用预Computing 嵌入
        """
        # 1. 语义similarity (S_semantic)
        semantic_score = self._calculate_similarity_from_embeddings(case_query_embedding, expert_data.get('embedding'))
        
        # 2. Keyword matching (S_keyword) - 使用预编码查询Vector避免Duplicate编码
        keyword_score = self._calculate_keyword_matching(case_query, expert_data, case_query_embedding_for_kw)
        
        # 3. 机制matching度 (S_mechanism) - 替代原hasS_question
        # 修复ParameterOrder：应the传递expertname而不iscase_query作as第一Parameter
        expert_specialty = expert_data.get('specialty_chinese_name') or expert_data.get('expert_name') or expert_data.get('specialty_name', 'Unknown')
        mechanism_score, matched_keywords = self._calculate_mechanism_score(expert_specialty, case_query)
        
        # 加权融合 - 新三维评分公式
        total_score = (
            self.alpha * semantic_score +
            self.beta * keyword_score +
            self.gamma * mechanism_score
        )
        
        # 存储详细得分InformationtoexpertData
        expert_data['score_details'] = {
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'mechanism_score': mechanism_score,
            'total_score': total_score
        }
        
        # matching机制关键词存储toexpertData，供后续使用
        expert_data['matched_mechanism_keywords'] = matched_keywords
        
        return min(max(total_score, 0.0), 1.0)

    def _build_expert_text(self, expert_data: Dict[str, Any]) -> str:
        """构建expertText，使用JSONcomplete字SegmentInformation"""
        text_parts = []
        
        # specialtyname
        if 'specialty_name' in expert_data:
            text_parts.append(expert_data['specialty_name'])
        
        if 'specialty_chinese_name' in expert_data:
            text_parts.append(expert_data['specialty_chinese_name'])
        
        # expertname
        if 'expert_name' in expert_data:
            text_parts.append(expert_data['expert_name'])
        
        # 描述
        if 'description' in expert_data:
            text_parts.append(expert_data['description'])
        
        # 核心能力
        if 'core_competencies' in expert_data:
            competencies = expert_data['core_competencies']
            if isinstance(competencies, list):
                text_parts.extend(competencies)
            else:
                text_parts.append(str(competencies))
        
        # 关键词
        if 'keywords' in expert_data:
            keywords = expert_data['keywords']
            if isinstance(keywords, list):
                text_parts.extend(keywords)
            else:
                text_parts.append(str(keywords))
        
        # 典型question
        if 'typical_questions_to_ask_patient' in expert_data:
            questions = expert_data['typical_questions_to_ask_patient']
            if isinstance(questions, list):
                text_parts.extend(questions)
            else:
                text_parts.append(str(questions))
        
        return " ".join(text_parts)

    def _calculate_mechanism_score(self, expert_specialty: str, case_query: str) -> Tuple[float, List[str]]:
        """
        Computing 机制matching度得分，改进版implements。
        - 使用三Layer关键词structure (core_mechanisms, related_concepts, symptom_and_general)
        - for不同Layer关键词应用明确Weight：core_mechanisms(10.0), related_concepts(2.0), symptom_and_general(0.5)
        - 使用RegularizationExpression单词Boundary\b进行精确matching，避免partialmatching噪声
        - 使用math.log1p进行for数缩放，防止极端高分for最终result产生不成比例影响
        - return归一化得分andmatchingto关键词List
        """
        import math
        
        specialty_key = expert_specialty

        # 移除详细debuglog，只保留关键Information
        
        specialty_keywords_data = self.mechanism_keywords.get(specialty_key)
        if not specialty_keywords_data:
            return 0.0, []

        # 定义明确Weight，core_mechanismsWeight最高，symptom_and_generalWeight极低
        weights = {
            "core_mechanisms": 10.0,      # 核心机制Weight最高
            "related_concepts": 2.0,      # 相关概念etc.Weight
            "symptom_and_general": 0.5    # symptomand一般性词汇Weight极低
        }

        weighted_score = 0.0
        all_matched_keywords = []

        # 遍历三关键词Layer
        for category, weight in weights.items():
            keywords = specialty_keywords_data.get(category, [])
            if not keywords:
                continue

            # 使用RegularizationExpression单词Boundary\b进行精确matching，确保matchingcomplete单词or短语
            matched_for_category = []
            for keyword in keywords:
                # 使用\b单词Boundary确保精确matching
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, case_query, re.IGNORECASE):
                    matched_for_category.append(keyword)
            
            # 去重
            matched_for_category = list(set(matched_for_category))
            
            if matched_for_category:
                # 加权计分：scoreismatchingto关键词Weightofand
                category_weighted_score = len(matched_for_category) * weight
                weighted_score += category_weighted_score
                all_matched_keywords.extend(matched_for_category)

        # 使用math.log1p进行for数缩放，防止极端高分
        if weighted_score > 0:
            scaled_score = math.log1p(weighted_score)
            # 进一Step归一化to[0,1]Range，使用合理缩放因子
            # Hypothesis最大可能加权score约as50（5核心词*10 = 50），log1p(50) ≈ 3.93
            max_expected_log_score = math.log1p(50)
            final_score = min(scaled_score / max_expected_log_score, 1.0)
        else:
            final_score = 0.0

        return final_score, list(set(all_matched_keywords))

    def _calculate_similarity_from_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """通过预Computing 嵌入Computing 余弦similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        # Ensure embeddings are 2D
        if emb1.ndim == 1:
            emb1 = np.expand_dims(emb1, axis=0)
        if emb2.ndim == 1:
            emb2 = np.expand_dims(emb2, axis=0)
        return float(cosine_similarity(emb1, emb2)[0][0])

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Computing Textsimilarity (Fallback)"""
        if not text1 or not text2:
            return 0.0
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.semantic_model is not None:
            try:
                embeddings = self.semantic_model.encode([text1, text2], show_progress_bar=False)
                return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            except Exception as e:
                self.logger.warning(f"Semantic similarity computationFailed : {e}")
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_matching(self, case_query: str, expert_data: Dict[str, Any], precomputed_query_embedding: np.ndarray = None) -> float:
        """
        Computing Keyword matchingscore (S_keyword) - based on语义Vectorsimilarity算法
        
        Optimization：接受预编码查询Vector，避免eachexpertsDuplicate编码case_query
        """
        if not self.semantic_model or not case_query:
            self.logger.debug("语义模型not loadedor查询Textas空")
            return 0.0
        
        # Getting expert关键词（兼容多字Segment名）
        keywords = expert_data.get('keywords', []) or expert_data.get('expertise', [])
        
        # 添加expert英文and文specialtynameto关键词List
        specialty_name = expert_data.get('specialty_name')
        if specialty_name:
            keywords.append(specialty_name)
        specialty_chinese_name = expert_data.get('specialty_chinese_name')
        if specialty_chinese_name:
            keywords.append(specialty_chinese_name)

        if not keywords:
            expert_name = expert_data.get('specialty_chinese_name', expert_data.get('name', 'Unknown'))
            self.logger.debug(f"expert '{expert_name}' no关键词")
            return 0.0

        # 过滤valid关键词
        valid_keywords = [k.strip() for k in keywords if k and str(k).strip()]
        if not valid_keywords:
            expert_name = expert_data.get('specialty_chinese_name', expert_data.get('name', 'Unknown'))
            self.logger.debug(f"expert '{expert_name}' novalid关键词")
            return 0.0

        try:
            # 1. 使用预编码查询Vector（如果has），避免Duplicate编码
            if precomputed_query_embedding is not None:
                query_embedding = precomputed_query_embedding
            else:
                query_embedding = self.semantic_model.encode([case_query], show_progress_bar=False)
            
            # 2. 编码all关键词（文关键词）
            keyword_embeddings = self.semantic_model.encode(valid_keywords, show_progress_bar=False)
            
            # 3. Computing 余弦similarity
            similarities = cosine_similarity(query_embedding, keyword_embeddings)[0]
            
            # 4. 取最高similarity作asS_keyword得分
            max_similarity = float(max(similarities)) if len(similarities) > 0 else 0.0
            
            # 确保得分合理Range内
            score = max(0.0, min(1.0, max_similarity))
            
            return score
            
        except Exception as e:
            expert_name = expert_data.get('specialty_chinese_name', expert_data.get('name', 'Unknown'))
            self.logger.error(f"Computing expert '{expert_name}' 语义关键词得分Failed : {e}")
            return 0.0