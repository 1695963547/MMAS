#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMAS Three-Agent Medical Diagnosis System - Main Runner
Multi-agent collaborative diagnosis system

Main features:
1. Coordinate Patient Agent, Evaluator Agent, and Doctor Agent
2. Implement complete 7-step diagnosis pipeline
3. Support MOE expert system and weight computation
4. Provide batch processing and performance statistics
"""

import argparse
import json
import traceback
import logging
import time
import warnings
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Error recording
def record_error(error_type: str, error_msg: str, case_id: str = None, step: int = None):
    """Record error info to log and error statistics"""
    logger = logging.getLogger(__name__)
    
    error_info = {
        'timestamp': datetime.now().isoformat(),
        'error_type': error_type,
        'error_message': error_msg,
        'case_id': case_id,
        'step': step
    }
    
    # Log to file
    logger.error(f"[Error record] {error_type}: {error_msg} (case: {case_id}, step: {step})")
    
    # Can be extended to write error statistics file
    return error_info

# Add project path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*unknown prompting method.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

# Set environment variables
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

# 导入新agentmodule
from global_model_manager import GlobalModelManager
from unified_expert_template_manager import get_unified_expert_template_manager
from mmas_patient_agent import MMASPatientAgent
from mmas_evaluator_agent import MMASEvaluatorAgent
from mmas_doctor_agent import MMASDoctorAgent
from moe_expert_system import MOEExpertSystem
from api_model_client import get_api_client

class NumpyEncoder(json.JSONEncoder):
    """CustomJSON编码器，Processing numpytype"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 导入partial添加log管理器
from log_manager import start_system_logging, stop_system_logging

class MMASThreeAgentSystem:
    """
    MMAS三agentsystem主控制器
    
    负责coordinatethree agents工作pipeline：
    1. Patient Agent - Patient Agent
    2. Evaluator Agent - Evaluator Agent  
    3. Doctor Agent - Doctor Agent
    """
    
    def __init__(self, args):
        """Initializing MMAS三agentsystem"""
        self.args = args
        self.start_time = time.time()
        self.total_cases = 0
        self.correct_cases = 0
        
        # 提取Model nameandData集name，用于logfile命名
        self.model_name, self.dataset_name = self._extract_names_from_args(args)
        
        # Set up logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initializing Error recordlogfile
        self._setup_error_logging()
        
        # Checkpoint resume相关Attribute
        self.checkpoint_file = 'checkpoint.json'
        self.use_resume = args.resume if hasattr(args, 'resume') else False
        self.checkpoint_interval = args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 1
        
        # Checking is否需要Checkpoint resume，并Getting logfilepath
        last_processed_index, resume_log_file = self._load_checkpoint()
        
        # Starting systemlogRecord（supportCheckpoint resume，file名包含模型名andData集名）
        self.log_manager = start_system_logging(
            resume_log_file=resume_log_file,
            model_name=self.model_name,
            dataset_name=self.dataset_name
        )
        
        # Initializing statisticsInformation
        self.timing_stats = {
            'initialization_time': 0.0,
            'model_loading_time': 0.0,
            'dataset_loading_time': 0.0,
            'case_processing_times': [],
            'total_processing_time': 0.0
        }
        
        self.performance_stats = {
            'total_cases': 0,
            'correct_cases': 0,
            'accuracy': 0.0,
            'average_case_time': 0.0,
            'total_processing_time': 0.0
        }
        
        self.agent_stats = {
            'patient_agent': {'active': False, 'last_used': None},
            'evaluator_agent': {'active': False, 'last_used': None},
            'doctor_agent': {'active': False, 'last_used': None}
        }
        
        # Initializing Component（只Initializing 一times）
        self._initialize_components()
        self.logger.info("MMAS three-agent system initialization complete")
    
    def _setup_logging(self):
        """Set up loggingConfiguring """
        # LogManager会统一Processing alllogOutput，这里只需要Setting 基thislogLevel
        # 不再需要DuplicateConfiguring handlers，因asLogManager会接管allOutput
        logging.getLogger().setLevel(logging.DEBUG)
        
        # 确保transformersetc.库logLevel保持asERROR，减少噪音
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
    
    def _extract_names_from_args(self, args) -> Tuple[str, str]:
        """fromCommand行Parameter提取Model nameandData集name，用于file命名"""
        # 提取Model name
        if args.use_api_model:
            model_name = args.api_model_name
        else:
            # from模型path提取file夹名作asModel name
            model_name = Path(args.model_path).name
        
        # 提取Data集name（file名，不含Extension名）
        dataset_name = Path(args.dataset_path).stem
        
        return model_name, dataset_name
    
    def _setup_error_logging(self):
        """Setting Error recordlogfile"""
        try:
            # Creating incorrectcaselogdirectory
            error_cases_dir = Path("./log/error_cases")
            error_cases_dir.mkdir(parents=True, exist_ok=True)
            
            # Creating Error case IDlogfile（包含模型名andData集名）
            timestamp = datetime.now().strftime("%Y%m%d")
            self.error_case_log_file = error_cases_dir / f"error_case_ids_{self.model_name}_{self.dataset_name}_{timestamp}.log"
            
            self.logger.info(f"Error case log file: {self.error_case_log_file}")
            
        except Exception as e:
            self.logger.warning(f"Cannot create error case log file: {e}")
            self.error_case_log_file = None
    
    def _record_error_case_id(self, case_id: str):
        """RecordError case IDto专门logfile"""
        if self.error_case_log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.error_case_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] Error case ID: {case_id}\n")
                self.logger.info(f"Recorded error case ID: {case_id}")
            except Exception as e:
                self.logger.error(f"Failed to record error case ID: {e}")
    
    def _initialize_components(self):
        """Initializing systemComponent"""
        try:
            init_start = time.time()
            
            # Initializing 模型管理器
            print("[Initializing ] 模型管理器...", flush=True)
            self.model_manager = GlobalModelManager()
            print("[Initializing ] 模型管理器 OK", flush=True)
            
            # Configure quantization settings
            if hasattr(self.args, 'quantization') and self.args.quantization != 'none':
                print(f"[Initializing ] Configuring {self.args.quantization}量化...")
                self.model_manager.configure_quantization(self.args.quantization)
            
            # 只使用this地模型时Configuring 主LLM预Loading 
            if not self.args.use_api_model:
                # Configuring 模型预Loading 
                main_llm_path = Path(self.args.model_path).resolve().as_posix()
                print(f"[Initializing ] resolvepath OK: {main_llm_path}", flush=True)
                self.model_manager.configure_preload_models(main_llm_path=main_llm_path)
                print(f"[Initializing ] this地模型预Loading : {main_llm_path}", flush=True)
            else:
                # Using API model时，Disable 主LLM预Loading ，只Configuring 语义模型预Loading 
                semantic_model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model_manager.configure_preload_models(
                    semantic_model_path=semantic_model_path, 
                    disable_main_llm=True
                )
                print("[Initializing ] API mode, semantic model preload configured")
            
            # Initialize template manager
            print("[Initializing ] template管理器...")
            # 消融实验：Generic CoT template
            force_generic_cot = getattr(self.args, 'ablation_generic_cot', False)
            self.template_manager = get_unified_expert_template_manager(force_generic_cot=force_generic_cot)
            
            # Initialize MOE expert system
            print("[Initializing ] MOEexpert system（Loading 知识graph谱+语义模型+预编码）...")
            expert_knowledge_path = self.args.expert_knowledge_path
            semantic_model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            moe_config = {
                'alpha': getattr(self.args, 'alpha', 0.4), 
                'beta': getattr(self.args, 'beta', 0.3), 
                'gamma': getattr(self.args, 'gamma', 0.3),
                'top_k_experts': getattr(self.args, 'top_k', 3),
                'min_score_threshold': 0.1,  # 固定Threshold，自适应时会覆盖
                'use_adaptive_threshold': self.args.use_adaptive_threshold, # fromCommand行ParameterGetting 
                'temperature': self.args.temperature, # fromCommand行ParameterGetting 
                'ablation_equal_expert_weights': getattr(self.args, 'ablation_equal_expert_weights', False)
            }
            self.moe_expert_system = MOEExpertSystem(expert_knowledge_path, semantic_model_path, moe_config, self.model_manager)
            print("[Initializing ] MOEexpert systemcomplete")
            
            # Initializing expert映射functionality（用于直接传递expert标签）
            if self.args.use_ground_truth_experts:
                print("[Initializing ] expert映射functionality...")
                self._initialize_expert_mapping()
            
            # Initializing three agents
            print("[Initializing ] agent...")
            
            # Initializing API client（如果Using API model）
            api_client = None
            if self.args.use_api_model:
                print(f"[Initializing ] API client: {self.args.api_model_name}")
                api_client = get_api_client()
                
                # 如果provideAPI密钥，Updating Configuring 
                if self.args.api_key:
                    from api_config import update_api_key
                    update_api_key(self.args.api_model_name, self.args.api_key)
                    print(f"[Initializing ] updated{self.args.api_model_name}API密钥")
            
            # Initialize patient agent，传递API相关Parameter
            self.patient_agent = MMASPatientAgent(
                api_client=api_client,
                use_api_model=self.args.use_api_model,
                api_model_name=self.args.api_model_name if self.args.use_api_model else None
            )
            
            # Initializing Evaluator Agent，传递API相关Parameter
            self.evaluator_agent = MMASEvaluatorAgent(
                template_manager=self.template_manager,
                show_step2_details=self.args.show_step2_details,
                clarification_threshold=self.args.clarification_threshold,
                api_client=api_client,
                use_api_model=self.args.use_api_model,
                api_model_name=self.args.api_model_name if self.args.use_api_model else None
            )
            
            # Initializing Doctor Agent，传递API相关Parameter
            self.doctor_agent = MMASDoctorAgent(
                model_manager=self.model_manager, 
                template_manager=self.template_manager,
                api_client=api_client,
                use_api_model=self.args.use_api_model,
                api_model_name=self.args.api_model_name if self.args.use_api_model else None,
                use_semantic_fallback=getattr(self.args, 'use_semantic_fallback', False)
            )
            
            self.timing_stats['initialization_time'] = time.time() - init_start
            print(f"[Initializing ] allcomplete，耗时: {self.timing_stats['initialization_time']:.2f}seconds")
            
        except Exception as e:
            print(f"[Initializing ] Failed : {e}")
            import traceback; traceback.print_exc()
            raise
    
    def _initialize_expert_mapping(self):
        """Initializing expert映射functionality，英文specialtyname映射to文expertname"""
        try:
            # fromexpert知识graph谱Loading 映射关系，使用utf-8-sig编码Processing BOM
            with open(self.args.expert_knowledge_path, 'r', encoding='utf-8-sig') as f:
                expert_knowledge = json.load(f)
            
            # Creating 英文specialtynameto文expertname映射
            self.specialty_to_expert_mapping = {}
            for chinese_expert_name, expert_info in expert_knowledge.items():
                if 'specialty_name' in expert_info:
                    english_specialty = expert_info['specialty_name']
                    self.specialty_to_expert_mapping[english_specialty] = chinese_expert_name
            
            # 添加Data集specialtynameAlias mapping，Processing 不matching情况
            specialty_aliases = {
                # Data集name -> expert知识graph谱for应name
                'Internal Medicine': 'General Medicine',  # 内科 -> 全科medical
                'Pediatrics': '儿科学',  # 儿科 -> 儿科学expert
                'Neurology': '神经内科',  # 神经内科 -> 神经内科expert
                'Urology': '泌尿外科',  # 泌尿外科 -> 泌尿外科expert
                'Family Medicine': 'General Practice',  # 家庭medical -> 全科medical
                'Cardiovascular Disease': 'Cardiology',  # 心血管disease -> 心血管内科
                'Endocrinology, Diabetes, and Metabolism': 'Endocrinology',  # 内分泌diabetes代谢 -> 内分泌科
                'Geriatrics': 'Geriatric Medicine',  # 老年medical -> 老年medical科
                'Microbiology': 'Clinical Microbiology',  # 微生物学 -> clinical微生物学
                'Colorectal Surgery': 'Colon and Rectal Surgery',  # 结直肠外科 -> 结肠直肠外科
                'Neurological Surgery': 'Neurosurgery',  # 神经外科 -> 神经外科
                'Pulmonology': 'Pulmonary Disease',  # 肺科 -> 肺部disease
                'Statistics': 'Medical Statistics',  # statistics学 -> medicalstatistics学
            }
            
            # 应用Alias mapping
            for dataset_name, expert_graph_name in specialty_aliases.items():
                if expert_graph_name in self.specialty_to_expert_mapping:
                    # 找tofor应文expertname
                    chinese_expert_name = self.specialty_to_expert_mapping[expert_graph_name]
                    self.specialty_to_expert_mapping[dataset_name] = chinese_expert_name
                    self.logger.info(f"Add alias映射: {dataset_name} -> {expert_graph_name} -> {chinese_expert_name}")
                else:
                    self.logger.warning(f"Alias mapping failed, expert not found: {dataset_name} -> {expert_graph_name}")
            
            self.logger.info(f"Expert mapping initialization complete，共Loading  {len(self.specialty_to_expert_mapping)} specialty mappings (including aliases)")
            
        except Exception as e:
            self.logger.error(f"Expert mapping initialization failed: {e}")
            raise
    
    def _get_ground_truth_experts(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """fromData集Getting standard答案expert标签，直接使用Data集字Segment作asexpertname"""
        try:
            # Getting patientInformationspecialty标签
            patient_info = case_data.get('patient', {})
            specialties = patient_info.get('specialties', [])
            subspecialties = patient_info.get('subspecialties', [])
            specialty_probs = patient_info.get('specialty_prob', [])
            subspecialty_probs = patient_info.get('subspecialty_prob', [])
            
            # 直接使用Data集字Segment构建activated_experts_info格式
            activated_experts_info = []
            expert_weights = {}
            
            # Processing specialty（specialties）- 直接使用Data集specialtyname
            for i, specialty in enumerate(specialties):
                # 使用specialty_prob作asWeight，如果没has则Defaultas1.0
                weight = specialty_probs[i] if i < len(specialty_probs) else 1.0
                
                # 构建andMOEsystem兼容Datastructure，直接使用Data集字Segment
                expert_info = {
                    'expert': {
                        'specialty_chinese_name': specialty,  # 直接使用Data集specialtyname
                        'specialty_english_name': specialty,  # 同样使用Data集name
                        'score_details': {
                            'semantic_score': 1.0,  # standard答案，设as满分
                            'keyword_score': 1.0,
                            'mechanism_score': 1.0
                        }
                    },
                    'score': 3.0,  # 总分as3.0（三Dimensioneach1.0）
                    'weight': weight,
                    'activation_reason': f'Ground truth specialty from dataset: {specialty} (prob: {weight})'
                }
                
                activated_experts_info.append(expert_info)
                expert_weights[specialty] = weight
                self.logger.info(f"Activationspecialtyexpert: {specialty} (Weight: {weight})")
            
            # Processing 亚specialty（subspecialties）- 直接使用Data集亚specialtyname
            for i, subspecialty in enumerate(subspecialties):
                # 使用subspecialty_prob作asWeight，如果没has则Defaultas1.0
                weight = subspecialty_probs[i] if i < len(subspecialty_probs) else 1.0
                
                # Checking is否已经Activation这experts（避免Duplicate）
                if subspecialty not in expert_weights:
                    expert_info = {
                        'expert': {
                            'specialty_chinese_name': subspecialty,  # 直接使用Data集亚specialtyname
                            'specialty_english_name': subspecialty,  # 同样使用Data集name
                            'score_details': {
                                'semantic_score': 1.0,
                                'keyword_score': 1.0,
                                'mechanism_score': 1.0
                            }
                        },
                        'score': 3.0,
                        'weight': weight,
                        'activation_reason': f'Ground truth subspecialty from dataset: {subspecialty} (prob: {weight})'
                    }
                    
                    activated_experts_info.append(expert_info)
                    expert_weights[subspecialty] = weight
                    self.logger.info(f"Activation亚specialtyexpert: {subspecialty} (Weight: {weight})")
                else:
                    # 如果expertalready exists，Updating Weight（取较大值or累加）
                    existing_weight = expert_weights[subspecialty]
                    new_weight = max(existing_weight, weight)  # 取较大Weight
                    expert_weights[subspecialty] = new_weight
                    # Updating for应expert_infoWeight
                    for expert_info in activated_experts_info:
                        if expert_info['expert']['specialty_chinese_name'] == subspecialty:
                            expert_info['weight'] = new_weight
                            expert_info['activation_reason'] += f" + {subspecialty} (prob: {weight})"
                            break
                    self.logger.info(f"Updating expertWeight: {subspecialty} (新Weight: {new_weight})")
            
            # Limitexpert数量as最多3，perWeightranking选择前3
            # 消融实验：Top-1expertActivation时Skipping thisLimit
            if not getattr(self.args, 'ablation_top1', False) and len(activated_experts_info) > 3:
                # perWeight降序ranking
                activated_experts_info.sort(key=lambda x: x['weight'], reverse=True)
                activated_experts_info = activated_experts_info[:3]
                
                # Updating expert_weightsDictionary，只保留前3experts
                top_3_experts = {info['expert']['specialty_chinese_name']: info['weight'] 
                               for info in activated_experts_info}
                expert_weights = top_3_experts
                
                self.logger.info(f"expert数量超过3，已Limitas前3Weight最高expert")
            
            # Weight归一化
            if expert_weights:
                total_weight = sum(expert_weights.values())
                if total_weight > 0:
                    # 消融实验：均etc.Weight
                    if getattr(self.args, 'ablation_equal_expert_weights', False):
                        # asallexpert分配均etc.Weight
                        equal_weight = 1.0 / len(expert_weights)
                        for expert_name in expert_weights:
                            expert_weights[expert_name] = equal_weight
                        self.logger.info("消融实验：使用均etc.expertWeight")
                    else:
                        # 归一化Weight
                        for expert_name in expert_weights:
                            expert_weights[expert_name] = expert_weights[expert_name] / total_weight
                    
                    # Updating activated_experts_infoWeight
                    for expert_info in activated_experts_info:
                        expert_name = expert_info['expert']['specialty_chinese_name']
                        expert_info['weight'] = expert_weights[expert_name]
                    
                    self.logger.info(f"expertWeight已归一化，总Weight: {sum(expert_weights.values()):.3f}")
            
            self.logger.info(f"fromData集Getting to {len(activated_experts_info)} experts: {list(expert_weights.keys())}")
            self.logger.info(f"expertWeightDistribution: {expert_weights}")
            
            return {
                'activated_experts': activated_experts_info,
                'expert_weights': expert_weights,
                'total_experts': len(activated_experts_info),
                'activation_method': 'ground_truth_from_dataset'
            }
            
        except Exception as e:
            self.logger.error(f"Getting standard答案expertFailed : {e}")
            # 如果Failed ，回退toOriginalMOEexpert选择
            return self.moe_expert_system.activate_experts(case_data)
    
    def preload_models(self):
        """Preloading model（移除Warmuppipeline）"""
        try:
            preload_start = time.time()
            print("[step0] 模型预Loading Starting ...")
            
            # Preloading modeltoGPU（Component已构造FunctionInitializing ）
            if hasattr(self.model_manager, 'preload_models'):
                self.model_manager.preload_models()
            
            self.timing_stats['model_loading_time'] = time.time() - preload_start
            print(f"[step0] 模型preloading complete，耗时: {self.timing_stats['model_loading_time']:.2f}seconds")
            
        except Exception as e:
            print(f"[step0] 模型预Loading Failed : {e}")
            import traceback; traceback.print_exc()
            # Continuing Running ，但可能性能较差
    
    # WarmupMethodremoved，避免DuplicateExecuting 
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loading Data集"""
        try:
            dataset_start = time.time()
            print(f"[Data集] Loading : {self.args.dataset_path}")
            
            with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
                cases = [json.loads(line) for line in f]
            
            # Limitcase数量
            if self.args.max_cases and self.args.max_cases > 0:
                cases = cases[:self.args.max_cases]
            
            self.timing_stats['dataset_loading_time'] = time.time() - dataset_start
            print(f"[Data集] Successfully Loading  {len(cases)} case，耗时: {self.timing_stats['dataset_loading_time']:.2f}seconds")
            
            return cases
            
        except Exception as e:
            print(f"[Data集] Loading Failed : {e}")
            import traceback; traceback.print_exc()
            return []
    
    def run_workflow(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Running completeMMAS三agent工作pipeline
        
        implements7Stepdiagnosis pipeline：
        1. Patient AgentInitial question (格式化asstandardmedical选择题)
        2. Evaluator AgentInformation质量evaluation
        3. Evaluator Agent澄清questionGenerating  (如需要)
        4. Patient Agent澄清answer (如需要)
        5. MOEexpertActivationandWeight分配
        6. Doctor Agentdiagnosisinference
        7. Evaluator Agent最终decision
        
        Args:
            case_data: caseData
            
        Returns:
            Dict[str, Any]: 工作pipelineresult
        """
        try:
            workflow_start = time.time()
            case_id = case_data.get('id', 'unknown')
            total_tokens = 0
            clarification_rounds = 0
            
            print(f"\n{'='*80}")
            print(f"[case] Starting Processing case {case_id}")
            print(f"{'='*80}")
            
            # Step 1: Patient agent initial question
            print(f"\n[step1] Patient AgentInitial question")
            print(f"{'─'*50}")
            print(f"InputcaseData:")
            print(f"  - Case ID: {case_id}")
            print(f"  - question: {case_data.get('question', 'N/A')[:100]}...")
            print(f"  - patientInformation: {case_data.get('patient', {})}")
            
            step1_start = time.time()
            step1_result = self.patient_agent.step1_initial_question(case_data)
            step1_time = time.time() - step1_start
            
            print(f"Outputresult:")
            if 'error' in step1_result:
                print(f"  [incorrect] incorrect: {step1_result['error']}")
                return self._create_error_result(step1_result['error'], 1, workflow_start)
            else:
                total_tokens += step1_result.get('total_tokens', 0)
                print(f"  [complete] Initial questioncomplete")
                print(f"  [Content] Initial questionContent: {step1_result.get('formatted_question', 'N/A')}")
                print(f"  [Time] Processing Time: {step1_time:.2f}seconds")
            
            # Step 2: Evaluator agent information quality assessment
            print(f"\n[step2] Evaluator AgentInformation质量evaluation")
            print(f"{'─'*50}")
            
            start_time = time.time()
            self.logger.info(f"Processing case {case_data.get('id', 'unknown')}...")

            # 复用step1result，不再Duplicate调用
            patient_output = step1_result

            # Step 2: Evaluator assesses the quality of the information
            step2_result = self.evaluator_agent.step_2_assess_quality(case_data)
            step2_time = time.time() - start_time

            if 'error' in step2_result:
                print(f"  [incorrect] incorrect: {step2_result['error']}")
                return self._create_error_result(step2_result['error'], 2, workflow_start)

            total_tokens += step2_result.get('total_tokens', 0)
            # evaluationresultsave tocase_data，供后续step使用
            case_data['assessment_result'] = step2_result

            decision = step2_result.get('decision', 'decisionUnknown ')
            needs_clarification = step2_result.get('needs_clarification', False)

            # 终端Outputstep2evaluationresult摘要（each评分嵌套 step2_scores 里）
            scores_dict = step2_result.get('step2_scores', {}) or {}
            decision_display = scores_dict.get('decision', decision)
            print(f"Outputresult:")
            print(f"  [评分] basic={scores_dict.get('basic_score', 'N/A')}, "
                  f"symptom={scores_dict.get('symptom_score', 'N/A')}, "
                  f"exam={scores_dict.get('exam_score', 'N/A')}, "
                  f"timeline={scores_dict.get('timeline_score', 'N/A')}, "
                  f"logic={scores_dict.get('logic_score', 'N/A')}")
            print(f"  [总分] {scores_dict.get('total_score', 'N/A')} | "
                  f"质量etc.级: {scores_dict.get('quality_level', 'N/A')}")
            print(f"  [decision] {decision_display} | 需要澄清: {needs_clarification}")
            print(f"  [Time] Processing Time: {step2_time:.2f}seconds")

            # 消融实验：Disable 澄清循环 - 最Starting 就Checking 
            if getattr(self.args, 'ablation_no_clarification', False):
                needs_clarification = False
                print(f"\n[消融实验] Disable 澄清循环 - Skipping step3and4")
                self.logger.info("消融实验：Disable 澄清循环")
                # 直接跳转tostep5，不Executing 任何澄清相关Code
            elif needs_clarification:
                # Redundant pre-loop clarification block removed; clarification handled inside the loop below
                pass
            
            # 新增：Record澄清for话
            clarification_loop_count = 0
            clarification_history = []
            step3_time = 0  # Initializing 澄清step耗时
            step4_time = 0
            
            # 初始evaluationresult
            current_assessment_result = step2_result
            needs_clarification = current_assessment_result.get('needs_clarification', False)
            
            # 消融实验：Disable 澄清循环
            if getattr(self.args, 'ablation_no_clarification', False):
                needs_clarification = False
                self.logger.info("消融实验：Disable 澄清循环")

            while needs_clarification and clarification_loop_count < self.args.max_clarification_loops:
                clarification_loop_count += 1
                clarification_rounds = clarification_loop_count
                print(f"\n[Loop] Starting clarification round {clarification_loop_count}...")
                print(f"{'─'*50}")

                # step3：Generating 澄清question
                step3_start = time.time()
                step3_result = self.evaluator_agent.step3_clarification_generation(current_assessment_result, case_data)
                if 'error' in step3_result:
                    print(f"  [Error] Step 3 error: {step3_result['error']}")
                    break 

                step3_time += time.time() - step3_start
                total_tokens += step3_result.get('total_tokens', 0)
                # 仅Recordstep3Output（不OutputorRecordInputPrompt）
                step3_prompt = step3_result.get('input_prompt', '')

                step3_raw = step3_result.get('raw_response', '')
                print("[step3] Evaluator Agent澄清questionGenerating  - OutputLLMresponse:")
                print(step3_raw)
                self.logger.info("[step3-Output] Raw Response:\n" + (step3_raw if isinstance(step3_raw, str) else str(step3_raw)))

                clarification_questions = step3_result.get('clarification_questions', [])
                if not clarification_questions:
                    print("  [Warning] Step 3 did not generate valid questions. Stopping clarification.")
                    break

                # 显示过滤后澄清questionList
                print("[step3] Evaluator Agent澄清questionGenerating  - 过滤后questionList:")
                for idx, q in enumerate(clarification_questions, 1):
                    print(f"  Q{idx}: {q}")
                try:
                    self.logger.info("[step3-Output] Clarification Questions: " + json.dumps(clarification_questions, ensure_ascii=False))
                except Exception:
                    pass

                # step4：patientanswerquestion
                step4_start = time.time()
                step4_result = self.patient_agent.step4_clarification_response(clarification_questions, case_data)
                if 'error' in step4_result:
                    print(f"  [Error] Step 4 error: {step4_result['error']}")
                    break

                step4_time += time.time() - step4_start
                total_tokens += step4_result.get('total_tokens', 0)
                # 仅Recordstep4Output（不OutputorRecordInputPrompt）
                step4_prompt = step4_result.get('input_prompt', '')

                step4_raw = step4_result.get('raw_response', '')
                print("[step4] Patient Agent澄清answer - OutputLLMresponse:")
                print(step4_raw)
                self.logger.info("[step4-Output] Raw Response:\n" + (step4_raw if isinstance(step4_raw, str) else str(step4_raw)))

                # 显示step4result
                info_completeness = step4_result.get('information_completeness', 0.0)
                print(f"  [Data] Step 4 result: information completeness={info_completeness:.2f}")
                
                # Record澄清History
                if 'clarification_context' in step4_result:
                    clarification_history.extend(step4_result['clarification_context'])
                    # 打印澄清Context
                    print("[step4] 澄清Context:")
                    for item in step4_result.get('clarification_context', []):
                        print(f"  Q: {item.get('question','')}\n  A: {item.get('answer','')}")
                    try:
                        self.logger.info("[step4-Output] Clarification Context: " + json.dumps(step4_result.get('clarification_context', []), ensure_ascii=False))
                    except Exception:
                        pass
                
                # 关键修复：Creating 一新、干净Context用于下一timesevaluation
                # 只Required、新Information添加to下一timesevaluationContext
                # Checking 澄清answeris否allas"Not provided"，如果is则Skipping Context拼接
                clarification_items = step4_result.get('clarification_context', [])
                all_not_provided = all(
                    item.get('answer', '').strip().lower() in ('not provided', 'Information缺失，不清楚', '')
                    for item in clarification_items
                ) if clarification_items else True

                new_context_for_assessment = case_data.get('context', [])
                if isinstance(new_context_for_assessment, list):
                    new_context_for_assessment = list(new_context_for_assessment)  # 浅拷贝，避免修改Originalcase_data
                    if not all_not_provided:
                        # 仅当澄清answer包含实质Information时，才拼接toContext
                        clarification_summary = "\n".join([f"Q: {item['question']} A: {item['answer']}" for item in clarification_items])
                        new_context_for_assessment.append(f"Additional information {clarification_loop_count}:\n{clarification_summary}")
                
                # Creating 一临时、轻量化 case_data 用于重新evaluation
                temp_case_data_for_reassessment = case_data.copy()
                temp_case_data_for_reassessment['context'] = new_context_for_assessment

                # 使用Updating 后Context重新进行step2evaluation
                print(f"\n[Step 2] Re-assess quality after clarification round {clarification_loop_count}...")
                current_assessment_result = self.evaluator_agent.step_2_assess_quality(temp_case_data_for_reassessment)
                total_tokens += current_assessment_result.get('total_tokens', 0)

                # 显示重新evaluationresult
                reassess_scores = current_assessment_result.get('step2_scores', {}) or {}
                print(f"  [重新评分] 总分: {reassess_scores.get('total_score', 'N/A')} | "
                      f"decision: {reassess_scores.get('decision', 'N/A')} | "
                      f"需要澄清: {current_assessment_result.get('needs_clarification', False)}")

                # Updating 循环condition：只has重新evaluation后score达标才Exiting 
                needs_clarification = current_assessment_result.get('needs_clarification', False)
                if not needs_clarification:
                    print(f"  [Done] 质量evaluation达标，End澄清循环")
                else:
                    print(f"  [Continue] 质量仍未达标，Continuing 澄清...")

            # 最终澄清History（如果存）Updating to主 case_data 
            if clarification_history:
                case_data['clarification_history'] = clarification_history

            # Updating step2_resultandcase_dataas澄清后最新evaluationresult
            # 确保后续stepand最终resultDictionary使用is澄清后evaluation，而非初始evaluation
            step2_result = current_assessment_result
            case_data['assessment_result'] = current_assessment_result

            # step5：MOEexpertActivationandWeight分配
            print(f"\n[step5] MOEexpertActivationandWeight分配")
            print(f"{'─'*50}")
            
            step5_start = time.time()
            
            # 根据Command行Parameter决定使用哪expertActivationMethod
            if self.args.use_ground_truth_experts:
                print("  [mode] 使用Data集standard答案expert标签")
                moe_result = self._get_ground_truth_experts(case_data)
            else:
                print("  [mode] 使用MOEexpert选择算法")
                # 使用MOEexpert system进行expertActivation（显示详细三维score）
                moe_result = self.moe_expert_system.activate_experts(case_data)
                
                # 消融实验：均etc.Weight - MOEsystemof后再times确保Weight均etc.
                if getattr(self.args, 'ablation_equal_expert_weights', False):
                    activated_experts_info = moe_result.get('activated_experts', [])
                    expert_weights = moe_result.get('expert_weights', {})
                    
                    if expert_weights:
                        # 重新分配均etc.Weight
                        equal_weight = 1.0 / len(expert_weights)
                        for expert_name in expert_weights:
                            expert_weights[expert_name] = equal_weight
                        
                        # Updating activated_experts_infoWeight
                        for expert_info in activated_experts_info:
                            expert_data = expert_info.get('expert', {})
                            expert_name = expert_data.get('specialty_chinese_name') or expert_data.get('expert_name') or 'Unknown'
                            if expert_name in expert_weights:
                                expert_info['weight'] = expert_weights[expert_name]
                        
                        # Updating moe_result
                        moe_result['expert_weights'] = expert_weights
                        moe_result['activated_experts'] = activated_experts_info
                        
                        self.logger.info(f"消融实验：Force 使用均etc.Weight，eachexpertsWeight: {equal_weight:.3f}")
            
            activated_experts_info = moe_result.get('activated_experts', [])
            expert_weights = moe_result.get('expert_weights', {})
            
            step5_time = time.time() - step5_start
            
            # 修复expertname提取逻辑，correct访问嵌套expertDictionary
            activated_experts_names = []
            for info in activated_experts_info:
                expert_data = info.get('expert', {})
                expert_name = expert_data.get('specialty_chinese_name') or expert_data.get('expert_name') or info.get('specialty_chinese_name', info.get('expert_name', 'Unknown'))
                activated_experts_names.append(expert_name)

            print(f"Outputresult:")
            if not activated_experts_info:
                error_msg = "expertActivationFailed ：Not found合适expert"
                print(f"  [incorrect] incorrect: {error_msg}")
                return self._create_error_result(error_msg, 5, workflow_start)
            else:
                print(f"  [complete] expertActivationcomplete")
                print(f"  [expert] Activationexpert数: {len(activated_experts_info)}")
                print(f"  [List] expertList: {', '.join(activated_experts_names)}")
                print(f"  [Data] expert得分详情:")
                
                # 显示eachexperts详细得分
                for info in activated_experts_info:
                    expert_data = info.get('expert', {})
                    expert_name = expert_data.get('specialty_chinese_name') or expert_data.get('expert_name') or 'Unknown'
                    total_score = info.get('score', 0.0)
                    weight = expert_weights.get(expert_name, 0.0)
                    
                    # Getting 详细得分Information
                    score_details = expert_data.get('score_details', {})
                    semantic_score = score_details.get('semantic_score', 0.0)
                    keyword_score = score_details.get('keyword_score', 0.0)
                    mechanism_score = score_details.get('mechanism_score', 0.0)
                    
                    print(f"    - {expert_name}:")
                    # 显示expert得分详情，使用实际Parameter值
                    alpha_value = getattr(self.args, 'alpha', 0.4)
                    beta_value = getattr(self.args, 'beta', 0.3)
                    print(f"      ├─ 语义similarity: {semantic_score:.4f} (Weight α={alpha_value})")
                    print(f"      ├─ Keyword matching: {keyword_score:.4f} (Weight β={beta_value})")
                    # 显示机制matching度，使用实际gammaParameter值
                    gamma_value = getattr(self.args, 'gamma', 0.3)
                    print(f"      ├─ 机制matching度: {mechanism_score:.4f} (Weight γ={gamma_value})")
                    print(f"      ├─ 综合得分: {total_score:.4f}")
                    print(f"      └─ ActivationWeight: {weight:.3f}")
                
                print(f"  [Weight] Weight分配:")
                for expert_name in activated_experts_names:
                    weight = expert_weights.get(expert_name, 0.0)
                    print(f"    - {expert_name}: {weight:.3f}")
                print(f"  [Time] Processing Time: {step5_time:.2f}seconds")
            
            # Weightandexpert详细Information添加to case_data 
            case_data["expert_weights"] = expert_weights
            case_data["activated_experts_info"] = activated_experts_info

            # step6：Doctor Agentdiagnosisinference
            print(f"\n[step6] Doctor Agentdiagnosisinference")
            print(f"{'─'*50}")
            
            step6_start = time.time()
            # 准备澄清Context
            clarification_context = case_data.get('clarification_history', [])
            
            # activated_experts已经isexpertnameList，expert_weightsisWeightDictionary
            step6_result = self.doctor_agent.step_6_diagnose_and_reasoning(
                case_data, activated_experts_info, clarification_context=clarification_context
            )
            step6_time = time.time() - step6_start
            
            print(f"Outputresult:")
            # Processing returnresult，supportListandDictionary两格式
            if isinstance(step6_result, dict) and 'expert_opinions' in step6_result:
                expert_opinions = step6_result['expert_opinions']
                total_tokens += step6_result.get('total_tokens', 0)
            elif isinstance(step6_result, list):
                expert_opinions = step6_result
            else:
                print(f"  [incorrect] incorrect: step6return意外Data格式")
                return self._create_error_result("step6return意外Data格式", 6, workflow_start)

            print(f"  [complete] expertdiagnosisinferencecomplete")
            print(f"  [Data] 收集expert意见数: {len(expert_opinions)}")
            
            # 显示eachexperts具体答案选择
            if expert_opinions:
                print(f"  [diagnosis] eachexpertdiagnosis result:")
                for opinion in expert_opinions:
                    expert_name = opinion.get('expert_name', 'Unknown expert')
                    expert_specialty = opinion.get('expert_specialty', 'Unknown specialty')
                    expert_weight = opinion.get('weight', 0.0) # 使用 'weight' 键
                    scores = opinion.get('scores', {})
                    
                    # 格式化Output，保持for齐
                    print(f"    ┌─ {expert_name} ({expert_specialty}) [Weight: {expert_weight:.3f}]")
                    
                    if scores:
                        # 显示allOptionscore
                        print(f"    │  eachOptionconfidence:")
                        for option, score in sorted(scores.items()):
                            print(f"    │    {option}: {score:.3f}")
                        
                        # 找toscore最高Option
                        best_choice = max(scores.items(), key=lambda x: x[1])
                        choice, score = best_choice
                        print(f"    └─ recommendation答案: {choice} (最高confidence: {score:.3f})")
                    else:
                        print(f"    └─ novalid答案")
                    print()  # 空行分隔
            
            print(f"  [Time] Processing Time: {step6_time:.2f}seconds")
            
            # Step 7: Evaluator agent final decision
            print(f"\n[step7] Evaluator Agent最终decision")
            print(f"{'─'*50}")
            
            step7_start = time.time()
            step7_result = self.evaluator_agent.step7_final_decision(
                expert_opinions, expert_weights, case_data
            )
            step7_time = time.time() - step7_start
            
            print(f"Outputresult:")
            if 'error' in step7_result:
                print(f"  [incorrect] incorrect: {step7_result['error']}")
                return self._create_error_result(step7_result['error'], 7, workflow_start)
            else:
                total_tokens += step7_result.get('total_tokens', 0)
                final_answer = step7_result.get('final_answer', 'N/A')
                confidence = step7_result.get('confidence_score', step7_result.get('confidence', 0))
                reasoning = step7_result.get('reasoning_path', step7_result.get('reasoning', 'N/A'))
                
                # 判断答案is否correct
                correct_answer = case_data.get('answer_idx', '').upper()
                predicted_answer = str(final_answer).upper()
                # N/A表示allexpert均未returnvalid答案，必须视asincorrect
                is_correct = (predicted_answer == correct_answer) and predicted_answer != 'N/A'
                
                # 如果答案incorrect，RecordincorrectIDto专门incorrectlogfile
                if not is_correct:
                    self._record_error_case_id(case_id)
                
                print(f"  [complete] 最终decisioncomplete")
                print(f"  [答案] 最终答案: {final_answer}")
                print(f"  [correct性] 答案correct性: {is_correct}")
                print(f"  [for比] prediction答案: {predicted_answer}, correct答案: {correct_answer}")
                print(f"  [confidence] confidence: {confidence:.3f}")
                print(f"  [inference] inference过程:")
                for line in reasoning.split('\n'):
                    print(f"    {line}")
                print(f"  [Time] Processing Time: {step7_time:.2f}seconds")
            
            # Computing 总Processing Time
            total_time = time.time() - workflow_start
            print(f"\n[complete] caseProcessing complete")
            print(f"{'─'*50}")
            print(f"[Time] 总Processing Time: {total_time:.2f}seconds")
            print(f"{'='*80}")
            
            # 构建completeresult
            result = {
                'case_id': case_id,
                'status': 'success',
                'final_answer': step7_result.get('final_answer'),
                'confidence': step7_result.get('confidence_score', step7_result.get('confidence', 0)),
                'reasoning': step7_result.get('reasoning_path', step7_result.get('reasoning', '')),
                'processing_time': total_time,
                'step_times': {
                    'step1_format_case': step1_time,
                    'step2_quality_assessment': step2_time,
                    'step3_clarification': step3_time,
                    'step4_clarification_response': step4_time,
                    'step5_expert_activation': step5_time,
                    'step6_expert_diagnosis': step6_time,
                    'step7_final_decision': step7_time
                },
                'activated_experts': activated_experts_names,
                'expert_weights': expert_weights,
                'expert_opinions': expert_opinions,
                'needs_clarification': step2_result.get('needs_clarification', False),
                'quality_score': step2_result.get('quality_score', 0),
                'total_tokens': total_tokens,
                'clarification_rounds': clarification_rounds,
                'num_activated_experts': len(activated_experts_names)
            }
            
            return result
            
        except Exception as e:
            error_msg = f"工作pipelineExecuting Failed : {e}"
            self.logger.error(error_msg)
            if self.args.debug:
                self.logger.error(traceback.format_exc())
            
            # Recordincorrecttoincorrectstatistics
            record_error("workflow_execution_error", str(e), case_id, 0)
            
            return self._create_error_result(str(e), 0, workflow_start)
    
    def _create_error_result(self, error_msg: str, step: int, workflow_start: float) -> Dict[str, Any]:
        """Creating incorrectresult"""
        return {
            'error': error_msg,
            'failed_step': step,
            'status': 'failed',
            'workflow_time': time.time() - workflow_start if workflow_start else 0
        }
    
    def process_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """batch processingcase"""
        results = []
        total_cases = len(cases)
        
        # Loading Checking 点，确定Starting Processing Position
        last_processed_index, _ = self._load_checkpoint()
        start_index = last_processed_index + 1
        
        print(f"[Processing ] Starting batch processing {total_cases} case")
        if start_index > 0:
            print(f"[Processing ] from第 {start_index + 1} caseStarting Continuing Processing ")
        
        for i, case in enumerate(cases):
            # 如果Enable Checkpoint resume且Currentcase已经Processing 过，则Skipping 
            if i <= last_processed_index:
                continue
                
            try:
                case_start = time.time()
                
                # Processing Singlecase
                result = self.run_workflow(case)
                
                # RecordProcessing Time
                processing_time = time.time() - case_start
                result['processing_time'] = processing_time
                self.timing_stats['case_processing_times'].append(processing_time)
                
                # Updating statistics
                self._update_statistics(result, case)
                
                # 打印进度
                self._print_progress(i + 1, total_cases, result, case)
                
                results.append(result)
                
                # 根据Setting 间隔Saving Checking 点（仅Enable Checkpoint resume时valid）
                if self.use_resume and (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(i, total_cases)
                
            except Exception as e:
                case_id = case.get('id', f'case_{i}')
                error_msg = f"Processing case {case_id} Failed : {e}"
                self.logger.error(error_msg)
                if self.args.debug:
                    self.logger.error(traceback.format_exc())
                
                # Recordincorrecttoincorrectstatistics
                record_error("case_processing_error", str(e), case_id)
                
                error_result = {
                    'case_id': case_id,
                    'error': str(e),
                    'status': 'failed',
                    'processing_time': 0
                }
                results.append(error_result)
                
                # 即使出错也要根据间隔Saving Checking 点（仅Enable Checkpoint resume时valid）
                if self.use_resume and (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(i, total_cases)
        
        # Computing Populationstatistics
        self._calculate_final_statistics()
        
        # allcaseProcessing complete后Saving 最终Checking 点并清理
        if self.use_resume:
            self._save_checkpoint(total_cases - 1, total_cases)  # Saving 最终status
        self._cleanup_checkpoint()
        
        return results
    
    def _update_statistics(self, result: Dict[str, Any], case: Dict[str, Any]):
        """Updating statisticsInformation"""
        self.total_cases += 1
        
        # Checking 答案correct性
        if result.get('status') == 'success':
            predicted_answer = result.get('final_answer', '').upper()
            # 使用answer_idx字Segment进行比较，这isOption索引（A、B、C、D）
            correct_answer = case.get('answer_idx', '').upper()
            
            # N/A表示allexpert均未returnvalid答案，不计入correctstatistics
            if predicted_answer == correct_answer and predicted_answer != 'N/A':
                self.correct_cases += 1
                if self.args.verbose:
                    self.logger.info(f"case {case.get('id', 'unknown')} 答案correct: {predicted_answer}")
            else:
                if self.args.verbose:
                    self.logger.info(f"case {case.get('id', 'unknown')} 答案incorrect: prediction={predicted_answer}, correct={correct_answer}")
        
        # Updating 新增metricstatistics
        self.performance_stats['total_activated_experts'] = self.performance_stats.get('total_activated_experts', 0) + len(result.get('activated_experts', []))
        self.performance_stats['total_clarification_rounds'] = self.performance_stats.get('total_clarification_rounds', 0) + result.get('clarification_rounds', 0)
        self.performance_stats['total_tokens'] = self.performance_stats.get('total_tokens', 0) + result.get('total_tokens', 0)
    
    def _print_progress(self, current: int, total: int, result: Dict[str, Any], case: Dict[str, Any]):
        """打印Processing 进度"""
        progress = (current / total) * 100
        case_id = case.get('id', f'case_{current}')
        
        if result.get('status') == 'success':
            answer = result.get('final_answer', 'N/A')
            confidence = result.get('confidence', 0)
            time_taken = result.get('processing_time', 0)
            
            print(f"[{current}/{total}] ({progress:.1f}%) case {case_id}: {answer} (confidence: {confidence:.3f}, 耗时: {time_taken:.2f}s)")
        else:
            error = result.get('error', 'Unknown error')
            print(f"[{current}/{total}] ({progress:.1f}%) case {case_id}: Failed  - {error}")
    
    def _calculate_final_statistics(self):
        """Computing 最终statisticsInformation"""
        self.timing_stats['total_processing_time'] = sum(self.timing_stats['case_processing_times'])
        
        if self.total_cases > 0:
            # Computing Timestatistics
            case_times = self.timing_stats['case_processing_times']
            self.timing_stats.update({
                'avg_processing_time': sum(case_times) / len(case_times) if case_times else 0,
                'min_processing_time': min(case_times) if case_times else 0,
                'max_processing_time': max(case_times) if case_times else 0
            })
            
            self.performance_stats.update({
                'total_cases': self.total_cases,
                'correct_cases': self.correct_cases,
                'accuracy': self.correct_cases / self.total_cases,
                'average_case_time': self.timing_stats['total_processing_time'] / self.total_cases,
                'total_processing_time': self.timing_stats['total_processing_time'],
                # 新增：Computing Average
                'avg_activated_experts': self.performance_stats.get('total_activated_experts', 0) / self.total_cases,
                'avg_clarification_rounds': self.performance_stats.get('total_clarification_rounds', 0) / self.total_cases,
                'avg_tokens_per_case': self.performance_stats.get('total_tokens', 0) / self.total_cases
            })
    
    def _load_checkpoint(self) -> Tuple[int, str]:
        """
        Loading Checking 点file，return上timesProcessing case索引andlogfilepath
        
        Returns:
            tuple: (上timesSuccessfully Processing case索引，logfilepath)，如果没hasChecking 点则return(-1, None)
        """
        if not self.use_resume:
            return -1, None
            
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    last_processed_index = checkpoint_data.get('last_processed_index', -1)
                    total_cases = checkpoint_data.get('total_cases', 0)
                    correct_cases = checkpoint_data.get('correct_cases', 0)
                    log_file = checkpoint_data.get('log_file', None)
                    
                    # Restore statisticsInformation
                    self.total_cases = last_processed_index + 1  # processedcase数
                    self.correct_cases = correct_cases
                    
                    print(f"📋 发现Checking 点file: {self.checkpoint_file}")
                    print(f"📋 上timesProcessing to第 {last_processed_index + 1} case（共 {total_cases} ）")
                    print(f"📋 Restore statisticsInformation: processed {self.total_cases} case，correct {self.correct_cases} ")
                    print(f"📋 from第 {last_processed_index + 2} caseStarting Continuing Processing ")
                    if log_file:
                        print(f"📋 Continuing 使用logfile: {log_file}")
                    
                    return last_processed_index, log_file
            else:
                print(f"📋 Not foundChecking 点file，from头Starting Processing ")
                return -1, None
                
        except Exception as e:
            print(f"⚠️ Reading Checking 点fileFailed : {e}")
            print(f"📋 from头Starting Processing ")
            return -1, None
    
    def _save_checkpoint(self, processed_index: int, total_cases: int):
        """
        Saving Checking 点file
        
        Args:
            processed_index (int): processedcase索引
            total_cases (int): 总case数
        """
        if not self.use_resume:
            return
            
        try:
            checkpoint_data = {
                'last_processed_index': processed_index,
                'total_cases': total_cases,
                'correct_cases': self.correct_cases,  # 添加correctcase数statistics
                'timestamp': datetime.now().isoformat(),
                'dataset_path': self.args.dataset_path,
                'log_file': getattr(self.log_manager, 'log_file_path', None) if hasattr(self, 'log_manager') else None
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Saving Checking 点fileFailed : {e}")
    
    def _cleanup_checkpoint(self):
        """
        清理Checking 点file
        """
        if not self.use_resume:
            return
            
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print(f"🗑️ 已清理Checking 点file: {self.checkpoint_file}")
        except Exception as e:
            print(f"⚠️ 清理Checking 点fileFailed : {e}")
    
    def print_summary(self):
        """打印Running summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏥 MMAS三agentsystemRunning summary")
        print("="*60)
        
        # performance statistics
        print(f"[Data] performance statistics:")
        print(f"  - 总case数: {self.total_cases}")
        print(f"  - correctcase数: {self.correct_cases}")
        print(f"  - 准确率: {self.correct_cases / self.total_cases * 100:.2f}%" if self.total_cases > 0 else "  - 准确率: 0.00%")
        print(f"  - 平均eachcaseProcessing Time (Latency): {self.performance_stats.get('average_case_time', 0):.2f}seconds")
        print(f"  - 平均Activationexpert数: {self.performance_stats.get('avg_activated_experts', 0):.2f}")
        print(f"  - 平均澄清轮数: {self.performance_stats.get('avg_clarification_rounds', 0):.2f}")
        print(f"  - Avg Token usage: {self.performance_stats.get('avg_tokens_per_case', 0):.0f}")
        
        print(f"\n[Time] Timestatistics:")
        print(f"  - 平均Processing Time: {self.timing_stats['avg_processing_time']:.2f}seconds")
        print(f"  - 最短Processing Time: {self.timing_stats['min_processing_time']:.2f}seconds")
        print(f"  - 最长Processing Time: {self.timing_stats['max_processing_time']:.2f}seconds")
        print(f"  - 总Processing Time: {self.timing_stats['total_processing_time']:.2f}seconds")
        print(f"  - Initializing Time: {self.timing_stats['initialization_time']:.2f}seconds")
        print(f"  - 模型Loading Time: {self.timing_stats['model_loading_time']:.2f}seconds")
        print(f"  - Data集Loading Time: {self.timing_stats['dataset_loading_time']:.2f}seconds")
        print(f"  - 总Running Time: {total_time:.2f}seconds")
        
        # systemstatus
        print(f"\n🔧 systemstatus:")
        print(f"  Patient Agent: [活跃] 活跃")
        print(f"  Evaluator Agent: [活跃] 活跃")
        print(f"  Doctor Agent: [活跃] 活跃")
        print(f"  Expert Matching System: [活跃] 活跃")
        print(f"  Model Manager: [活跃] 活跃")
        print(f"  Log Manager: [活跃] 活跃")
        
        print("="*60)
        
        # OutputError case log filePosition
        if hasattr(self, 'error_case_log_file') and self.error_case_log_file:
            print(f"\n📋 Error case log file: {self.error_case_log_file}")
        
        # Stopping logRecord
        stop_system_logging()

def parse_arguments():
    """Parsing Command行Parameter"""
    parser = argparse.ArgumentParser(description='MMAS三agent医疗diagnosissystem')
    
    # Data集Parameter
    parser.add_argument('--dataset_path', type=str, 
                       default='./data/all_dev_convo.jsonl',
                       help='Data集filepath')
    
    # Processing Parameter
    parser.add_argument('--max_cases', type=int, default=None,
                       help='最大Processing case数量')
    
    # OutputParameter
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Outputdirectory')
    
    # debugParameter
    parser.add_argument('--debug', action='store_true',
                       help='Enable debugmode')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable 详细Output')
    
    parser.add_argument('--force_clarification', action='store_true',
                       help='Force Executing step二澄清pipeline')
    
    parser.add_argument('--show_step2_details', action='store_true',
                       help='显示step二详细InputPromptandLLMresponse')
    
    # 澄清ThresholdParameter
    parser.add_argument('--clarification_threshold', type=int, default=20,
                       help='澄清Threshold：低于thisscorecase进入澄清pipeline (Default: 20)')
    
    # 澄清循环Parameter
    parser.add_argument('--max_clarification_loops', type=int, default=1,
                       help='最大澄清循环times数 (Default: 1)')
    
    # expert选择Parameter
    parser.add_argument('--use_ground_truth_experts', action='store_true',
                       help='使用Data集standard答案expert标签，绕过MOEexpert选择算法')
    
    # MOEexpert system高级Parameter
    parser.add_argument('--use_adaptive_threshold', action='store_true',
                       help='Enable 自适应剪枝strategy，代替固定Threshold')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax温度Parameterτ，控制WeightDistribution集度 (Default: 1.0，仅用于MOEexpertWeightSoftmax，不影响LLMGenerating temperature)')

    
    # 模型Parameter
    parser.add_argument('--model_path', type=str, 
                        default='./models/Qwen3-4B-Instruct-2507',
                        help='LLM模型path')
    parser.add_argument('--quantization', type=str, default='none',
                       choices=['none', '8bit', '4bit'],
                       help='quantization type：none(不量化), 8bit(8量化), 4bit(4量化)')
    parser.add_argument('--expert_knowledge_path', type=str, 
                       default='./data/dev_dataset_expert_knowledge_graph_enhanced.json',
                       help='Expert knowledge graph file path')
    
    # API模型Parameter
    parser.add_argument('--use_api_model', action='store_true',
                       help='Using API model而不isthis地模型')
    parser.add_argument('--api_model_name', type=str, default='deepseek-chat',
                       choices=['deepseek-chat', 'glm-5.1'],
                       help='API model name')
    parser.add_argument('--api_key', type=str,
                       help='API密钥（如果不provide，fromConfiguring fileReading ）')
    
    # Checkpoint resumeParameter
    parser.add_argument('--resume', action='store_true',
                       help='Enable Checkpoint resumefunctionality，from上timesInterruptedPositionContinuing Processing ')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                       help='Checking 点Saving 间隔：eachProcessing 多少caseSaving 一timesChecking 点（Default：1，仅Enable Checkpoint resume时valid）')
    
    # 消融实验Parameter
    parser.add_argument('--ablation_generic_cot', action='store_true',
                       help='消融实验：使用Generic CoT template替代专业化意graphtemplate')
    parser.add_argument('--ablation_top1', action='store_true',
                       help='消融实验：只ActivationTop-1expert（k=1）')
    parser.add_argument('--ablation_equal_expert_weights', action='store_true',
                       help='消融实验：allActivationexpert使用均etc.Weight')
    parser.add_argument('--ablation_no_mechanism', action='store_true',
                       help='消融实验：expertActivation时不使用机制评分（gamma=0）')
    parser.add_argument('--ablation_no_clarification', action='store_true',
                       help='消融实验：Disable 澄清循环')
    
    # MOE算法WeightParameter
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='MOE算法语义similarityWeightParameter（Default：0.4）')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='MOE算法Keyword matchingWeightParameter（Default：0.3）')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='MOE算法机制matching度WeightParameter（Default：0.3）')
    
    # expertActivation数量Parameter
    parser.add_argument('--top_k', type=int, default=3,
                       help='Activationexpert数量（Default：3）')
    
    parser.add_argument('--use_semantic_fallback', action='store_true',
                          help='使用语义回退进行意graphclassification')
    
    return parser.parse_args()

def main():
    """主Function"""
    try:
        # Parsing Parameter
        args = parse_arguments()
        
        print("MMAS三agent医疗diagnosissystemStarting ")
        print(f"Data集: {args.dataset_path}")
        print(f"最大case数: {args.max_cases or 'noLimit'}")
        if args.use_api_model:
            print(f"Using API model: {args.api_model_name}")
        else:
            print(f"使用this地模型: {args.model_path}")
            print(f"quantization config: {args.quantization}")
        print("-" * 60)
        
        # Initializing system
        system = MMASThreeAgentSystem(args)
        
        # Preloading model
        system.preload_models()
        
        # Loading Data集
        cases = system.load_dataset()
        if not cases:
            print("incorrect: no法Loading Data集")
            return
        
        # Processing case
        results = system.process_cases(cases)
        
        # 打印summary
        system.print_summary()
        
        # OutputlogfilePosition
        from log_manager import get_log_manager
        log_manager = get_log_manager()
        if log_manager and log_manager.is_active:
            print(f"logfilesaved to: {log_manager.log_file}")
        
    except KeyboardInterrupt:
        print("\n用户InterruptedExecuting ")
    except Exception as e:
        print(f"systemExecuting Failed : {e}")
        if args.debug:
            traceback.print_exc()
    finally:
        # 确保ProgramEnd时OutputlogfilePosition
        try:
            from log_manager import get_log_manager
            log_manager = get_log_manager()
            if log_manager and log_manager.is_active:
                print(f"\n最终logfilePosition: {log_manager.log_file}")
        except:
            pass

if __name__ == "__main__":
    main()
