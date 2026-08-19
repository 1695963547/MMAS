#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMAS Evaluator Agent
Implemented based on the MMAS three-agent pipeline

Features:
1. Step 2: Evaluator agent information quality assessment
2. Step 3: Information clarification
3. Step 7: Evaluator agent final decision
"""

import json
import re
import logging
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from transformers import StoppingCriteria, StoppingCriteriaList
from unified_expert_template_manager import UnifiedExpertTemplateManager
from global_model_manager import GlobalModelManager

# Set up logging
logger = logging.getLogger(__name__)

class StopOnToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_token_str, skip_first_occurrence=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_str = stop_token_str
        self.stop_token_ids = tokenizer.encode(stop_token_str, add_special_tokens=False)
        self.skip_first_occurrence = skip_first_occurrence
        self.occurrence_count = 0
        self.last_checked_length = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check newly generated part to avoid repeated decoding
        current_length = len(input_ids[0])
        if current_length <= self.last_checked_length:
            return False
            
        # Get newly generated part (excluding input prompt)
        if not hasattr(self, 'input_length'):
            # Record input length on first call
            self.input_length = current_length
            return False
            
        # Only decode newly generated tokens
        if current_length > self.input_length:
            generated_tokens = input_ids[0][self.input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            return False
        
        # Check if contains complete stop sequence
        if self.stop_token_str in generated_text:
            # Count occurrences of stop sequence in generated text
            current_count = generated_text.count(self.stop_token_str)
            
            # If count increased, a new stop sequence was encountered
            if current_count > self.occurrence_count:
                self.occurrence_count = current_count
                self.last_checked_length = current_length
                
                # If skipping first occurrence, only stop on second and subsequent
                if self.skip_first_occurrence:
                    if self.occurrence_count >= 2:
                        return True
                else:
                    return True
        
        self.last_checked_length = current_length
        return False

class MMASEvaluatorAgent:
    """
    Evaluator Agent
    负责evaluationInformation质量、Generating 澄清question、evaluation最终diagnosis质量
    """
    def __init__(self, template_manager: UnifiedExpertTemplateManager, show_step2_details: bool = False, clarification_threshold: int = 20, api_client=None, use_api_model=False, api_model_name=None):
        """Initializing Evaluator Agent"""
        self.model_manager = GlobalModelManager()
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        self.show_step2_details = show_step2_details
        self.clarification_threshold = clarification_threshold
        self.api_client = api_client
        self.use_api_model = use_api_model
        self.api_model_name = api_model_name
        self._cached_eos_token_ids = None  # Cache EOS token IDs to avoid repeated computation

    def step_2_assess_quality(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """evaluation第二Step：evaluationcaseInformation质量"""
        
        # from case_data 提取caseInformation
        case_info = self._build_case_info_for_assessment(case_data)
        case_id = case_data.get('id', 'unknown')
        
        assessment_template = self.template_manager.get_evaluator_template(
            "step2_quality_assessment",
            case_info=case_info
        )

        # Record详细InputInformationtolog
        self.logger.info("="*80)
        self.logger.info(f"🔍 step二质量evaluation - caseID: {case_id}")
        self.logger.info("="*80)
        self.logger.info(f"📝 InputcaseInformation:\n{case_info}")
        self.logger.info("-"*80)
        self.logger.info(f"🤖 LLMInput提示:\n{assessment_template}")
        self.logger.info("="*80)

        raw_response, total_tokens = self._call_llm_for_assessment(assessment_template)
        
        # Record详细OutputInformationtolog
        self.logger.info("="*80)
        self.logger.info(f"🔤 LLMOriginalresponse - caseID: {case_id}")
        self.logger.info("="*80)
        self.logger.info(raw_response)
        self.logger.info("="*80)

        parsed_result = self._parse_assessment_result(raw_response, case_id)

        # RecordParsing resulttolog
        self.logger.info("="*80)
        self.logger.info(f"📊 step二Parsing result - caseID: {case_id}")
        self.logger.info("="*80)
        self.logger.info(json.dumps(parsed_result, indent=2, ensure_ascii=False))
        self.logger.info("="*80)

        step2_result = {
            'case_id': case_id,
            'step2_scores': parsed_result,
            'step2_raw_output': raw_response,
            'needs_clarification': False, 
            'total_tokens': total_tokens
        }

        if parsed_result and 'total_score' in parsed_result:
            if parsed_result['total_score'] < self.clarification_threshold:
                step2_result['needs_clarification'] = True
        
        # Record最终decisiontolog
        clarification_status = "需要澄清" if step2_result['needs_clarification'] else "直接diagnosis"
        total_score = parsed_result.get('total_score', 0) if parsed_result else 0
        self.logger.info(f"✅ step二complete - caseID: {case_id}, 总分: {total_score}, Threshold: {self.clarification_threshold}, decision: {clarification_status}")
        
        return step2_result

    def _prepare_case_data_for_assessment(self, case_data: Dict[str, Any]) -> Tuple[str, str, Dict, Dict]:
        """准备用于evaluationcaseData"""
        context = case_data.get('context', [])
        question = case_data.get('question', '')
        options = case_data.get('options', {})
        patient_info = case_data.get('patient', {})
        
        context_str = context[0] if isinstance(context, list) and len(context) > 0 else str(context)
        
        return context_str, question, options, patient_info





    def _sanitize_assessment_for_template(self, assessment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        轻量化assessment_result，仅保留Generating 澄清question所需字Segment，排除raw_responseetc.大字Segment，防止Prompt爆炸。
        """
        allowed_keys = [
            'decision', 'quality_score', 'quality_level', 
            'clarification_points', 'clarification_questions',
            'low_scores', 'summary'
        ]
        sanitized = {k: assessment_result[k] for k in allowed_keys if k in assessment_result}
        # 确保关键decisionInformation存
        if not sanitized.get('clarification_points') and not sanitized.get('clarification_questions'):
            sanitized['decision'] = assessment_result.get('decision', sanitized.get('decision'))
            sanitized['quality_level'] = assessment_result.get('quality_level', sanitized.get('quality_level'))
            sanitized['quality_score'] = assessment_result.get('quality_score', sanitized.get('quality_score'))
        return sanitized

    def _call_llm_for_assessment(self, prompt: str) -> Tuple[str, int]:
        """调用LLM进行evaluation，return (Generating Text, 总token数)"""
        if self.use_api_model and self.api_client and self.api_model_name:
            # Using API model
            try:
                api_result = self.api_client.generate(
                    model_name=self.api_model_name,
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )
                generated_text = api_result.get('generated_text', '')
                total_tokens = api_result.get('usage', {}).get('total_tokens', 0)
                return generated_text, total_tokens
            except Exception as e:
                self.logger.error(f"API model call failed: {e}")
                raise
        else:
            # 使用this地模型
            model, tokenizer = self.model_manager.get_model('main_llm')
            
            # 构建消息格式
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            input_tokens = inputs.input_ids.shape[-1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=self._get_safe_eos_token_ids(tokenizer)
                )
            
            generated_tokens = outputs[0].shape[-1] - input_tokens
            total_tokens = int(input_tokens + generated_tokens)
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            
            return generated_text, total_tokens
    
    def _get_safe_eos_token_ids(self, tokenizer):
        """安全Getting 并cacheeos_token_idList，避免DuplicateComputing """
        if self._cached_eos_token_ids is not None:
            return self._cached_eos_token_ids

        eos_token_ids = {tokenizer.eos_token_id}
        
        # 尝试Getting 特殊EOS tokenID（适配LlamaandQwen模型）
        special_tokens = ["<|eot_id|>", "<|im_end|>"]
        for token in special_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and isinstance(token_id, int) and token_id > 0:
                    eos_token_ids.add(token_id)
                    self.logger.debug(f"Successfully 添加特殊EOS token '{token}' (ID: {token_id})")
            except Exception as e:
                self.logger.debug(f"Getting 特殊EOS token '{token}' IDFailed : {e}")
        
        final_ids = list(eos_token_ids)
        self.logger.info(f"最终确定EOS token IDs: {final_ids}")
        self._cached_eos_token_ids = final_ids
        return final_ids

    def _build_case_info_for_assessment(self, case_data: Dict[str, Any]) -> str:
        """构建用于evaluationcaseInformationString"""
        context = case_data.get('context', [])
        question = case_data.get('question', '')
        options = case_data.get('options', {})
        patient_info = case_data.get('patient', {})
        
        case_info = ""
        
        # 只使用context[0]
        if isinstance(context, list) and len(context) > 0:
            case_info += "case描述:\n" + context[0] + "\n\n"
        elif context:
            case_info += "case描述:\n" + str(context) + "\n\n"
        
        # 添加patient年龄and性别Information
        if patient_info:
            patient_age = patient_info.get('age', '')
            patient_gender = patient_info.get('gender', '')
            if patient_age or patient_gender:
                case_info += "patientInformation:\n"
                if patient_age:
                    case_info += f"年龄: {patient_age}\n"
                if patient_gender:
                    case_info += f"性别: {patient_gender}\n"
                case_info += "\n"
        
        if question:
            case_info += "question:\n" + question + "\n\n"
        
        if options:
            case_info += "Option:\n"
            for key, value in options.items():
                case_info += f"{key}. {value}\n"
        
        return case_info
    
    def _parse_assessment_result(self, raw_response: str, case_id: str) -> Optional[Dict[str, Any]]:
        """Parsing evaluationresult - 增强版，优先提取```jsonCodeBlock并implements清理管道"""
        try:
            # 提取 thinking partial
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
            thinking_content = thinking_match.group(1).strip() if thinking_match else ""
            
            # step1：优先提取 ```json CodeBlock
            json_str = None
            json_block_match = re.search(r'```json\s*\n?(.*?)\n?```', raw_response, re.DOTALL | re.IGNORECASE)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
                self.logger.info(f"提取toJSONCodeBlock: {json_str}")
            else:
                # 如果没hasCodeBlock，尝试提取普通JSONObject
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response)
                if json_match:
                    json_str = json_match.group(0)
                    self.logger.info(f"提取toJSONObject: {json_str}")
            
            if json_str:
                # step2：implements清理管道 - perOrder尝试Parsing and修复
                scores = None
                
                # 2.1 尝试直接Parsing JSON
                try:
                    scores = json.loads(json_str)
                    self.logger.info("JSON直接Parsing Successfully ")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON直接Parsing Failed : {e}")
                
                # 2.2 如果Failed ，perOrder管道清理
                if scores is None:
                    # 管道step1: _fix_malformed_json
                    try:
                        fixed_json_str = self._fix_malformed_json(json_str)
                        scores = json.loads(fixed_json_str)
                        self.logger.info(f"_fix_malformed_json修复Successfully : {fixed_json_str}")
                    except json.JSONDecodeError:
                        self.logger.warning("_fix_malformed_json修复Failed ")
                
                # 管道step2: _fix_common_json_issues
                if scores is None:
                    try:
                        common_fixed_str = self._fix_common_json_issues(json_str)
                        scores = json.loads(common_fixed_str)
                        self.logger.info(f"_fix_common_json_issues修复Successfully : {common_fixed_str}")
                    except json.JSONDecodeError:
                        self.logger.warning("_fix_common_json_issues修复Failed ")
                
                # 管道step3: _clean_json_string_robust
                if scores is None:
                    try:
                        robust_cleaned_str = self._clean_json_string_robust(json_str)
                        scores = json.loads(robust_cleaned_str)
                        self.logger.info(f"_clean_json_string_robust修复Successfully : {robust_cleaned_str}")
                    except json.JSONDecodeError:
                        self.logger.warning("_clean_json_string_robust修复Failed ")
                
                # 管道step4: _attempt_json_repair
                if scores is None:
                    try:
                        repaired_str = self._attempt_json_repair(json_str)
                        if repaired_str:
                            scores = json.loads(repaired_str)
                            self.logger.info(f"_attempt_json_repair修复Successfully : {repaired_str}")
                    except json.JSONDecodeError:
                        self.logger.warning("_attempt_json_repair修复Failed ")
                
                # 管道step5: 最后才使用 _extract_scores_from_malformed_json
                if scores is None:
                    self.logger.warning("allJSON修复MethodFailed ，使用备用Parsing Method")
                    scores = self._extract_scores_from_malformed_json(json_str)
                
                if not scores:
                    self.logger.warning("no法Parsing JSON，使用Defaultscore")
                    return self._create_default_scores(case_id)
                
                # standard化字Segment名并提取score
                normalized_scores = self._normalize_score_fields(scores)
                
                # Computing 总分
                total_score = sum([
                    normalized_scores.get('basic_score', 0),
                    normalized_scores.get('symptom_score', 0),
                    normalized_scores.get('exam_score', 0),
                    normalized_scores.get('timeline_score', 0),
                    normalized_scores.get('logic_score', 0)
                ])
                
                # 判断is否需要澄清 - 当总分大于etc.于Threshold时不需要澄清
                needs_clarification = total_score < self.clarification_threshold
                quality_level = self._get_quality_level(total_score)
                
                return {
                    'basic_score': normalized_scores.get('basic_score', 0),
                    'symptom_score': normalized_scores.get('symptom_score', 0),
                    'exam_score': normalized_scores.get('exam_score', 0),
                    'timeline_score': normalized_scores.get('timeline_score', 0),
                    'logic_score': normalized_scores.get('logic_score', 0),
                    'total_score': total_score,
                    'needs_clarification': needs_clarification,
                    'decision': 'needs_clarification' if needs_clarification else 'direct_diagnosis',
                    'quality_level': quality_level,
                    'quality_score': total_score,
                    'case_id': case_id,
                    'step': 'step2_quality_assessment',
                    'agent': 'evaluator_agent',
                    'timestamp': datetime.now().isoformat(),
                    'raw_response': raw_response,
                    'thinking_content': thinking_content
                }
            else:
                logger.warning("no法fromresponse提取JSON格式评分")
                return self._create_default_scores(case_id)
                
        except Exception as e:
            logger.error(f"Parsing evaluationresultFailed : {str(e)}")
            return self._create_default_scores(case_id)
    
    def _get_quality_level(self, total_score: int) -> str:
        """根据总分判断质量etc.级"""
        if total_score >= 40:
            return "优秀"
        elif total_score >= 30:
            return "良好"
        elif total_score >= 20:
            return "一般"
        elif total_score >= 10:
            return "较差"
        else:
            return "很差"
    
    def _create_default_scores(self, case_id: int = 0) -> Dict[str, Any]:
        """Creating Default评分result"""
        return {
            'basic_score': 0,
            'symptom_score': 0,
            'exam_score': 0,
            'timeline_score': 0,
            'logic_score': 0,
            'total_score': 0,
            'needs_clarification': True,
            'decision': '需要澄清',
            'quality_level': '很差',
            'quality_score': 0,
            'case_id': case_id,
            'step': 'step2_quality_assessment',
            'agent': 'evaluator_agent',
            'timestamp': datetime.now().isoformat(),
            'raw_response': ''
        }
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """修复格式incorrectJSONString"""
        # 替换文冒号as英文冒号
        fixed_str = json_str.replace('：', ':')
        
        # 转换单引号as双引号（Processing 键and值）
        # 先Processing 键名单引号
        fixed_str = re.sub(r"'([^']*?)'(\s*:)", r'"\1"\2', fixed_str)
        # Processing String值单引号
        fixed_str = re.sub(r":\s*'([^']*?)'", r': "\1"', fixed_str)
        
        # 移除非法Character（如文逗号、加号etc.）
        fixed_str = re.sub(r'[，＋]', '', fixed_str)
        fixed_str = re.sub(r'["""]', '"', fixed_str)  # 统一引号
        
        # 修复StringNumber（移除引号）
        fixed_str = re.sub(r':\s*"(\d+)"', r': \1', fixed_str)
        
        # 修复带has非法CharacterNumber值
        fixed_str = re.sub(r':\s*"?\+?(\d+)[，。]*"?', r': \1', fixed_str)
        
        # Extension字Segment名映射，包括连Character、下划线andSize写变体
        field_mappings = {
            # 基thisInformation评分变体
            'basic-score': 'basic_score',
            'basic_info': 'basic_score',
            'basic-info': 'basic_score',
            'basicScore': 'basic_score',
            'Basic_Score': 'basic_score',
            'BASIC_SCORE': 'basic_score',
            'basicscore': 'basic_score',
            'basic_info_completeness': 'basic_score',
            'basic-info-completeness': 'basic_score',
            
            # symptom评分变体
            'symptom-score': 'symptom_score',
            'symptomScore': 'symptom_score',
            'Symptom_Score': 'symptom_score',
            'SYMPTOM_SCORE': 'symptom_score',
            'syntax_score': 'symptom_score',
            'symphom_score': 'symptom_score',
            'scores_symptom': 'symptom_score',
            'scoresymptom': 'symptom_score',
            'symptom_adequacy': 'symptom_score',
            'symptom-adequacy': 'symptom_score',
            
            # 体检评分变体
            'exam-score': 'exam_score',
            'examScore': 'exam_score',
            'Exam_Score': 'exam_score',
            'EXAM_SCORE': 'exam_score',
            'physical_exam_completeness': 'exam_score',
            'physical-exam-completeness': 'exam_score',
            'physicalExamCompleteness': 'exam_score',
            
            # Time线评分变体
            'timeline-score': 'timeline_score',
            'timelineScore': 'timeline_score',
            'Timeline_Score': 'timeline_score',
            'TIMELINE_SCORE': 'timeline_score',
            'timeline_clarity': 'timeline_score',
            'timeline-clarity': 'timeline_score',
            'timelineClarity': 'timeline_score',
            
            # 逻辑评分变体
            'logic-score': 'logic_score',
            'logicScore': 'logic_score',
            'Logic_Score': 'logic_score',
            'LOGIC_SCORE': 'logic_score',
            'logicscore': 'logic_score',
            'logical_consistency': 'logic_score',
            'logical-consistency': 'logic_score',
            'logicalConsistency': 'logic_score'
        }
        
        # 应用字Segment名映射
        for old_field, new_field in field_mappings.items():
            # Processing 带引号字Segment名
            fixed_str = re.sub(f'"{old_field}"', f'"{new_field}"', fixed_str, flags=re.IGNORECASE)
            # Processing 不带引号字Segment名（可能出现某格式）
            fixed_str = re.sub(f'\\b{re.escape(old_field)}\\b(?=\\s*:)', f'"{new_field}"', fixed_str, flags=re.IGNORECASE)
        
        # 调用更强清理Function
        fixed_str = self._clean_json_string_robust(fixed_str)
        
        return fixed_str
    
    def _extract_scores_from_malformed_json(self, json_str: str) -> Dict[str, Any]:
        """from格式incorrectJSON提取score"""
        scores = {}
        
        # ExtensionRegularizationExpressionmode，support连Character、下划线andSize写混合
        patterns = [
            # 基thisInformation评分变体
            (r'["\']?basic[-_\s]*(?:info[-_\s]*)?(?:completeness[-_\s]*)?score["\']?\s*:\s*(\d+)', 'basic_score'),
            (r'["\']?basic[-_\s]*(?:info|Info|INFO)[-_\s]*(?:completeness|Completeness|COMPLETENESS)?["\']?\s*:\s*(\d+)', 'basic_score'),
            (r'["\']?(?:basic|Basic|BASIC)[-_\s]*(?:score|Score|SCORE)["\']?\s*:\s*(\d+)', 'basic_score'),
            
            # symptom评分变体
            (r'["\']?symptom[-_\s]*(?:adequacy[-_\s]*)?score["\']?\s*:\s*(\d+)', 'symptom_score'),
            (r'["\']?(?:symptom|Symptom|SYMPTOM)[-_\s]*(?:adequacy|Adequacy|ADEQUACY)?["\']?\s*:\s*(\d+)', 'symptom_score'),
            (r'["\']?(?:symptom|Symptom|SYMPTOM)[-_\s]*(?:score|Score|SCORE)["\']?\s*:\s*(\d+)', 'symptom_score'),
            (r'["\']?(?:syntax|symphom)[-_\s]*score["\']?\s*:\s*(\d+)', 'symptom_score'),  # 常见incorrect
            
            # 体检评分变体
            (r'["\']?exam[-_\s]*score["\']?\s*:\s*(\d+)', 'exam_score'),
            (r'["\']?(?:exam|Exam|EXAM)[-_\s]*(?:score|Score|SCORE)["\']?\s*:\s*(\d+)', 'exam_score'),
            (r'["\']?physical[-_\s]*exam[-_\s]*(?:completeness[-_\s]*)?["\']?\s*:\s*(\d+)', 'exam_score'),
            (r'["\']?(?:physical|Physical|PHYSICAL)[-_\s]*(?:exam|Exam|EXAM)[-_\s]*(?:completeness|Completeness|COMPLETENESS)?["\']?\s*:\s*(\d+)', 'exam_score'),
            
            # Time线评分变体
            (r'["\']?timeline[-_\s]*(?:clarity[-_\s]*)?score["\']?\s*:\s*(\d+)', 'timeline_score'),
            (r'["\']?(?:timeline|Timeline|TIMELINE)[-_\s]*(?:clarity|Clarity|CLARITY)?["\']?\s*:\s*(\d+)', 'timeline_score'),
            (r'["\']?(?:timeline|Timeline|TIMELINE)[-_\s]*(?:score|Score|SCORE)["\']?\s*:\s*(\d+)', 'timeline_score'),
            
            # 逻辑评分变体
            (r'["\']?logic[-_\s]*(?:al[-_\s]*)?(?:consistency[-_\s]*)?score["\']?\s*:\s*(\d+)', 'logic_score'),
            (r'["\']?(?:logic|Logic|LOGIC)[-_\s]*(?:al|Al|AL)?[-_\s]*(?:consistency|Consistency|CONSISTENCY)?["\']?\s*:\s*(\d+)', 'logic_score'),
            (r'["\']?(?:logic|Logic|LOGIC)[-_\s]*(?:score|Score|SCORE)["\']?\s*:\s*(\d+)', 'logic_score'),
            (r'["\']?logical[-_\s]*consistency["\']?\s*:\s*(\d+)', 'logic_score'),
        ]
        
        for pattern, field_name in patterns:
            match = re.search(pattern, json_str, re.IGNORECASE)
            if match:
                try:
                    score_value = int(match.group(1))
                    # 只has当the字Segment还没hasbySetting 时才Setting ，避免Duplicatematching
                    if field_name not in scores:
                        scores[field_name] = score_value
                except ValueError:
                    continue
        
        return scores
    
    def _normalize_score_fields(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        standard化评分result字Segment名，并值转换asInteger。
        例如： 'basic_completeness' or '基thiscomplete性' 转换as 'basic_score'。
        """
        if not scores:
            return {}

        normalized = {
            'basic_score': 0,
            'symptom_score': 0,
            'exam_score': 0,
            'timeline_score': 0,
            'logic_score': 0
        }

        # Extension字Segment映射，support更多变体
        field_mappings = {
            # 基础评分变体
            'basic_score': 'basic_score',
            'basic-score': 'basic_score',
            'basicScore': 'basic_score',
            'Basic_Score': 'basic_score',
            'BASIC_SCORE': 'basic_score',
            'basic_completeness': 'basic_score',
            'basicCompleteness': 'basic_score',
            'patient_demographics': 'basic_score',
            'patient-demographics': 'basic_score',
            'patientDemographics': 'basic_score',
            'basic_info': 'basic_score',
            'basic-info': 'basic_score',
            'basic_info_completeness': 'basic_score',
            'basic-info-completeness': 'basic_score',
            'basicInfoCompleteness': 'basic_score',
            'basicscore': 'basic_score',

            # symptom评分变体
            'symptom_score': 'symptom_score',
            'symptom-score': 'symptom_score',
            'symptomScore': 'symptom_score',
            'Symptom_Score': 'symptom_score',
            'SYMPTOM_SCORE': 'symptom_score',
            'syntax_score': 'symptom_score',
            'symphom_score': 'symptom_score',
            'scores_symptom': 'symptom_score',
            'scoresymptom': 'symptom_score',
            'symptom_adequacy': 'symptom_score',
            'symptom-adequacy': 'symptom_score',
            'symptomAdequacy': 'symptom_score',
            
            # 体检评分变体
            'exam_score': 'exam_score',
            'exam-score': 'exam_score',
            'examScore': 'exam_score',
            'Exam_Score': 'exam_score',
            'EXAM_SCORE': 'exam_score',
            'physical_exam_completeness': 'exam_score',
            'physical-exam-completeness': 'exam_score',
            'physicalExamCompleteness': 'exam_score',
            'Physical_Exam_Completeness': 'exam_score',
            'PHYSICAL_EXAM_COMPLETENESS': 'exam_score',
            
            # Time线评分变体
            'timeline_score': 'timeline_score',
            'timeline-score': 'timeline_score',
            'timelineScore': 'timeline_score',
            'Timeline_Score': 'timeline_score',
            'TIMELINE_SCORE': 'timeline_score',
            'timeline_clarity': 'timeline_score',
            'timeline-clarity': 'timeline_score',
            'timelineClarity': 'timeline_score',
            'Timeline_Clarity': 'timeline_score',
            'TIMELINE_CLARITY': 'timeline_score',
            
            # 逻辑评分变体
            'logic_score': 'logic_score',
            'logic-score': 'logic_score',
            'logicScore': 'logic_score',
            'Logic_Score': 'logic_score',
            'LOGIC_SCORE': 'logic_score',
            'logicscore': 'logic_score',
            'logical_consistency': 'logic_score',
            'logical-consistency': 'logic_score',
            'logicalConsistency': 'logic_score',
            'Logical_Consistency': 'logic_score',
            'LOGICAL_CONSISTENCY': 'logic_score'
        }
        
        def _standardize_key(key: str) -> str:
            """standard化键名：去除非字母Character并转换as小写"""
            # 去除非字母Character（保留下划线用于分隔）
            clean_key = re.sub(r'[^a-zA-Z_]', '', key.lower())
            return clean_key
        
        for original_field, value in scores.items():
            # 首先尝试直接matching
            standard_field = field_mappings.get(original_field, None)
            
            # 如果直接matchingFailed ，尝试standard化后matching
            if not standard_field:
                standardized_key = _standardize_key(original_field)
                standard_field = field_mappings.get(standardized_key, None)
            
            # 如果还is没hasmatching，尝试更宽松matching
            if not standard_field:
                for mapped_key, target_field in field_mappings.items():
                    if _standardize_key(mapped_key) == standardized_key:
                        standard_field = target_field
                        break
            
            if standard_field:
                # 确保值asInteger，并截断to合法Range [0, 10]
                try:
                    if isinstance(value, str):
                        # 移除String非NumberCharacter
                        clean_value = re.sub(r'[^\d]', '', value)
                        if clean_value:
                            score_val = int(clean_value)
                        else:
                            continue
                    elif isinstance(value, (int, float)):
                        score_val = int(value)
                    else:
                        continue
                    # 截断to [0, 10] Range
                    if score_val < 0 or score_val > 10:
                        logger.warning(f"评分超Range，已截断to[0,10]: {original_field}={value} -> {max(0, min(10, score_val))}")
                        score_val = max(0, min(10, score_val))
                    normalized[standard_field] = score_val
                except (ValueError, TypeError):
                    logger.warning(f"no法转换score值: {original_field}={value}")
                    continue
        
        return normalized
    
    def _clean_json_string_robust(self, json_str: str) -> str:
        """增强JSONString清理Function - Processing 更多格式question"""
        # 移除多余逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 修复引号question - 确保键名has引号
        json_str = re.sub(r'([{,]\s*)"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:', r'\1"\2":', json_str)
        
        # 移除字Segment名多余空格并用下划线替换
        json_str = re.sub(r'"([^"]*)\s+([^"]*)":', r'"\1_\2":', json_str)
        
        # 修复常见字Segment名incorrectand变体
        field_corrections = {
            '"basic_compleness"': '"basic_completeness"',
            '"basic_complete"': '"basic_completeness"',
            '"basicCompleteness"': '"basic_completeness"',
            '"基础complete性"': '"basic_completeness"',
            '"基础Informationcomplete性"': '"basic_completeness"',
            '"symptom_adecuacy"': '"symptom_adequacy"',
            '"symptom_adequate"': '"symptom_adequacy"',
            '"symptomAdequacy"': '"symptom_adequacy"',
            '"symptom充分性"': '"symptom_adequacy"',
            '"symptom描述充分性"': '"symptom_adequacy"',
            '"logical_coherency"': '"logical_consistency"',
            '"logical_coherence"': '"logical_consistency"',
            '"logicalConsistency"': '"logical_consistency"',
            '"逻辑一致性"': '"logical_consistency"',
            '"逻辑连贯性"': '"logical_consistency"',
            '"assessmentScores"': '"assessment_scores"',
            '"assessment_scores"': '"scores"',
            '"Total_Score"': '"total_score"',
            '"totalScore"': '"total_score"',
            '"Quality_Level"': '"quality_level"',
            '"qualityLevel"': '"quality_level"',
            '"Decision"': '"decision"',
            '"_Clarification_Questions"': '"clarification_questions"'
        }
        
        for wrong, correct in field_corrections.items():
            json_str = json_str.replace(wrong, correct)
        
        # 修复值英文Content
        value_corrections = {
            '"good"': '"良好"',
            '"excellent"': '"优秀"',
            '"Excellent"': '"优秀"',
            '"directly_transfer_to_expert"': '"直接转expert"',
            '"need_clarification"': '"需要澄清"'
        }
        
        for wrong, correct in value_corrections.items():
            json_str = json_str.replace(wrong, correct)
        
        return json_str

    def _fix_common_json_issues(self, json_str: str) -> str:
        """修复常见JSON格式question"""
        # 移除可能前后缀Text
        json_str = re.sub(r'^[^{]*', '', json_str)  # 移除开头非JSONCharacter
        json_str = re.sub(r'[^}]*$', '', json_str)  # 移除结尾非JSONCharacter
        
        # 修复单引号as双引号
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # 修复没has引号键名
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # 修复Number后面多余逗号
        json_str = re.sub(r'(\d+),\s*([}\]])', r'\1\2', json_str)
        
        # 修复String后面多余逗号
        json_str = re.sub(r'("[^"]*"),\s*([}\]])', r'\1\2', json_str)
        
        return json_str

    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """尝试修复损坏JSONString"""
        try:
            # 尝试1: 移除all换行符and多余空格
            repaired = re.sub(r'\s+', ' ', json_str.strip())
            json.loads(repaired)
            return repaired
        except:
            pass
        
        try:
            # 尝试2: 确保allString都has引号
            repaired = json_str
            # 查找all可能键值for
            pattern = r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,}]+)'
            
            def fix_value(match):
                prefix = match.group(1)
                key = match.group(2)
                value = match.group(3).strip()
                
                # 如果值isNumber，保持不变
                if re.match(r'^\d+$', value):
                    return f'{prefix}"{key}": {value}'
                # 如果值已经has引号，保持不变
                elif value.startswith('"') and value.endswith('"'):
                    return f'{prefix}"{key}": {value}'
                # 否则添加引号
                else:
                    return f'{prefix}"{key}": "{value}"'
            
            repaired = re.sub(pattern, fix_value, repaired)
            json.loads(repaired)
            return repaired
        except:
            pass
        
        try:
            # 尝试3: 构建一最小validJSON
            # fromOriginalString提取Number
            numbers = re.findall(r'\d+', json_str)
            if len(numbers) >= 3:
                return f'{{"basic_completeness": {numbers[0]}, "symptom_adequacy": {numbers[1]}, "logical_consistency": {numbers[2]}}}'
        except:
            pass
        
        return None

    def _extract_scores_from_json(self, data: dict) -> Dict[str, int]:
        """fromJSONData提取score，support新五Dimension评分"""
        result = {
            'basic_info_completeness': 0,
            'symptom_adequacy': 0,
            'physical_exam_completeness': 0,
            'timeline_clarity': 0,
            'logical_consistency': 0,
        }
        
        # 定义可能键名变体
        key_variants = {
            'basic_info_completeness': [
                'basic_info_completeness', 'basic_completeness', 'basicCompleteness', 'basic_complete',
                'basic_compleness', '基础complete性', '基础Informationcomplete性', 'basic_info'
            ],
            'symptom_adequacy': [
                'symptom_adequacy', 'symptomAdequacy', 'symptom_adequate',
                'symptom_adecuacy', 'symptom充分性', 'symptom描述充分性', 'symptom_description_adequacy'
            ],
            'physical_exam_completeness': [
                'physical_exam_completeness', 'physical_exam', 'exam_completeness', 'physical',
                'examination', '体格Checking complete性', '体检complete性', 'physical_examination'
            ],
            'timeline_clarity': [
                'timeline_clarity', 'timeline', 'clarity', 'time_clarity', 'chronology',
                'Time线清晰度', 'Time脉络', 'TimeOrder', 'temporal_clarity'
            ],
            'logical_consistency': [
                'logical_consistency', 'logicalConsistency', 'logical_coherence',
                'logical_coherency', '逻辑一致性', '逻辑连贯性', 'logical_coherence'
            ]
        }
        
        # 尝试from不同hierarchy提取score
        for target_key, variants in key_variants.items():
            score = 0
            
            # 直接from根Level查找
            for variant in variants:
                if variant in data:
                    value = data[variant]
                    if isinstance(value, (int, float)):
                        score = int(value)
                        break
                    elif isinstance(value, dict) and 'score' in value:
                        score = int(value['score'])
                        break
                    elif isinstance(value, str) and value.isdigit():
                        score = int(value)
                        break
            
            # 如果没找to，尝试from嵌套Object查找
            if score == 0:
                for key, value in data.items():
                    if isinstance(value, dict):
                        for variant in variants:
                            if variant in value:
                                nested_value = value[variant]
                                if isinstance(nested_value, (int, float)):
                                    score = int(nested_value)
                                    break
                                elif isinstance(nested_value, str) and nested_value.isdigit():
                                    score = int(nested_value)
                                    break
                        if score > 0:
                            break
            
            result[target_key] = score
        
        # 提取额外Information
        if 'overall_score' in data:
            result['overall_score'] = data['overall_score']
        if 'major_defect_dimensions' in data:
            result['major_defect_dimensions'] = data['major_defect_dimensions']
        if 'specific_defects' in data:
            result['specific_defects'] = data['specific_defects']
        
        return result

    def _extract_scores_from_text(self, text: str) -> Optional[Dict[str, int]]:
        """from纯Text提取score作as最后备选plan - support新五Dimension评分"""
        result = {
            'basic_info_completeness': 0,
            'symptom_adequacy': 0,
            'physical_exam_completeness': 0,
            'timeline_clarity': 0,
            'logical_consistency': 0,
        }
        
        try:
            # 定义score提取mode - ExtensionVersion
            patterns = {
                'basic_info_completeness': [
                    r'基础[Information]*complete性[：:]\s*(\d+)',
                    r'基础[Information]*complete性.*?(\d+)[分点]',
                    r'basic[_\s]*info[_\s]*completeness[：:]\s*(\d+)',
                    r'基础.*?(\d+)/10',
                    r'基础.*?(\d+)分',
                    r'complete性.*?(\d+)',
                    r'基础Information.*?(\d+)',
                    r'"basic_info_completeness"[：:\s]*(\d+)',
                    r'基础complete性[：:\s]*(\d+)',
                    r'Informationcomplete.*?(\d+)'
                ],
                'symptom_adequacy': [
                    r'symptom[描述]*充分性[：:]\s*(\d+)',
                    r'symptom[描述]*充分性.*?(\d+)[分点]',
                    r'symptom[_\s]*adequacy[：:]\s*(\d+)',
                    r'symptom.*?(\d+)/10',
                    r'symptom.*?(\d+)分',
                    r'充分性.*?(\d+)',
                    r'symptom描述.*?(\d+)',
                    r'"symptom_adequacy"[：:\s]*(\d+)',
                    r'symptom充分性[：:\s]*(\d+)',
                    r'symptomInformation.*?(\d+)'
                ],
                'physical_exam_completeness': [
                    r'体格Checking complete性[：:]\s*(\d+)',
                    r'体格Checking complete性.*?(\d+)[分点]',
                    r'physical[_\s]*exam[_\s]*completeness[：:]\s*(\d+)',
                    r'体格Checking .*?(\d+)/10',
                    r'体格Checking .*?(\d+)分',
                    r'体检.*?(\d+)',
                    r'Checking complete.*?(\d+)',
                    r'"physical_exam_completeness"[：:\s]*(\d+)',
                    r'体检complete性[：:\s]*(\d+)',
                    r'体格.*?(\d+)'
                ],
                'timeline_clarity': [
                    r'Time线清晰度[：:]\s*(\d+)',
                    r'Time线清晰度.*?(\d+)[分点]',
                    r'timeline[_\s]*clarity[：:]\s*(\d+)',
                    r'Time线.*?(\d+)/10',
                    r'Time线.*?(\d+)分',
                    r'Time脉络.*?(\d+)',
                    r'TimeOrder.*?(\d+)',
                    r'"timeline_clarity"[：:\s]*(\d+)',
                    r'Time清晰度[：:\s]*(\d+)',
                    r'Time.*?(\d+)'
                ],
                'logical_consistency': [
                    r'逻辑一致性[：:]\s*(\d+)',
                    r'逻辑一致性.*?(\d+)[分点]',
                    r'logical[_\s]*consistency[：:]\s*(\d+)',
                    r'逻辑.*?(\d+)/10',
                    r'逻辑.*?(\d+)分',
                    r'一致性.*?(\d+)',
                    r'逻辑连贯.*?(\d+)',
                    r'"logical_consistency"[：:\s]*(\d+)',
                    r'逻辑一致性[：:\s]*(\d+)',
                    r'连贯性.*?(\d+)'
                ]
            }
            
            # 尝试提取eachscore
            for key, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            score = int(match)
                            if 0 <= score <= 10:  # validationscoreRange
                                result[key] = score
                                self.logger.debug(f"fromText提取to {key}: {score}")
                                break
                        except ValueError:
                            continue
                    if result[key] > 0:
                        break
            
            # 如果上述Method都没找toscore，尝试更通用Number提取
            if not any(score > 0 for score in result.values()):
                self.logger.debug("尝试通用Number提取Method")
                # 查找all可能scoremode
                score_patterns = [
                    r'(\d+)[分点]',  # X分 or X点
                    r'(\d+)/10',     # X/10
                    r'[：:]\s*(\d+)', # : X
                    r'得分[：:\s]*(\d+)', # 得分: X
                    r'score[：:\s]*(\d+)', # score: X
                ]
                
                all_scores = []
                for pattern in score_patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        try:
                            score = int(match)
                            if 0 <= score <= 10:
                                all_scores.append(score)
                        except ValueError:
                            continue
                
                # 如果找to5or更多score，取前5
                if len(all_scores) >= 5:
                    result['basic_info_completeness'] = all_scores[0]
                    result['symptom_adequacy'] = all_scores[1]
                    result['physical_exam_completeness'] = all_scores[2]
                    result['timeline_clarity'] = all_scores[3]
                    result['logical_consistency'] = all_scores[4]
                    self.logger.debug(f"通用Method提取toscore: {all_scores[:5]}")
                elif len(all_scores) > 0:
                    # 如果只找topartialscore，平均分配
                    avg_score = sum(all_scores) // len(all_scores)
                    result['basic_info_completeness'] = avg_score
                    result['symptom_adequacy'] = avg_score
                    result['physical_exam_completeness'] = avg_score
                    result['timeline_clarity'] = avg_score
                    result['logical_consistency'] = avg_score
                    self.logger.debug(f"partialscore平均分配: {avg_score}")
            
            # 如果找to至少一validscore，returnresult
            if any(score > 0 for score in result.values()):
                self.logger.debug(f"最终fromText提取score: {result}")
                return result
            else:
                self.logger.warning("未能fromText提取to任何validscore")
                return None
                
        except Exception as e:
            self.logger.error(f"fromText提取score时出错: {str(e)}")
            return None


    def _add_metadata_to_result(self, result: Dict, case_data: Dict, raw_response: str):
        """asresult添加元Data"""
        result.update({
            'case_id': case_data.get('id', 'unknown'),
            'step': 'step2_quality_assessment',
            'agent': 'evaluator_agent',
            'timestamp': self._get_timestamp(),
            'raw_response': raw_response
        })

    def _handle_step2_error(self, error_message: str) -> Dict:
        """Processing step二incorrect"""
        return {
            'error': error_message,
            'decision': 'error',
            'quality_score': 0,
            'quality_level': 'error',
            'step': 'step2_quality_assessment',
            'agent': 'evaluator_agent',
            'timestamp': self._get_timestamp()
        }

    def step3_clarification_generation(self, assessment_result: Dict[str, Any], case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Information clarification
        based onevaluationresultGenerating 针for性澄清question
        
        Args:
            assessment_result: step2evaluationresult
            case_data: caseData
            
        Returns:
            澄清questionList
        """
        try:
            # Checking is否需要澄清
            if not assessment_result.get('needs_clarification', False):
                self.logger.info("no需澄清，Skipping step3。")
                return {
                    'needs_clarification': False,
                    'clarification_questions': [],
                    'step': 'step3_clarification_generation',
                    'agent': 'evaluator_agent',
                    'timestamp': self._get_timestamp(),
                    'total_tokens': 0
                }
            
            # Getting 澄清template
            prompt = self.template_manager.get_evaluator_template(
                "step3_clarification_generation", 
                assessment_result=assessment_result, 
                case_data=case_data
            )
            
            # 不Recordstep3InputPrompttolog，仅保留Outputlog

            # 使用模型Generating 澄清question
            raw_response, total_tokens = self._call_llm_for_assessment(prompt)
            self.logger.debug(f"LLM raw response for clarification: {raw_response}")

            # Parsing 澄清question
            clarification_questions = self._parse_clarification_questions(raw_response)
            
            # 自动过滤泛泛question，保留has鉴别价值question
            filtered_questions = self._filter_discriminative_questions(clarification_questions)
            
            # 如果过滤后没hasquestion，使用高鉴别价值Defaultquestion
            if not filtered_questions:
                self.logger.warning("过滤后novalid澄清question，使用高鉴别价值Defaultquestion。")
                filtered_questions = self._get_discriminative_default_questions(case_data)

            result = {
                'needs_clarification': True,
                'clarification_questions': filtered_questions,
                'case_id': case_data.get('id', 'unknown'),
                'step': 'step3_clarification_generation',
                'agent': 'evaluator_agent',
                'timestamp': self._get_timestamp(),
                'raw_response': raw_response,
                'input_prompt': prompt,
                'total_tokens': total_tokens
            }
            
            self.logger.info(f"Evaluator Agent - Step 3: Generating 澄清question {case_data.get('id', 'unknown')}, Originalquestion数: {len(clarification_questions)}, 过滤后question数: {len(filtered_questions)}")
            # RecordOutputtolog
            try:
                self.logger.info("[step3-Output] OriginalLLMresponse:\n" + (raw_response if isinstance(raw_response, str) else str(raw_response)))
                self.logger.info("[step3-Output] 过滤后澄清questionList: " + json.dumps(filtered_questions, ensure_ascii=False))
            except Exception:
                pass
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluator Agent Step 3 incorrect: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'needs_clarification': True, # 出错时也认as需要澄清，但questionListas空
                'clarification_questions': ["请详细描述symptom具体部、性质、持续Timeand诱发因素。"],
                'step': 'step3_clarification_generation',
                'agent': 'evaluator_agent',
                'timestamp': self._get_timestamp()
            }
    
    def _filter_discriminative_questions(self, questions: List[str]) -> List[str]:
        """
        过滤澄清question：去重、基础清洗、质量过滤
        
        过滤rule：
        1. 去重and空值清理
        2. 过短question（<10Character）可能价值低
        3. 过于笼统question（如“请描述symptom”）没has针for性
        
        Args:
            questions: OriginalquestionList
            
        Returns:
            过滤后questionList
        """
        if not questions:
            return []
        
        # 过于笼统questionmode（缺乏针for性泛泛question）
        generic_patterns = [
            r'^\s*请(您?)?描述.{0,4}symptom\s*[??？]?\s*$',
            r'^\s*您?has什么symptom\s*[??？]?\s*$',
            r'^\s*请(您?)?描述.{0,4}情况\s*[??？]?\s*$',
            r'^\s*您?感觉如何\s*[??？]?\s*$',
            r'^\s*请详细说明\s*[??？]?\s*$',
        ]
        
        seen = set()
        filtered_questions = []
        for question in questions:
            if not question:
                continue
            question_clean = question.strip()
            if not question_clean:
                continue
            # 过滤过短question
            if len(question_clean) < 10:
                self.logger.debug(f"过滤过短question: '{question_clean}'")
                continue
            # 过滤过于笼统question
            is_generic = False
            for pattern in generic_patterns:
                if re.match(pattern, question_clean):
                    is_generic = True
                    self.logger.debug(f"过滤笼统question: '{question_clean}'")
                    break
            if is_generic:
                continue
            # 去重
            if question_clean not in seen:
                seen.add(question_clean)
                filtered_questions.append(question_clean)
        return filtered_questions
    
    def _get_discriminative_default_questions(self, case_data: Dict[str, Any]) -> List[str]:
        """
        Getting 高鉴别价值Defaultquestion
        
        Args:
            case_data: caseData
            
        Returns:
            DefaultquestionList
        """
        # 根据casetypeprovide不同Defaultquestion
        case_info = case_data.get('case_info', '').lower()
        
        default_questions = []
        
        # pain相关
        if 'pain' in case_info or '痛' in case_info:
            default_questions.append("请详细描述pain具体部、性质（刺痛/胀痛/绞痛etc.）andpain程度？")
        
        # imaging相关
        if any(keyword in case_info for keyword in ['ct', 'mri', 'imaging', '片子', 'x线']):
            default_questions.append("请描述病灶具体形态、Size、Boundary特征and密度/信号特点？")
        
        # symptom相关
        if any(keyword in case_info for keyword in ['symptom', '不适', 'exception']):
            default_questions.append("请说明symptom发作Time、持续Time、诱发因素and伴随symptom？")
        
        # 如果没hasmatching特定question，使用通用高鉴别价值question
        if not default_questions:
            default_questions = [
                "请详细描述symptom具体部、性质、持续Timeand诱发因素？",
                "请说明相关examination results具体数值、形态特征andTime变化？"
            ]
        
        return default_questions[:2]
    
    def step7_final_decision(self, doctor_opinions: List[Dict[str, Any]], expert_weights: Dict[str, float], case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 7: Evaluator agent final decision
        based ondoctordiagnosis意见andexpertWeight进行加权融合，Generating 最终答案
        
        Args:
            doctor_opinions: Doctor Agentdiagnosis意见List
            expert_weights: MOEexpertWeightDictionary
            case_data: caseData
            
        Returns:
            最终decisionresult
        """
        try:
            options = case_data.get('options', {})
            
            # Initializing Option得分
            option_scores = {option: 0.0 for option in options.keys()}
            total_weight = 0.0
            
            # 加权融合expert意见
            valid_expert_count = 0  # 跟踪validexpert数
            for opinion in doctor_opinions:
                expert_name = opinion.get('expert_name', '')
                expert_specialty = opinion.get('expert_specialty', '')
                expert_choice = opinion.get('choice', '')
                expert_confidence = opinion.get('confidence', 0.0)
                expert_scores = opinion.get('scores', {})
                
                # Skipping invalidexpert意见（no答案orhasincorrect）
                if opinion.get('error') or (not expert_choice and not expert_scores):
                    self.logger.warning(f"Skipping invalidexpert意见: {expert_name} (incorrect: {opinion.get('error', 'no答案')})")
                    continue
                
                valid_expert_count += 1
                
                # 使用specification化WeightmatchingFunctionGetting expertWeight
                weight, match_description = self._find_expert_weight(expert_name, expert_specialty, expert_weights)
                
                # 如果allmatchingstrategy都Failed ，使用均etc.Weight作as最后回退
                if weight == 0.0:
                    weight = 1.0 / len(doctor_opinions) if doctor_opinions else 1.0
                    self.logger.warning(f"WeightmatchingFailed  - expert: '{expert_name}' (specialty: '{expert_specialty}'), Reason: {match_description}")
                    self.logger.warning(f"availableWeight键: {list(expert_weights.keys())}")
                    self.logger.warning(f"使用均etc.Weight作as最后回退: {weight:.3f}")
                else:
                    self.logger.debug(f"WeightmatchingSuccessfully  - expert: '{expert_name}', {match_description}, Weight: {weight:.3f}")
                
                if weight > 0:
                    total_weight += weight
                    
                    # 如果has详细score，使用score进行加权
                    if expert_scores:
                        for option, score in expert_scores.items():
                            if option in option_scores:
                                option_scores[option] += weight * score
                    # 否则，给选择Option加权confidence
                    elif expert_choice in option_scores:
                        option_scores[expert_choice] += weight * expert_confidence
            
            # Checking is否hasvalidexpert意见
            if valid_expert_count == 0:
                self.logger.error("allexpert都未returnvalid答案，no法进行validdecision")
                # 构建inferencepath说明情况
                reasoning_path = "expert意见汇总：\n"
                for opinion in doctor_opinions:
                    name = opinion.get('expert_name', 'Unknown')
                    error = opinion.get('error', 'Unknown incorrect')
                    reasoning_path += f"- {name}: novalid答案 (incorrect: {error})\n"
                reasoning_path += "\nallexpert均未returnvaliddiagnosis，no法进行可靠decision。"
                
                return {
                    'final_answer': 'N/A',
                    'confidence_score': 0.0,
                    'option_scores': option_scores,
                    'expert_weights': expert_weights,
                    'reasoning_path': reasoning_path,
                    'is_correct': False,
                    'correct_answer': case_data.get('answer_idx', ''),
                    'case_id': case_data.get('id', 'unknown'),
                    'step': 'step7_final_decision',
                    'agent': 'evaluator_agent',
                    'timestamp': self._get_timestamp(),
                    'total_tokens': 0,
                    'no_valid_experts': True
                }
            
            # 归一化得分
            if total_weight > 0:
                for option in option_scores:
                    option_scores[option] /= total_weight
            
            # 选择最高得分Option
            final_choice = max(option_scores, key=option_scores.get) if option_scores else 'A'
            final_confidence = option_scores.get(final_choice, 0.0)
            
            # Processing 特殊情况：allvalidexpert选择相同Option
            valid_opinions = [op for op in doctor_opinions if not op.get('error') and op.get('choice')]
            unique_choices = set(op.get('choice', '') for op in valid_opinions)
            if len(unique_choices) == 1 and len(valid_opinions) > 1:
                # Computing 加权confidence - 使用_find_expert_weight进行specification化matching
                weighted_confidence = 0.0
                for opinion in valid_opinions:
                    expert_name = opinion.get('expert_name', '')
                    expert_specialty = opinion.get('expert_specialty', '')
                    w, _ = self._find_expert_weight(expert_name, expert_specialty, expert_weights)
                    weighted_confidence += w * opinion.get('confidence', 0.0)
                weighted_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
                
                if weighted_confidence > 0.8:
                    final_choice = list(unique_choices)[0]
                    final_confidence = weighted_confidence
            
            # 构建inferencepath
            reasoning_path = self._build_reasoning_path(doctor_opinions, expert_weights, option_scores)
            
            # Computing 准确率（如果hasstandard答案）
            correct_answer = case_data.get('answer_idx', '')
            is_correct = (final_choice == correct_answer) if correct_answer else None
            
            result = {
                'final_answer': final_choice,
                'confidence_score': final_confidence,
                'option_scores': option_scores,
                'expert_weights': expert_weights,
                'reasoning_path': reasoning_path,
                'is_correct': is_correct,
                'correct_answer': correct_answer,
                'case_id': case_data.get('id', 'unknown'),
                'step': 'step7_final_decision',
                'agent': 'evaluator_agent',
                'timestamp': self._get_timestamp(),
                'total_tokens': 0
            }
            
            self.logger.info(f"Evaluator Agent - Step 7: 最终decision {case_data.get('id', 'unknown')}, 选择: {final_choice}, confidence: {final_confidence:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluator Agent Step 7 incorrect: {str(e)}")
            return {
                'error': str(e),
                'final_answer': 'N/A',  # exception时returnN/A，而非DefaultA
                'confidence_score': 0.0,
                'step': 'step7_final_decision',
                'agent': 'evaluator_agent',
                'timestamp': self._get_timestamp()
            }
    
    def _clean_and_normalize_response(self, response: str) -> str:
        """清理andspecification化LLMresponse"""
        # 移除CodeBlock标记
        response = re.sub(r'```\w*\n?', '', response)
        response = re.sub(r'```', '', response)

        # 移除Note:后英文解释
        response = re.sub(r'Note:.*?(?=\n|$)', '', response, flags=re.DOTALL | re.IGNORECASE)

        # 移除Markdown标题and多余换行符
        response = re.sub(r'##\s*evaluationreport\s*', '', response)
        response = re.sub(r'###\s*提交result\s*', '', response)
        response = re.sub(r'\n{2,}', '\n', response)

        # 繁体转简体常见映射
        traditional_to_simplified = {
            '基础情報complete性': '基础Informationcomplete性',
            '基thiscomplete度': '基础Informationcomplete性',
            '疗状描述完備性': 'symptom描述充分性',
            'symptom描述complete度': 'symptom描述充分性',
            '逻智一致性評估': '逻辑一致性',
            '逻辑一致性': '逻辑一致性',
            '总得分': 'total_score',
            '評分': 'quality_level',
            '基礎': '基础', 'completeity': 'complete性', '症狀': 'symptom', '充分ity': '充分性',
            'logic': '逻辑', '總體': 'Population', '評分': '评分', '質量': '质量',
            'etc.級': 'etc.级', '決策': 'decision', '澄清問題': '澄清question'
        }

        for traditional, simplified in traditional_to_simplified.items():
            response = response.replace(traditional, simplified)

        return response.strip()

    def _parse_clarification_questions(self, response: str) -> List[str]:
        """
        Parsing 澄清question，support多格式：
        1. 直接StringList格式：["question1", "question2"]
        2. JSONObject格式：{"questions": ["question1", "question2"]}
        3. 备用Parsing ：提取问号结尾行
        """
        try:
            import json
            
            # Method1：尝试直接Parsing StringList格式 ["question1", "question2"]
            list_pattern = r'\[([^\]]*)\]'
            list_matches = re.findall(list_pattern, response)
            
            for match in list_matches:
                try:
                    # 尝试Parsing asJSONList
                    json_str = f'[{match}]'
                    questions = json.loads(json_str)
                    if isinstance(questions, list) and questions:
                        # 过滤validquestion
                        valid_questions = []
                        for q in questions:
                            if isinstance(q, str) and q.strip():
                                valid_questions.append(q.strip())
                        if valid_questions:
                            return valid_questions[:2]  # Limit最多2question
                except json.JSONDecodeError:
                    continue
            
            # Method2：尝试Parsing JSONObject格式 {"questions": [...]}
            json_candidates = []
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
            
            if thinking_match:
                text_after_thinking = response[thinking_match.end():]
                json_match = re.search(r'\{[^}]*"questions"\s*:\s*\[[^\]]*\][^}]*\}', text_after_thinking)
                if json_match:
                    json_candidates.append(json_match.group())
            
            if not json_candidates:
                json_matches = re.findall(r'\{[^}]*"questions"\s*:\s*\[[^\]]*\][^}]*\}', response)
                json_candidates.extend(json_matches)
            
            for json_str in json_candidates:
                try:
                    data = json.loads(json_str)
                    questions = data.get('questions', [])
                    if questions:
                        return questions[:2]  # Limit最多2question
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.logger.debug(f"JSONParsing Failed : {str(e)}")
        
        # Method3：备用Parsing ：查找问号结尾行
        questions = []
        for line in response.splitlines():
            line = line.strip()
            # 移除可能引号and编号前缀
            line = re.sub(r'^["\']|["\']$', '', line)  # 移除首尾引号
            line = re.sub(r'^\d+\.?\s*', '', line)    # 移除编号前缀
            
            if (line and 
                (line.endswith('?') or line.endswith('？')) and
                len(line) >= 5 and len(line) <= 200):
                questions.append(line)
        
        return questions[:2]

    def _clean_json_string(self, json_str: str) -> str:
        """清理JSONString常见incorrect"""
        # 移除多余逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 修复引号question
        json_str = re.sub(r'([{,]\s*)"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:', r'\1"\2":', json_str)
        
        # 移除字Segment名多余空格并用下划线替换
        json_str = re.sub(r'"([^"]*)\s+([^"]*)":', r'"\1_\2":', json_str)
        
        # 修复常见字Segment名incorrect
        field_corrections = {
            '"basic_compleness"': '"basic_completeness"',
            '"symptom_adecuacy"': '"symptom_adequacy"',
            '"logical_coherency"': '"logical_coherence"',
            '"clarification_question"': '"clarification_questions"',
            '"Total_Score"': '"total_score"',
            '"Quality_Level"': '"quality_level"',
            '"Decision"': '"decision"',
            '"_Clarification_Questions"': '"clarification_questions"'
        }
        
        for wrong, correct in field_corrections.items():
            json_str = json_str.replace(wrong, correct)
        
        # 修复值英文Content
        value_corrections = {
            '"good"': '"良好"',
            '"excellent"': '"优秀"',
            '"Excellent"': '"优秀"',
            '"directly_transfer_to_expert"': '"直接转expert"',
            '"need_clarification"': '"需要澄清"'
        }
        
        for wrong, correct in value_corrections.items():
            json_str = json_str.replace(wrong, correct)
        
        return json_str

    def _extract_score_field(self, data: dict, field_names: list) -> int:
        """fromData提取score字Segment，support多字Segment名"""
        for field_name in field_names:
            if field_name in data:
                value = data[field_name]
                if isinstance(value, dict) and 'score' in value:
                    return int(value['score'])
                elif isinstance(value, (int, float)):
                    return int(value)
        return 0

    def _extract_quality_level(self, summary: dict) -> str:
        """提取质量etc.级"""
        quality_fields = ['quality_level', 'Quality_Level']
        for field in quality_fields:
            if field in summary:
                level = summary[field]
                if level in ['优秀', '良好', '一般', '较差', '很差']:
                    return level
                # 英文to文映射
                level_mapping = {
                    'excellent': '优秀',
                    'good': '良好', 
                    'average': '一般',
                    'poor': '较差',
                    'very_poor': '很差'
                }
                return level_mapping.get(level.lower(), '一般')
        return '一般'

    def _extract_decision(self, summary: dict) -> str:
        """提取decision"""
        decision_fields = ['decision', 'Decision']
        for field in summary:
            if field in summary:
                decision = summary[field]
                if decision in ['直接转expert', '需要澄清']:
                    return decision
                # 英文to文映射
                decision_mapping = {
                    'directly_transfer_to_expert': '直接转expert',
                    'direct_transfer': '直接转expert',
                    'need_clarification': '需要澄清',
                    'clarification_needed': '需要澄清'
                }
                return decision_mapping.get(decision.lower(), '需要澄清')
        return '需要澄清'

    def _extract_clarification_questions(self, summary: dict) -> list:
        """提取澄清question"""
        question_fields = ['clarification_questions', '_Clarification_Questions', 'clarification_question']
        for field in question_fields:
            if field in summary:
                questions = summary[field]
                if isinstance(questions, list):
                    return questions
                elif isinstance(questions, str) and questions.strip():
                    return [questions.strip()]
        return []


    
    def _extract_score_from_variants(self, parsed_json: dict, field_variants: list) -> int:
        """fromJSON提取score，尝试多字Segment名变体"""
        for field_name in field_variants:
            if field_name in parsed_json:
                try:
                    score = int(parsed_json[field_name])
                    if 0 <= score <= 10:  # validationscoreRange
                        return score
                except (ValueError, TypeError):
                    continue
        return 0

    def _extract_scores_with_regex(self, response: str) -> Dict[str, Any]:
        """
        使用RegularizationExpression直接提取五Dimensionscore
        """
        try:
            # 定义score提取mode
            patterns = {
                'basic_score': r'"basic_score":\s*(\d+)',
                'symptom_score': r'"symptom_score":\s*(\d+)',
                'exam_score': r'"exam_score":\s*(\d+)',
                'timeline_score': r'"timeline_score":\s*(\d+)',
                'logic_score': r'"logic_score":\s*(\d+)',
                'total_score': r'"total_score":\s*(\d+)'
            }
            
            scores = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response)
                scores[key] = int(match.group(1)) if match else 0
            
            # 如果没has找tototal_score，Computing 总分
            if scores['total_score'] == 0:
                scores['total_score'] = sum([scores['basic_score'], scores['symptom_score'], 
                                           scores['exam_score'], scores['timeline_score'], scores['logic_score']])
            
            return {
                'basic_score': scores['basic_score'],
                'symptom_score': scores['symptom_score'],
                'exam_score': scores['exam_score'],
                'timeline_score': scores['timeline_score'],
                'logic_score': scores['logic_score'],
                'total_score': scores['total_score'],
                'needs_clarification': scores['total_score'] < self.clarification_threshold,
                'decision': 'needs_clarification' if scores['total_score'] < self.clarification_threshold else 'sufficient'
            }
            
        except Exception as e:
            self.logger.error(f"RegularizationExpression提取scoreFailed : {str(e)}")
            # returnDefault值
            return {
                'basic_score': 0,
                'symptom_score': 0,
                'exam_score': 0,
                'timeline_score': 0,
                'logic_score': 0,
                'total_score': 0,
                'needs_clarification': True,
                'decision': 'needs_clarification'
            }






    def _determine_quality_level_from_score(self, score: int) -> str:
        """根据score确定质量etc.级"""
        if score >= 26:
            return '优秀'
        elif score >= 21:
            return '良好'
        elif score >= 16:
            return '一般'
        elif score >= 11:
            return '较差'
        else:
            return '很差'
    
    def _determine_decision_from_score(self, score: int) -> str:
        """根据score确定decision"""
        if score >= 16:
            return '直接转expert'
        else:
            return '需要澄清'
    
    def _get_timestamp(self) -> str:
        """Getting CurrentTime戳"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _normalize_string(self, text: str) -> str:
        """specification化String，用于Weightmatching"""
        if not text:
            return ""
        # 移除空格、标点符号，转换as小写
        import re
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower().strip())
        return normalized
    
    def _find_expert_weight(self, expert_name: str, expert_specialty: str, expert_weights: Dict[str, float]) -> Tuple[float, str]:
        """
        specification化WeightmatchingFunction，support多matchingstrategy
        
        Args:
            expert_name: expertname
            expert_specialty: expertspecialty
            expert_weights: WeightDictionary
            
        Returns:
            Tuple[Weight值, matchingMethod描述]
        """
        if not expert_weights:
            return 0.0, "WeightDictionaryas空"
        
        # strategy1: 精确matching - 文名
        if expert_name in expert_weights:
            return expert_weights[expert_name], f"精确matchingexpertname: {expert_name}"
        
        # strategy2: 精确matching - 英文名/specialty名
        if expert_specialty in expert_weights:
            return expert_weights[expert_specialty], f"精确matchingspecialtyname: {expert_specialty}"
        
        # strategy3: specification化Stringmatching
        normalized_expert_name = self._normalize_string(expert_name)
        normalized_expert_specialty = self._normalize_string(expert_specialty)
        
        for weight_key, weight_value in expert_weights.items():
            normalized_key = self._normalize_string(weight_key)
            
            # 完全matchingspecification化String
            if normalized_key and (normalized_key == normalized_expert_name or 
                                 normalized_key == normalized_expert_specialty):
                return weight_value, f"specification化Stringmatching: {weight_key}"
        
        # strategy4: Local包含matching
        for weight_key, weight_value in expert_weights.items():
            normalized_key = self._normalize_string(weight_key)
            
            # Checking is否包含expertnameorspecialty（双to包含）
            if normalized_key and normalized_expert_name and (
                normalized_expert_name in normalized_key or 
                normalized_key in normalized_expert_name
            ):
                return weight_value, f"Local包含matching(expert名): {weight_key}"
            
            if normalized_key and normalized_expert_specialty and (
                normalized_expert_specialty in normalized_key or 
                normalized_key in normalized_expert_specialty
            ):
                return weight_value, f"Local包含matching(specialty名): {weight_key}"
        
        # strategy5: Keyword matching（针for复合specialty名）
        # 提取关键词进行matching
        expert_keywords = set()
        if expert_name:
            expert_keywords.update(self._normalize_string(expert_name).split())
        if expert_specialty:
            expert_keywords.update(self._normalize_string(expert_specialty).split())
        
        for weight_key, weight_value in expert_weights.items():
            key_keywords = set(self._normalize_string(weight_key).split())
            # 如果has共同关键词且关键词Length大于1
            common_keywords = expert_keywords.intersection(key_keywords)
            if common_keywords and any(len(kw) > 1 for kw in common_keywords):
                return weight_value, f"Keyword matching: {weight_key} (共同关键词: {', '.join(common_keywords)})"
        
        # allmatchingstrategy都Failed 
        return 0.0, "allmatchingstrategy都Failed "
    
    def _build_reasoning_path(self, doctor_opinions: List[Dict[str, Any]], 
                             expert_weights: Dict[str, float], 
                             option_scores: Dict[str, float]) -> str:
        """构建inferencepath"""
        reasoning = []
        reasoning.append("expert意见汇总:")
        
        for opinion in doctor_opinions:
            expert_name = opinion.get('expert_name', '')
            expert_choice = opinion.get('choice', '')
            expert_confidence = opinion.get('confidence', 0.0)
            weight = expert_weights.get(expert_name, 0.0)
            
            reasoning.append(f"- {expert_name}: 选择 {expert_choice}, confidence {expert_confidence:.2f}, Weight {weight:.2f}")
        
        reasoning.append("\n加权得分:")
        for option, score in option_scores.items():
            reasoning.append(f"- Option {option}: {score:.3f}")
        
        return '\n'.join(reasoning)