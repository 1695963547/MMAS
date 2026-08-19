#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMAS Patient Agent
Implemented based on the MMAS three-agent pipeline

Features:
1. Step 1: Patient agent initial question (Patient Agent → Evaluator Agent)
2. Step 4: Patient agent clarification response (Patient Agent → Evaluator Agent)
"""

import json
import time
import logging
import re
import torch
from typing import Dict, Any, List, Optional
from unified_expert_template_manager import UnifiedExpertTemplateManager
from global_model_manager import GlobalModelManager

class MMASPatientAgent:
    """
    MMAS Patient Agent
    
    Responsibilities：
    - Initial question：Format raw case data into standardized medical multiple-choice question
    - Answer clarification questions by role-playing as patient based on patient_facts
    """
    
    def __init__(self, api_client=None, use_api_model=False, api_model_name=None):
        """Initialize patient agent"""
        self.logger = logging.getLogger(__name__)
        self.template_manager = UnifiedExpertTemplateManager()
        self.model_manager = GlobalModelManager()
        self.api_client = api_client
        self.use_api_model = use_api_model
        self.api_model_name = api_model_name
        self.llm_client = api_client  # Add llm_client attribute
        self._cached_eos_token_ids = None  # Cache EOS token IDs to avoid repeated computation
        
    def step1_initial_question(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 1: Patient agent initial question
        Combine context, question, options fields from dataset into standardized medical MCQ format
        
        Args:
            case_data: Case data containing context, question, options, patient fields
            
        Returns:
            格式化standardmedical选择题and相关Information
        """
        try:
            # 提取基thisInformation
            context = case_data.get('context', [])
            question = case_data.get('question', '')
            options = case_data.get('options', {})
            patient_info = case_data.get('patient', {})
            
            # 格式化case描述 - 只使用context[0]
            if isinstance(context, list) and len(context) > 0:
                context_str = context[0]
            else:
                context_str = str(context)
            
            # 格式化选择题Option
            options_str = ""
            for key, value in options.items():
                options_str += f"{key}. {value}\n"
            
            # 合成standardmedical选择题格式
            formatted_question = f"""【case描述】
{context_str}

【clinicalquestion】
{question}

【选择题Option】
{options_str.strip()}

请选择最佳答案："""

            # 构建returnresult
            result = {
                "case_id": case_data.get("id", "unknown"),
                "formatted_question": formatted_question,
                "status": "success",
                "agent": "patient_agent",
                'original_context': context,
                'original_question': question,
                'options': options,
                'patient_info': patient_info,
                'case_id': case_data.get('id', 'unknown'),
                'step': 'step1_initial_question',
                'agent': 'patient_agent',
                'timestamp': self._get_timestamp()
            }
            
            self.logger.info(f"Patient Agent - Step 1: 格式化case {case_data.get('id', 'unknown')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Patient Agent Step 1 incorrect: {str(e)}")
            return {
                'error': str(e),
                'step': 'step1_initial_question',
                'agent': 'patient_agent',
                'timestamp': self._get_timestamp()
            }
    
    def step4_clarification_response(self, clarification_questions: List[str], case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        step四：patient澄清answer
        based oncontextandfacts严格answer澄清question
        """
        try:
            # validationInput
            if not clarification_questions:
                self.logger.warning("step四：没has澄清question")
                return {
                    'success': False,
                    'error': '没has澄清question',
                    'information_completeness': 0.0,
                    'clarification_context': []
                }
            
            # Getting template
            template = self.template_manager.get_patient_template(
                'step4_patient_response',
                clarification_questions=clarification_questions,
                case_data=case_data
            )
            
            # 不Recordstep四InputPrompttolog，仅保留Outputlog

            self.logger.info(f"step四：Starting answer {len(clarification_questions)} 澄清question")
            
            # Initializing token计数（APIandthis地模型共用）
            input_tokens = 0
            generated_tokens = 0
            
            # 调用LLMGenerating answer
            if self.use_api_model and self.llm_client and self.api_model_name:
                # Using API model
                response = self.llm_client.call_llm(
                    messages=[{"role": "user", "content": template}],
                    temperature=0.2,
                    max_tokens=1024,
                    model_name=self.api_model_name
                )
            else:
                # 使用this地模型
                model, tokenizer = self.model_manager.get_model('main_llm')
                messages = [{"role": "user", "content": template}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
                
                with torch.no_grad():
                    start_ts = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.2,
                        do_sample=False,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=self._get_safe_eos_token_ids(tokenizer)
                    )
                    elapsed = time.time() - start_ts
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                input_tokens = 0
                generated_tokens = 0
                try:
                    input_tokens = int(inputs.input_ids.shape[-1])
                    generated_tokens = int(outputs[0].shape[-1] - input_tokens)
                    self.logger.info(f"step四Generating statistics - Inputtokens: {input_tokens}, Outputtokens: {generated_tokens}, 耗时: {elapsed:.2f}s")
                except Exception:
                    pass
            
            if not response:
                self.logger.error("step四：LLMresponseas空")
                return {
                    'success': False,
                    'error': 'LLMresponseas空',
                    'information_completeness': 0.0,
                    'clarification_context': [],
                    'total_tokens': input_tokens + generated_tokens
                }
            
            # Parsing patientanswer
            patient_answers = self._parse_patient_answers(response)
            
            # 清理andvalidationeachanswer，确保严格based on事实
            cleaned_answers = []
            for answer in patient_answers:
                cleaned_answer = self._clean_and_validate_answer(answer, case_data)
                cleaned_answers.append(cleaned_answer)
            
            # 如果Parsing Failed or清理后没hasvalidanswer，provideDefaultanswer
            if not cleaned_answers and clarification_questions:
                self.logger.warning("step四：Parsing patientanswerFailed ，使用Defaultanswer")
                cleaned_answers = ["Information缺失，不清楚"] * len(clarification_questions)
            
            # 确保answer数量andquestion数量matching
            while len(cleaned_answers) < len(clarification_questions):
                cleaned_answers.append("Information缺失，不清楚")
            
            # 构建澄清Context
            clarification_context = []
            for i, (question, answer) in enumerate(zip(clarification_questions, cleaned_answers)):
                clarification_context.append({
                    'question': question,
                    'answer': answer,
                    'question_id': i + 1
                })
            
            # Computing Informationcomplete度
            fact_based_answers = sum(1 for answer in cleaned_answers if self._is_fact_based_answer(answer))
            total_questions = len(clarification_questions)
            information_completeness = fact_based_answers / total_questions if total_questions > 0 else 0.0
            
            self.logger.info(f"step四complete：Informationcomplete度={information_completeness:.2f}, question数={total_questions}, based on事实answer={fact_based_answers}")
            
            return {
                'success': True,
                'patient_answers': cleaned_answers,
                'information_completeness': information_completeness,
                'clarification_context': clarification_context,
                'total_questions': total_questions,
                'fact_based_answers': fact_based_answers,
                'raw_response': response,
                'input_prompt': template,
                'total_tokens': input_tokens + generated_tokens
            }
            
        except Exception as e:
            self.logger.error(f"step四Executing Failed : {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'information_completeness': 0.0,
                'clarification_context': []
            }
    
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

    def _get_timestamp(self) -> str:
        """Getting CurrentTime戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def validate_case_data(self, case_data: Dict[str, Any]) -> bool:
        """
        validationcaseDatacomplete性
        
        Args:
            case_data: caseData
            
        Returns:
            is否通过validation
        """
        required_fields = ['context', 'question', 'options']
        
        for field in required_fields:
            if field not in case_data:
                self.logger.warning(f"缺少Required字Segment: {field}")
                return False
        
        # validationoptions格式
        options = case_data.get('options', {})
        if not isinstance(options, dict) or len(options) == 0:
            self.logger.warning("Option格式incorrectoras空")
            return False
        
        return True
    
    def get_patient_basic_info(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取patient基thisInformation
        
        Args:
            case_data: caseData
            
        Returns:
            patient基thisInformation
        """
        patient_info = case_data.get('patient', {})
        
        return {
            'age': patient_info.get('age', 'unknown'),
            'gender': patient_info.get('gender', 'unknown'),
            'specialties': patient_info.get('specialties', []),
            'subspecialties': patient_info.get('subspecialties', []),
            'gpt_specialty': patient_info.get('gpt_specialty', 'unknown')
        }
    
    def _clean_and_validate_answer(self, answer: str, case_data: Dict[str, Any]) -> str:
        """
        清理answer，不做事实校验；直接return模型Output
        
        Args:
            answer: Originalanswer
            case_data: caseData
            
        Returns:
            清理后answer；仅明显空orinvalid时return“Information缺失，不清楚”
        """
        if not answer or len(answer.strip()) < 2:
            return "Information缺失，不清楚"
        
        # 保留基this清理，但不再进行事实matching校验orForce 替换
        cleaned_answer = self._clean_answer(answer)
        return cleaned_answer
    
    def _is_answer_fact_based(self, answer: str, available_facts: List[str], context: List[str]) -> bool:
        """
        Checking answeris否based onavailable事实andContext
        
        Args:
            answer: answerContent
            available_facts: available事实List
            context: ContextList
            
        Returns:
            is否based on事实
        """
        if not answer or len(answer.strip()) < 3:
            return False
        
        answer_lower = answer.lower().strip()
        
        # 如果isstandard"Information缺失"answer，认asisbased on事实（表示诚实answer）
        if any(phrase in answer_lower for phrase in ['Information缺失', '不清楚', '不知道', 'no法确定', '没has相关Information']):
            return True
        
        # Checking answeris否包含事实Information
        all_available_info = available_facts + context
        if not all_available_info:
            # 如果没hasavailableInformation，只has"Information缺失"typeanswer才isbased on事实
            return False
        
        # Checking answeris否包含availableInformation关键Content
        for info in all_available_info:
            if not info:
                continue
            info_lower = info.lower()
            
            # 提取关键词进行matching
            info_keywords = self._extract_keywords(info_lower)
            answer_keywords = self._extract_keywords(answer_lower)
            
            # 如果hasKeyword matching，认asisbased on事实
            if any(keyword in answer_keywords for keyword in info_keywords if len(keyword) > 2):
                return True
        
        # Checking is否包含medical相关具体Information（数值、Time、部etc.）
        if self._contains_specific_medical_info(answer):
            return True
        
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取Text关键词
        
        Args:
            text: InputText
            
        Returns:
            关键词List
        """
        # 移除标点符号，分割成词
        import re
        words = re.findall(r'[\u4e00-\u9fff]+|\w+', text)
        # 过滤掉过短词and常见停用词
        stop_words = {'', '', '', 'is', 'has', 'and', 'and', 'or', '但', '而', '也', '都', '很', '更', '最', '这', 'that', '一', '二', '三'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        return keywords
    
    def _contains_specific_medical_info(self, answer: str) -> bool:
        """
        Checking answeris否包含具体medicalInformation
        
        Args:
            answer: answerContent
            
        Returns:
            is否包含具体medicalInformation
        """
        import re
        
        # Checking 数值Information
        if re.search(r'\d+\.?\d*\s*[a-zA-Z%/]+', answer):  # 如：8.5g/dL, 37.5°C, 120/80mmHg
            return True
        
        # Checking TimeInformation
        if re.search(r'\d+\s*[年月日天小时分钟]', answer):  # 如：3天前, 2小时
            return True
        
        # Checking medical术语and部
        medical_terms = [
            '血压', '心率', '体温', '血糖', '血红蛋白', '白细胞', '血小板',
            '肝functionality', '肾functionality', '心电graph', 'CT', 'MRI', 'X线',
            '头部', '胸部', '腹部', '四肢', '心脏', '肺部', '肝脏', '肾脏',
            'pain', 'fever', '咳嗽', '呼吸困难', 'nausea', 'vomiting', 'diarrhea'
        ]
        
        for term in medical_terms:
            if term in answer:
                return True
        
        return False
    
    def _parse_patient_answers(self, response: str) -> List[str]:
        """
        Parsing patient澄清answer，优先support严格List格式，并兼容HistoryJSON/编号格式。
        return清洗后StringList；Parsing Failed return空List，让调用方走Defaultanswer。
        """
        try:
            cleaned = self._clean_response(response)

            # 1) 优先Parsing 严格List格式 ["答案1","答案2"]
            import json, re
            list_match = re.search(r'\[\s*(?:"[^"]*"\s*,?\s*)+\]', cleaned, re.S)
            if list_match:
                try:
                    arr = json.loads(list_match.group(0))
                    if isinstance(arr, list) and arr:
                        return [self._clean_answer(str(a)) for a in arr if str(a).strip()]
                except json.JSONDecodeError:
                    pass

            # 2) 兼容JSONObject：answers or 澄清answer（History格式）
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    # 英文键
                    if 'answers' in data and isinstance(data['answers'], list):
                        return [self._clean_answer(str(a)) for a in data['answers'] if str(a).strip()]
                    # 文键structure：[{question, answer, ...}]
                    if '澄清answer' in data and isinstance(data['澄清answer'], list):
                        answers = []
                        for item in data['澄清answer']:
                            if isinstance(item, dict) and 'answer' in item:
                                answers.append(self._clean_answer(str(item['answer'])))
                            elif isinstance(item, str) and item.strip():
                                answers.append(self._clean_answer(item))
                        if answers:
                            return answers
            except Exception:
                pass

            # 3) 编号制后备Parsing ：1. 答案 or question1: 答案
            answers = []
            for line in cleaned.splitlines():
                m = re.match(r'^\s*(?:question)?\s*\d+\s*[.:：]\s*(.+)$', line)
                if m:
                    answers.append(self._clean_answer(m.group(1)))
            if answers:
                return answers

            # 4) 引号围绕Content作as最后后备
            quoted = re.findall(r'"([^"]+)"', cleaned)
            if quoted:
                return [self._clean_answer(s) for s in quoted if len(s.strip()) > 2]

        except Exception as e:
            self.logger.debug(f"Parsing patientanswerFailed : {str(e)}")

        # Parsing Failed  → return空List，让上Layer走Default“Information缺失，不清楚”
        return []
    
    def _clean_response(self, response: str) -> str:
        """清理LLMresponse，移除CodeBlockandno关Content"""
        # 移除CodeBlock
        response = re.sub(r'```[\s\S]*?```', '', response)
        # 移除明显Code行（但保留正常JSONandTextContent）
        response = re.sub(r'.*\.append\(.*\).*', '', response)
        # 简单清理空行and多余空格
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # 只保留非空行
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def _clean_answer(self, answer: str) -> str:
        """清理Singleanswer"""
        # 移除前缀标记
        answer = re.sub(r'^(question\d+answer[:：\s]*|answer\d*[:：\s]*|\d+\.?\s*[:：\s]*)', '', answer).strip()
        # 移除尾部no关Content
        answer = re.sub(r'(assistant|Assistant).*$', '', answer, flags=re.IGNORECASE).strip()
        # 移除多余标点
        answer = re.sub(r'^[。，、：；]+|[。，、：；]+$', '', answer).strip()
        return answer
    
    def _is_fact_based_answer(self, answer: str) -> bool:
        """
        判断answeris否based on事实
        改进Version：更宽松地识别based on事实answer
        """
        if not answer or len(answer.strip()) < 3:
            return False
        
        answer_lower = answer.lower().strip()
        
        # 明确否定answer
        negative_indicators = [
            '不知道', '不清楚', '不解', '不记得', '没has', 'Information缺失',
            '不太清楚', '不确定', 'no法确定', '不明确'
        ]
        
        # 如果包含否定metric，Checking is否同时包含具体Information
        has_negative = any(neg in answer_lower for neg in negative_indicators)
        
        # 积极事实metric（更宽松）
        positive_indicators = [
            '根据', 'medical record', 'Checking ', 'result', '显示', '提to', 'Record',
            'doctor', 'diagnosis', 'symptom', 'treatment', 'medication', '数值', 'metric',
            '发现', '存', '出现', '具体', '明确', '确实', 'is',
            'has', 'as', '约', '大概', '左右', 'Time', '日期'
        ]
        
        has_positive = any(pos in answer_lower for pos in positive_indicators)
        
        # Checking is否包含Number、日期etc.具体Information
        has_specific_info = bool(re.search(r'\d+', answer)) or bool(re.search(r'[年月日]', answer))
        
        # 如果has否定metric但同时has积极metricor具体Information，仍然认asisbased on事实
        if has_negative and (has_positive or has_specific_info):
            return True
        
        # 如果没has否定metric，且has积极metricor具体Information，or者answer足够长且不is简单answer
        if not has_negative:
            if has_positive or has_specific_info:
                return True
            # for于较长answer，如果不包含明显否定词，也认asisbased on事实
            if len(answer.strip()) > 10 and not any(simple in answer_lower for simple in ['好', 'is', '嗯', '哦']):
                return True
        
        return False