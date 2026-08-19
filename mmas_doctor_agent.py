import time
import json
import re
import torch
import logging
import os
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from sentence_transformers import util
from global_model_manager import get_global_model_manager
from unified_expert_template_manager import get_unified_expert_template_manager
from api_model_client import get_api_client

class MMASDoctorAgent:
    def __init__(self, model_manager, template_manager, api_client=None, use_api_model=False, api_model_name=None, use_semantic_fallback=False):
        """
        Initialize MMAS Doctor Agent
        
        Args:
            model_manager: Global model manager
            template_manager: Unified expert template manager
            api_client: API client (optional)
            use_api_model: Whether to use API model
            api_model_name: API model name
            use_semantic_fallback: Whether to use semantic fallback for intent classification
        """
        self.model_manager = model_manager
        self.template_manager = template_manager
        self.api_client = api_client
        self.use_api_model = use_api_model
        self.api_model_name = api_model_name
        self.use_semantic_fallback = use_semantic_fallback
        self.logger = logging.getLogger(__name__)
        self._cached_eos_token_ids = None
        
        # Load intent keywords
        intent_keywords_path = "intent_keywords.json"
        if os.path.exists(intent_keywords_path):
            with open(intent_keywords_path, 'r', encoding='utf-8') as f:
                self.intent_keywords = json.load(f)
        else:
            # Default intent keywords
            self.intent_keywords = {
                "diagnosis": ["diagnosis", "condition", "disease", "symptom", "Checking ", "treatment"],
                "medication": ["medication", "medication", "medication", "dosage", "side effect"],
                "lifestyle": ["lifestyle", "diet", "exercise", "sleep schedule", "habits"],
                "follow_up": ["follow-up visit", "follow-up", "monitoring", "observation"]
            }
        
        # If using semantic fallback, load intent prototype vectors
        if self.use_semantic_fallback:
            self.intent_vectors = self._load_intent_vectors()

    def _load_intent_vectors(self):
        """Load precomputed intent prototype vectors"""
        vectors_path = 'intent_prototype_vectors.pkl'
        if not os.path.exists(vectors_path):
            self.logger.error(f"意graph原型VectorfileNot found: {vectors_path}")
            return None
        try:
            with open(vectors_path, 'rb') as f:
                intent_vectors = pickle.load(f)
            self.logger.info(f"Successfully Loading 意graph原型Vector: {list(intent_vectors.keys())}")
            return intent_vectors
        except Exception as e:
            self.logger.error(f"Loading 意graph原型VectorFailed : {e}")
            return None

    def _extract_thinking_content(self, response_text: str) -> str:
        """fromresponseText提取thinkingContent"""
        thinking_match = re.search(r"<thinking>(.*?)(?:</think(?:ing)?>|<json>)", response_text, re.DOTALL)
        return thinking_match.group(1).strip() if thinking_match else ""

    def _classify_question_intent(self, question, options=None, context=None):
        """
        两stage意graphclassificationstrategy：
        第一stage：强触发器（Mechanism/Tests优先识别）
        第二stage：如果未触发，则根据Configuring 选择Keyword matchingorSemantic similarity computation
        """
        if not question:
            self.logger.debug("Empty question, defaulting to 'Diagnosis'")
            return "Diagnosis"

        question_lower = question.lower()

        # 第一stage：强触发器
        strong_triggers = {
            "Mechanism": ["mechanism of action", "moa", "作用机制"],
            "Tests": ["confirmatory test", "which test", "best initial test"]
        }
        for intent, triggers in strong_triggers.items():
            for trigger in triggers:
                if trigger.lower() in question_lower:
                    self.logger.info(f"Strong trigger detected: Question classified as '{intent}' based on trigger '{trigger}'")
                    return intent

        # 第二stage：根据Configuring 选择strategy
        if self.use_semantic_fallback and self.intent_vectors:
            # 使用语义similarity进行classification
            try:
                semantic_model = self.model_manager.get_model('semantic')
                if not semantic_model:
                    self.logger.error("no法Getting 语义模型，回退toKeyword matching")
                    return self._classify_by_keywords(question_lower)

                question_embedding = semantic_model.encode(question, convert_to_tensor=True)
                
                intent_names = list(self.intent_vectors.keys())
                prototype_vectors = torch.stack([torch.from_numpy(self.intent_vectors[name]) for name in intent_names])

                cos_scores = util.pytorch_cos_sim(question_embedding, prototype_vectors)[0]
                best_match_idx = torch.argmax(cos_scores).item()
                
                # Setting 一Threshold，低于theThreshold则认as不matching
                if cos_scores[best_match_idx] < 0.3:  # 可调Threshold
                    self.logger.info(f"Semantic similarity score ({cos_scores[best_match_idx]:.4f}) is below threshold, defaulting to 'Diagnosis'")
                    return "Diagnosis"

                best_intent = intent_names[best_match_idx]
                self.logger.info(f"Semantic classification: Question classified as '{best_intent}' with similarity score {cos_scores[best_match_idx]:.4f}")
                return best_intent

            except Exception as e:
                self.logger.error(f"语义classificationFailed : {e}，回退toKeyword matching")
                return self._classify_by_keywords(question_lower)
        else:
            # 使用Keyword matching
            return self._classify_by_keywords(question_lower)

    def _classify_by_keywords(self, question_lower):
        """based on关键词意graphclassification"""
        if not self.intent_keywords:
            self.logger.debug("No intent keywords, defaulting to 'Diagnosis'")
            return "Diagnosis"

        fallback_priority_order = ["Diagnosis", "Management", "Mechanism", "Tests", "Knowledge", "Ethics", "Prevention"]
        
        matched_intents = []
        for intent in fallback_priority_order:
            if intent in self.intent_keywords:
                keywords = self.intent_keywords[intent]
                for keyword in keywords:
                    if keyword.lower() in question_lower:
                        matched_intents.append((intent, keyword))
                        break
        
        if matched_intents:
            selected_intent, matched_keyword = matched_intents[0]
            self.logger.info(f"Keyword classification: Question classified as '{selected_intent}' based on keyword '{matched_keyword}'")
            return selected_intent
        
        self.logger.info("No keywords matched, defaulting to 'Diagnosis'")
        return "Diagnosis"

    def _get_safe_eos_token_ids(self, tokenizer):
        """
        安全Getting 并cacheeos_token_idList，适配不同模型（如Qwen<|im_end|>）。
        避免None值andDuplicateComputing 导致question。
        """
        if self._cached_eos_token_ids is not None:
            return self._cached_eos_token_ids

        eos_token_ids = {tokenizer.eos_token_id}  # 使用Set避免Duplicate
        special_tokens_to_check = ["<|eot_id|>", "<|im_end|>"]

        for token in special_tokens_to_check:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and isinstance(token_id, int) and token_id > 0:
                    if token_id not in eos_token_ids:
                        eos_token_ids.add(token_id)
                        self.logger.info(f"Successfully 添加特殊EOS token '{token}' (ID: {token_id})")
                else:
                    # 这logetc.级可with调低，因as大partial模型只会matching一
                    self.logger.debug(f"特殊EOS token '{token}' IDinvalid: {token_id}，Skipping 添加")
            except Exception as e:
                self.logger.warning(f"Getting 特殊EOS token '{token}' IDFailed : {e}")

        final_ids = list(eos_token_ids)
        self.logger.info(f"最终确定EOS token IDs: {final_ids}")
        self._cached_eos_token_ids = final_ids
        return final_ids

    def _get_timestamp(self):
        return time.strftime("%Y%m%d_%H%M%S")

    def _clean_and_fix_json_string(self, json_str: str) -> str:
        """
        清理and修复JSONString常见question
        """
        if not json_str:
            return json_str
            
        # 替换常见invalid值as0.0
        replacements = {
            ': N/A': ': 0.0',
            ': null': ': 0.0', 
            ': undefined': ': 0.0',
            ': NaN': ': 0.0',
            ': None': ': 0.0',
            ':"N/A"': ': 0.0',
            ':"null"': ': 0.0',
            ':"undefined"': ': 0.0',
            ':"NaN"': ': 0.0',
            ':"None"': ': 0.0'
        }
        
        cleaned_json = json_str
        for old_val, new_val in replacements.items():
            cleaned_json = cleaned_json.replace(old_val, new_val)
            
        # 移除多余逗号
        cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
        cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
        
        return cleaned_json

    def _extract_json_from_response(self, response: str) -> str:
        """
        from模型response提取JSONString。
        OptimizationVersion：如果找不tovalidJSONBlock，则直接fromresponseTextParsing score作as后备。
        """
        # 1. 优先matchingby```json ...```包围JSON
        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            json_str = match.group(1)
            cleaned_json = self._clean_and_fix_json_string(json_str)
            try:
                json.loads(cleaned_json)
                self.logger.debug("Successfully extracted JSON from markdown block")
                return cleaned_json
            except json.JSONDecodeError:
                self.logger.warning(f"Found JSON markdown block but content is invalid: {cleaned_json}")

        # 2. 查找<json>标签包围JSON
        json_tag_match = re.search(r'<json>\s*(\{.*?\})\s*</json>', response, re.DOTALL)
        if json_tag_match:
            json_str = json_tag_match.group(1)
            cleaned_json = self._clean_and_fix_json_string(json_str)
            try:
                json.loads(cleaned_json)
                self.logger.debug("Successfully extracted JSON from <json> tags")
                return cleaned_json
            except json.JSONDecodeError:
                self.logger.warning(f"Found JSON in tags but content is invalid: {cleaned_json}")

        # 3. 手动查找第一completeJSONObject
        start_index = response.find('{')
        if start_index != -1:
            brace_count = 1
            for i in range(start_index + 1, len(response)):
                char = response[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                
                if brace_count == 0:
                    end_index = i + 1
                    json_str = response[start_index:end_index]
                    cleaned_json = self._clean_and_fix_json_string(json_str)
                    try:
                        json.loads(cleaned_json)
                        self.logger.debug("Successfully extracted JSON object from response")
                        return cleaned_json
                    except json.JSONDecodeError:
                        self.logger.debug(f"Found potential JSON object but failed to parse: {cleaned_json[:100]}...")
                        break # from这JSONObject末尾Continuing 寻找

        # 4. 回退：直接fromresponseTextParsing score并构建JSON
        self.logger.info("No valid JSON found, attempting to parse scores directly from response text as a fallback.")
        scores = {}
        
        # 更强大后备Parsing 器，使用RegularizationExpressionfromText（特别is'Option analysis'partial）提取score。
        score_patterns = [
            re.compile(r"Option\s+([A-D]).*?(?:Probability|Score):\s*([0-9.]+)", re.IGNORECASE | re.DOTALL),
            re.compile(r"^\s*([A-D])[:.].*?(?:Probability|Score):\s*([0-9.]+)", re.IGNORECASE | re.DOTALL | re.MULTILINE),
            re.compile(r"→ Probability:\s*([0-9.]+)", re.IGNORECASE) # 捕获没hasOption标签score
        ]

        # 寻找 "Option analysis" partialwith提高精度
        analysis_section_match = re.search(r"Option analysis:(.*)", response, re.DOTALL | re.IGNORECASE)
        search_text = analysis_section_match.group(1) if analysis_section_match else response

        # perOrder尝试fromAtoD提取score
        options_to_find = ["A", "B", "C", "D"]
        found_scores = {}

        for pattern in score_patterns:
            matches = pattern.findall(search_text)
            if not matches:
                continue

            # Processing 不同RegularizationExpression捕获
            if len(matches[0]) == 2: # (option, score)
                for option, score_str in matches:
                    opt_key = option.upper()
                    if opt_key in options_to_find and opt_key not in found_scores:
                        try:
                            found_scores[opt_key] = float(score_str)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert score '{score_str}' to float for option {option}.")
            elif len(matches[0]) == 1: # (score)
                # 这mode下，我们HypothesisscoreisperA,B,C,DOrder出现
                unassigned_options = [opt for opt in options_to_find if opt not in found_scores]
                for i, score_str in enumerate(matches):
                    if i < len(unassigned_options):
                        opt_key = unassigned_options[i]
                        try:
                            found_scores[opt_key] = float(score_str)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert score '{score_str}' to float for option {opt_key}.")
            
            if len(found_scores) == 4:
                break

        # 确保我们找toall四Optionscore
        if len(found_scores) == 4:
            self.logger.info(f"Successfully parsed scores from response text via fallback: {found_scores}. Constructing JSON.")
            return json.dumps({"scores": found_scores})

        self.logger.warning("No valid JSON found in response and fallback parsing failed.")
        return ""

    def _diagnose_single_expert(self, case_data, expert_info, clarification_context=None):
        """
        as单expertsGenerating diagnosisandinference,并根据template要求进行Parsing 。
        """
        # OptimizationexpertInformationParsing 逻辑，确保correctGetting expertnameandspecialty
        expert_data = expert_info.get('expert', expert_info)  # 兼容不同Datastructure
        
        # 优先使用specialty_chinese_name作asexpert_name，这is最可靠标识符
        expert_name = (expert_data.get('specialty_chinese_name') or 
                      expert_info.get('specialty_chinese_name') or
                      expert_data.get('expert_name') or 
                      expert_info.get('expert_name') or
                      expert_data.get('expert_key') or
                      expert_info.get('expert_key'))
        
        expert_specialty = (expert_data.get('specialty_chinese_name') or 
                           expert_info.get('specialty_chinese_name') or
                           expert_data.get('specialty_name') or
                           expert_info.get('specialty_name'))
        
        # 如果仍然没hasGetting tovalidexpertname，Recordincorrect并Skipping 
        if not expert_name or expert_name == 'Unknown':
            self.logger.error(f"no法Getting validexpertname，expert_info: {expert_info}")
            return {"expert_name": "Unknown", "error": "Invalid expert name", "expert_info": expert_info}
        
        if not expert_specialty or expert_specialty == 'Unknown':
            expert_specialty = expert_name  # 使用expertname作asspecialtyfallback
        
        # 尝试Generating diagnosis，最多retry一times
        for attempt in range(2):  # 0: 第一times尝试, 1: retry
            try:
                result = self._generate_expert_diagnosis(case_data, expert_info, expert_name, expert_specialty, attempt, clarification_context)
                if "error" not in result:
                    return result
                elif attempt == 0:  # 第一timesFailed ，准备retry
                    self.logger.warning(f"expert {expert_name} 第一times尝试Failed ，准备retry: {result.get('error', 'Unknown error')}")
                    continue
                else:  # retry也Failed 
                    self.logger.error(f"expert {expert_name} retry后仍然Failed : {result.get('error', 'Unknown error')}")
                    return result
            except Exception as e:
                if attempt == 0:
                    self.logger.warning(f"expert {expert_name} 第一times尝试出现exception，准备retry: {e}")
                    continue
                else:
                    self.logger.error(f"expert {expert_name} retry后仍然出现exception: {e}", exc_info=True)
                    return {"expert_name": expert_name, "error": f"Exception after retry: {e}"}
        
        # 不应theto达这里
        return {"expert_name": expert_name, "error": "Unexpected error in retry logic"}

    def _generate_expert_diagnosis(self, case_data, expert_info, expert_name, expert_specialty, attempt=0, clarification_context=None):
        """
        Generating expertdiagnosis核心逻辑
        """
        log_messages = []

        try:
            # from case_data Getting theexpertWeight - 修复Weightmatching逻辑
            expert_weights = case_data.get("expert_weights", {})
            weight = 0.0
            
           
            
            # 首先尝试用expert_namematching
            if expert_name in expert_weights:
                weight = expert_weights[expert_name]
              
            # 然后尝试用expert_specialtymatching
            elif expert_specialty in expert_weights:
                weight = expert_weights[expert_specialty]
                
            # 尝试fromexpert_infoGetting its他可能matching键
            elif 'expert_key' in expert_info and expert_info['expert_key'] in expert_weights:
                weight = expert_weights[expert_info['expert_key']]
               
            # 最后尝试WeightDictionary查找包含expert_name键
            else:
                for weight_key in expert_weights.keys():
                    if (expert_name in weight_key or weight_key in expert_name or 
                        expert_specialty in weight_key or weight_key in expert_specialty):
                        weight = expert_weights[weight_key]
                      
                        break
                
                if weight == 0.0:
                    self.logger.warning(f"未能asexpert {expert_name} ({expert_specialty}) 找toWeight，使用Default值 0.0")

            # classificationquestion意graph
            question = case_data.get("question", "")
            options = case_data.get("options", {})
            context = case_data.get("context", [])
            intent = self._classify_question_intent(question, options, context)
            
            self.logger.debug(f"Question intent classified as: {intent} for expert: {expert_name}")

            # Creating 增强case_data，包含clarification_context
            enhanced_case_data = case_data.copy()
            if clarification_context:
                enhanced_case_data['clarification_context'] = clarification_context

            messages = self.template_manager.get_expert_template(
                expert_name=expert_name,
                case_data=enhanced_case_data,
                options=case_data.get("options", {}),
                expert_info=expert_info,
                intent=intent
            )
            if not messages:
                self.logger.error(f"Failed to generate prompt for expert {expert_name}")
                return {"expert_name": expert_name, "error": "Prompt generation failed."}

            # Initializing token计数（APIandthis地模型共用）
            input_tokens = 0
            generated_tokens = 0

            # Checking Whether to use API model
            if self.use_api_model and self.api_client and self.api_model_name and self.api_client.is_api_model(self.api_model_name):
                # Using API model
                try:
                    # 构建complete消息格式，包含system提示and用户消息
                    if isinstance(messages, list) and len(messages) > 0:
                        # 使用complete消息List
                        full_messages = messages
                        # aslog构建completepromptString
                        prompt_for_log = ""
                        for msg in messages:
                            if isinstance(msg, dict):
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                prompt_for_log += f"[{role.upper()}]\n{content}\n\n"
                            else:
                                prompt_for_log += str(msg) + "\n\n"
                    else:
                        # 如果不isList格式，Creating 用户消息
                        full_messages = [{"role": "user", "content": str(messages)}]
                        prompt_for_log = str(messages)
                    
                    # 准备LLMInputlog
                    log_messages.append("="*80)
                    log_messages.append(f"🔤 API LLMInput - expert: {expert_name} (尝试 {attempt + 1}) - 模型: {self.api_model_name}")
                    log_messages.append("="*80)
                    log_messages.append(prompt_for_log)
                    log_messages.append("="*80)
                    
                    # 调用API模型，传递complete消息格式
                    # API model only: unified temperature=0.7, max_tokens=4096 (consistent with MEDIQ/MDAgents/AI Hospital)
                    api_result = self.api_client.generate_with_messages(
                        model_name=self.api_model_name,
                        messages=full_messages,
                        max_new_tokens=4096,
                        temperature=0.7,  # 仅API模型
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        length_penalty=1.0
                    )
                    
                    full_response = api_result.get('generated_text', '')
                    
                    # fromAPIresponse提取token用量
                    usage = api_result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    generated_tokens = usage.get('completion_tokens', 0)
                    if input_tokens or generated_tokens:
                        log_messages.append(f"⏱️ APIGenerating statistics - Inputtokens: {input_tokens}, Outputtokens: {generated_tokens}")
                    
                    # 准备LLMOutputlog
                    log_messages.append("="*80)
                    log_messages.append(f"🔤 API LLMOutput - expert: {expert_name} (尝试 {attempt + 1}) - 模型: {self.api_model_name}")
                    log_messages.append("="*80)
                    log_messages.append(full_response)
                    log_messages.append("="*80)
                    
                except Exception as e:
                    self.logger.error(f"API model call failed: {e}")
                    return {"expert_name": expert_name, "error": f"API model call failed: {e}", "log_messages": log_messages}
            else:
                # 使用this地模型
                model, tokenizer = self.model_manager.get_model('main_llm')
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # 准备LLMInputlog
                log_messages.append("="*80)
                log_messages.append(f"🔤 LLMInput - expert: {expert_name} (尝试 {attempt + 1})")
                log_messages.append("="*80)
                log_messages.append(prompt)
                log_messages.append("="*80)

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

                with torch.no_grad():
                    start_ts = time.time()
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,#4096
                        temperature=0.3,  # unified diagnosis temperature
                        do_sample=True,
                        top_p=0.9,#0.8
                        top_k=50,#20
                        min_p=0,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=self._get_safe_eos_token_ids(tokenizer),
                        repetition_penalty=1.05,
                        length_penalty=1.0
                    )
                    elapsed = time.time() - start_ts

                full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                input_tokens = 0
                generated_tokens = 0
                try:
                    input_tokens = int(inputs.input_ids.shape[-1])
                    generated_tokens = int(outputs[0].shape[-1] - input_tokens)
                    log_messages.append(f"⏱️ Generating statistics - Inputtokens: {input_tokens}, Outputtokens: {generated_tokens}, 耗时: {elapsed:.2f}s")
                except Exception:
                    pass
                
                # 准备LLMOutputlog
                log_messages.append("="*80)
                log_messages.append(f"🔤 LLMOutput - expert: {expert_name} (尝试 {attempt + 1})")
                log_messages.append("="*80)
                log_messages.append(full_response)
                log_messages.append("="*80)

            # 1. 提取 <thinking> Content作as reasoning
            # support </thinking> and </think> 两End标签，or者直接to <json> 标签
            thinking_match = re.search(r"<thinking>(.*?)(?:</think(?:ing)?>|<json>)", full_response, re.DOTALL)
            reasoning = thinking_match.group(1).strip() if thinking_match else ""
            if not reasoning:
                # 回退strategy：如果没has<thinking>标签，提取<json>of前allText作asinference过程
                json_tag_pos = full_response.find('<json>')
                if json_tag_pos > 0:
                    reasoning = full_response[:json_tag_pos].strip()
                elif '```json' in full_response:
                    reasoning = full_response[:full_response.find('```json')].strip()
                if reasoning:
                    self.logger.debug(f"使用回退strategy提取inferenceContent from {expert_name}, Length: {len(reasoning)}")
                else:
                    self.logger.warning(f"Could not find <thinking> block in response from {expert_name}.")

            # 2. 提取 JSON Content
            json_str = self._extract_json_from_response(full_response)
            if not json_str:
                self.logger.error(f"Failed to find JSON in model output for {expert_name}.\\nRaw output: {full_response}")
                return {"expert_name": expert_name, "error": "Failed to find JSON in model output.", "raw_output": full_response, "log_messages": log_messages, "total_tokens": input_tokens + generated_tokens}


            # 3. Parsing  scores 并Computing  choice and confidence
            try:
                diagnosis_result = json.loads(json_str)
                scores = diagnosis_result.get("scores")
                if not scores or not isinstance(scores, dict):
                    raise ValueError("JSON output must contain a 'scores' dictionary.")

                # 修正超Rangescore：截断to[0.0, 1.0]，而非拒绝entireresult
                has_out_of_range = False
                for k, v in scores.items():
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"Score for '{k}' is not a number: {v}")
                    if v < 0.0 or v > 1.0:
                        has_out_of_range = True
                if has_out_of_range:
                    self.logger.warning(f"expert {expert_name} scorehas超Range值，已截断to[0,1]: {scores}")
                    scores = {option: max(0.0, min(1.0, float(score))) for option, score in scores.items()}
                
                # 归一化score，确保总andas1.0
                total_score = sum(scores.values())
                if total_score > 0:
                    scores = {option: score / total_score for option, score in scores.items()}
                else:
                    # 如果allscore都is0，则均etc.分配
                    num_options = len(scores)
                    scores = {option: 1.0 / num_options for option in scores.keys()}
                
                # 找toscore最高Option
                if not scores:
                    raise ValueError("Scores dictionary is empty.")
                
                choice = max(scores, key=scores.get)
                confidence = scores[choice]

                # 4. 装最终result
                final_result = {
                    "expert_name": expert_name,
                    "expert_specialty": expert_specialty,
                    "weight": float(weight),  # 确保WeightisPython floattype
                    "choice": choice,
                    "confidence": confidence,
                    "scores": scores,
                    "reasoning": reasoning,
                    "log_messages": log_messages,
                    "total_tokens": input_tokens + generated_tokens
                }
                
                # 添加Parsing result详细log（Creating 一不包含log_messagesCopy用于JSON序列化）
                log_messages.append("="*80)
                log_messages.append(f"📊 step六Parsing result - expert: {expert_name}")
                log_messages.append("="*80)
                result_for_logging = {k: v for k, v in final_result.items() if k != 'log_messages'}
                log_messages.append(json.dumps(result_for_logging, indent=2, ensure_ascii=False))
                log_messages.append("="*80)
                
                return final_result

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse JSON or validate data from {expert_name}: {e}\\nRaw JSON string: {json_str}\\nFull response: {full_response}")
                return {"expert_name": expert_name, "weight": weight, "error": f"Failed to parse or validate JSON: {e}", "raw_output": full_response, "log_messages": log_messages}

        except Exception as e:
            self.logger.error(f"Exception in _generate_expert_diagnosis for {expert_name}: {e}", exc_info=True)
            return {"expert_name": expert_name, "expert_specialty": expert_specialty, "weight": 0.0, "error": str(e), "log_messages": log_messages}

    def _prepare_expert_prompt_meta(self, case_data, expert_info, clarification_context, tokenizer):
        """准备单expertspromptand元Data（用于batchinference）

        from _generate_expert_diagnosis 提取prompt准备逻辑，
        不Executing 模型inference，仅returnpromptStringandexpert元Data。
        """
        expert_data = expert_info.get('expert', expert_info)
        expert_name = (expert_data.get('specialty_chinese_name') or
                      expert_info.get('specialty_chinese_name') or
                      expert_data.get('expert_name') or
                      expert_info.get('expert_name') or
                      expert_data.get('expert_key') or
                      expert_info.get('expert_key'))

        expert_specialty = (expert_data.get('specialty_chinese_name') or
                           expert_info.get('specialty_chinese_name') or
                           expert_data.get('specialty_name') or
                           expert_info.get('specialty_name'))

        if not expert_name or expert_name == 'Unknown':
            self.logger.error(f"no法Getting validexpertname，expert_info: {expert_info}")
            return None

        if not expert_specialty or expert_specialty == 'Unknown':
            expert_specialty = expert_name

        # Getting Weight
        expert_weights = case_data.get("expert_weights", {})
        weight = 0.0
        if expert_name in expert_weights:
            weight = expert_weights[expert_name]
        elif expert_specialty in expert_weights:
            weight = expert_weights[expert_specialty]
        elif 'expert_key' in expert_info and expert_info['expert_key'] in expert_weights:
            weight = expert_weights[expert_info['expert_key']]
        else:
            for weight_key in expert_weights.keys():
                if (expert_name in weight_key or weight_key in expert_name or
                    expert_specialty in weight_key or weight_key in expert_specialty):
                    weight = expert_weights[weight_key]
                    break

        # classificationquestion意graph
        question = case_data.get("question", "")
        options = case_data.get("options", {})
        context = case_data.get("context", [])
        intent = self._classify_question_intent(question, options, context)

        # Creating 增强case_data
        enhanced_case_data = case_data.copy()
        if clarification_context:
            enhanced_case_data['clarification_context'] = clarification_context

        # Getting template消息
        messages = self.template_manager.get_expert_template(
            expert_name=expert_name,
            case_data=enhanced_case_data,
            options=case_data.get("options", {}),
            expert_info=expert_info,
            intent=intent
        )
        if not messages:
            self.logger.error(f"no法asexpert {expert_name} Generating prompt")
            return None

        # 转换aspromptString
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            self.logger.error(f"expert {expert_name} prompttemplate应用Failed : {e}")
            return None

        log_messages = []
        log_messages.append("=" * 80)
        log_messages.append(f"🔤 LLMInput - expert: {expert_name} (Batch)")
        log_messages.append("=" * 80)
        log_messages.append(prompt)
        log_messages.append("=" * 80)

        return {
            'expert_name': expert_name,
            'expert_specialty': expert_specialty,
            'weight': weight,
            'expert_info': expert_info,
            'prompt': prompt,
            'log_messages': log_messages
        }

    def _postprocess_expert_response(self, full_response, expert_name, expert_specialty, weight, log_messages, input_tokens=0, generated_tokens=0):
        """Parsing expertdiagnosisresponse（from _generate_expert_diagnosis 提取后Processing 逻辑）

        Args:
            full_response: 模型OriginalOutputText
            expert_name: expertname
            expert_specialty: expertspecialty
            weight: expertWeight
            log_messages: log消息List
            input_tokens: Inputtoken数
            generated_tokens: Generating token数

        Returns:
            Parsing 后resultDictionary，Parsing Failed returnNone
        """
        # 1. 提取 <thinking> Content作as reasoning
        thinking_match = re.search(r"<thinking>(.*?)(?:</think(?:ing)?>|<json>)", full_response, re.DOTALL)
        reasoning = thinking_match.group(1).strip() if thinking_match else ""
        if not reasoning:
            # 回退strategy：提取<json>of前allText作asinference过程
            json_tag_pos = full_response.find('<json>')
            if json_tag_pos > 0:
                reasoning = full_response[:json_tag_pos].strip()
            elif '```json' in full_response:
                reasoning = full_response[:full_response.find('```json')].strip()
            if reasoning:
                self.logger.debug(f"使用回退strategy提取inferenceContent from {expert_name} (Batch), Length: {len(reasoning)}")
            else:
                self.logger.warning(f"Could not find <thinking> block in response from {expert_name}.")

        # 2. 提取 JSON Content
        json_str = self._extract_json_from_response(full_response)
        if not json_str:
            self.logger.error(f"Failed to find JSON in model output for {expert_name}.\nRaw output: {full_response}")
            return None

        # 3. Parsing  scores 并Computing  choice and confidence
        try:
            diagnosis_result = json.loads(json_str)
            scores = diagnosis_result.get("scores")
            if not scores or not isinstance(scores, dict):
                raise ValueError("JSON output must contain a 'scores' dictionary.")

            if not all(isinstance(v, (int, float)) for v in scores.values()):
                raise ValueError("All scores must be numbers.")

            # 修正超Rangescore：截断to[0.0, 1.0]，而非拒绝entireresult
            has_out_of_range = any(v < 0.0 or v > 1.0 for v in scores.values())
            if has_out_of_range:
                self.logger.warning(f"expert {expert_name} scorehas超Range值，已截断to[0,1]: {scores}")
                scores = {option: max(0.0, min(1.0, float(score))) for option, score in scores.items()}

            total_score = sum(scores.values())
            if total_score > 0:
                scores = {option: score / total_score for option, score in scores.items()}
            else:
                num_options = len(scores)
                scores = {option: 1.0 / num_options for option, score in scores.items()}

            if not scores:
                raise ValueError("Scores dictionary is empty.")

            choice = max(scores, key=scores.get)
            confidence = scores[choice]

            final_result = {
                "expert_name": expert_name,
                "expert_specialty": expert_specialty,
                "weight": float(weight),
                "choice": choice,
                "confidence": confidence,
                "scores": scores,
                "reasoning": reasoning,
                "log_messages": log_messages,
                "total_tokens": input_tokens + generated_tokens
            }

            log_messages.append("=" * 80)
            log_messages.append(f"📊 step六Parsing result - expert: {expert_name} (Batch)")
            log_messages.append("=" * 80)
            result_for_logging = {k: v for k, v in final_result.items() if k != 'log_messages'}
            log_messages.append(json.dumps(result_for_logging, indent=2, ensure_ascii=False))
            log_messages.append("=" * 80)

            return final_result

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse JSON or validate data from {expert_name}: {e}\nRaw JSON string: {json_str}\nFull response: {full_response}")
            return None

    def _batch_diagnose_experts_local(self, case_data, activated_experts_info, clarification_context=None):
        """使用BatchinferenceParallelProcessing allexpertdiagnosis（this地模型专用Optimization）

        多expertsprompt打包成batch，一timesmodel.generate()调用ParallelGenerating allresult。
        利用GPUforbatchDimensionParallelComputing 能力，相比SerialExecuting 可提升2-3倍速度。
        Parsing Failed expert会自动回退to单独inferenceretry。

        Returns:
            (results, total_tokens): resultListand总token数
        """
        try:
            model, tokenizer = self.model_manager.get_model('main_llm')
        except Exception as e:
            self.logger.error(f"Getting this地模型Failed ，回退toSerialmode: {e}")
            # 回退toSerialmode
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_expert = {executor.submit(self._diagnose_single_expert, case_data, ei, clarification_context): ei for ei in activated_experts_info}
                results = []
                total_tokens = 0
                for future in as_completed(future_to_expert):
                    try:
                        result = future.result()
                        results.append(result)
                        total_tokens += result.get('total_tokens', 0)
                    except Exception as exc:
                        ei = future_to_expert[future]
                        expert_name = ei.get('expert', ei).get('specialty_chinese_name', 'Unknown') if 'expert' in ei else ei.get('expert_name', 'Unknown')
                        results.append({'expert_name': expert_name, 'error': str(exc)})
            return results, total_tokens

        # 确保tokenizerhaspad_token（batch tokenization需要）
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Phase 1: aseachexperts准备promptand元Data
        expert_meta_list = []
        prompts = []
        failed_experts = []  # Recordprompt准备Failed expert

        for expert_info in activated_experts_info:
            meta = self._prepare_expert_prompt_meta(case_data, expert_info, clarification_context, tokenizer)
            if meta is not None:
                expert_meta_list.append(meta)
                prompts.append(meta['prompt'])
            else:
                failed_experts.append(expert_info)

        if not prompts:
            self.logger.warning("allexpertprompt准备Failed ，回退toSerialmode")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_expert = {executor.submit(self._diagnose_single_expert, case_data, ei, clarification_context): ei for ei in activated_experts_info}
                results = []
                total_tokens = 0
                for future in as_completed(future_to_expert):
                    try:
                        result = future.result()
                        results.append(result)
                        total_tokens += result.get('total_tokens', 0)
                    except Exception:
                        pass
            return results, total_tokens

        # Phase 2: Batch tokenize with padding
        self.logger.info(f"📦 Starting Batchinference: {len(prompts)}experts同时Processing ")
        batch_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(model.device)

        # Phase 3: 单timesbatch generate（GPUParallelComputing allexpert）
        input_seq_len = batch_inputs.input_ids.shape[-1]  # padding后统一InputLength
        with torch.no_grad():
            batch_start = time.time()
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=2048,  # and单expertinference保持一致，避免batchmode下paddingwaiting过长
                temperature=0.3,  # unified diagnosis temperature
                do_sample=True,
                top_p=0.9,
                top_k=50,
                min_p=0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=self._get_safe_eos_token_ids(tokenizer),
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            batch_elapsed = time.time() - batch_start

        self.logger.info(f"📦 Batchinferencecomplete: {len(prompts)}experts, Total time {batch_elapsed:.2f}s")

        # Phase 4: Decode and post-process each response
        results = []
        total_tokens = 0
        retry_experts = []  # 需要单独retryexpert

        for i, meta in enumerate(expert_meta_list):
            expert_name = meta['expert_name']
            log_messages = meta['log_messages']

            try:
                # 实际Inputtoken数（排除padding）
                actual_input_tokens = int(batch_inputs.attention_mask[i].sum())
                generated_tokens = int(batch_outputs[i].shape[-1] - input_seq_len)

                log_messages.append(f"⏱️ BatchGenerating statistics - Inputtokens: {actual_input_tokens}, Outputtokens: {generated_tokens}, Total time: {batch_elapsed:.2f}s")

                # Decode: Skipping entireInputpartial（含padding），只取新Generating token
                full_response = tokenizer.decode(
                    batch_outputs[i][input_seq_len:],
                    skip_special_tokens=True
                )

                log_messages.append("=" * 80)
                log_messages.append(f"🔤 LLMOutput - expert: {expert_name} (Batch)")
                log_messages.append("=" * 80)
                log_messages.append(full_response)
                log_messages.append("=" * 80)

                # Post-process response
                result = self._postprocess_expert_response(
                    full_response, expert_name, meta['expert_specialty'],
                    meta['weight'], log_messages,
                    input_tokens=actual_input_tokens,
                    generated_tokens=generated_tokens
                )

                if result is not None:
                    results.append(result)
                    total_tokens += result.get('total_tokens', 0)
                else:
                    self.logger.warning(f"expert {expert_name} BatchresultParsing Failed ，单独retry")
                    retry_experts.append(meta['expert_info'])

            except Exception as e:
                self.logger.error(f"Processing expert {expert_name} Batchresult时出错: {e}", exc_info=True)
                retry_experts.append(meta['expert_info'])

        # forprompt准备Failed expert也加入retryList
        retry_experts.extend(failed_experts)

        # Phase 5: forFailed expert进行单独retry
        if retry_experts:
            self.logger.info(f"🔄 for {len(retry_experts)} experts进行单独retry...")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_expert = {executor.submit(self._diagnose_single_expert, case_data, ei, clarification_context): ei for ei in retry_experts}
                for future in as_completed(future_to_expert):
                    try:
                        result = future.result()
                        results.append(result)
                        total_tokens += result.get('total_tokens', 0)
                    except Exception as exc:
                        ei = future_to_expert[future]
                        expert_name = ei.get('expert', ei).get('specialty_chinese_name', 'Unknown') if 'expert' in ei else ei.get('expert_name', 'Unknown')
                        self.logger.error(f'retryexpert {expert_name} Failed : {exc}')
                        results.append({'expert_name': expert_name, 'error': str(exc)})

        return results, total_tokens

    def step_6_diagnose_and_reasoning(self, case_data, activated_experts_info, clarification_context=None):
        self.logger.info("🩺 step6：Doctor Agentdiagnosisinference")
        if clarification_context:
            self.logger.info(f"收to澄清Context，包含 {len(clarification_context)} 澄清Record")
        start_time = time.time()
        diagnoses = []
        total_tokens = 0
        results = []

        if not self.use_api_model:
            # this地模型: 使用Batchinference（GPUParallelComputing ，不依赖线程安全）
            # allexpertprompt打包成batch，一timesmodel.generate()调用complete
            results, total_tokens = self._batch_diagnose_experts_local(
                case_data, activated_experts_info, clarification_context
            )
        else:
            # API模型: 使用多线程Parallel（APInoGPU竞争question，线程安全）
            max_workers = min(len(activated_experts_info), 3)
            self.logger.info(f"🌐 APImode: 使用 {max_workers} 线程ParallelProcessing ")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_expert = {executor.submit(self._diagnose_single_expert, case_data, expert_info, clarification_context): expert_info for expert_info in activated_experts_info}
                for future in as_completed(future_to_expert):
                    expert_info = future_to_expert[future]
                    try:
                        result = future.result()
                        results.append(result)
                        total_tokens += result.get('total_tokens', 0)
                    except Exception as exc:
                        expert_name = expert_info.get('expert_name', 'Unknown')
                        self.logger.error(f'{expert_name} generated an exception: {exc}')
                        results.append({'expert_name': expert_name, 'error': str(exc)})

        # Recordallexpert详细logtologfile
        for result in sorted(results, key=lambda x: x.get('expert_name', '')):
            if 'log_messages' in result and result['log_messages']:
                for msg in result['log_messages']:
                    self.logger.info(msg)

        # 收集diagnosis result
        for result in sorted(results, key=lambda x: x.get('expert_name', '')):
            diagnoses.append(result)

        end_time = time.time()
        self.logger.info(f"step6complete，耗时 {end_time - start_time:.2f} seconds")

        return {"expert_opinions": diagnoses, "total_tokens": total_tokens}

    # _extract_json_from_responseMethod后添加retry逻辑
    def _generate_expert_response_with_retry(self, expert_name, prompt, max_retries=2):
        for attempt in range(max_retries + 1):
            try:
                result = self._generate_expert_response(expert_name, prompt)
                if 'error' not in result:
                    return result
                
                if attempt < max_retries:
                    self.logger.warning(f"retryexpert{expert_name}response，尝试{attempt + 1}/{max_retries}")
                    # 调整Parameter进行retry
                    continue
            except Exception as e:
                if attempt == max_retries:
                    return {"expert_name": expert_name, "error": str(e)}
        
        return result