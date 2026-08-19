"""
Optimized Unified Expert Template Manager
Eliminates redundancy by moving chain of thought into <thinking> block
"""

import json
import re
from typing import Dict, Any, Optional, List


class UnifiedExpertTemplateManager:
    """
    Unified Expert Template Manager - Optimized Version
    Eliminate duplicate structures, move chain-of-thought into <thinking>
    """

    def __init__(self, force_generic_cot: bool = False):
        """Initialize template manager"""
        # Expert template related
        self.default_template_name = "high_accuracy_json_expert_template"
        self.force_generic_cot = force_generic_cot  # Ablation: force use of generic CoT template
        

        
        # Evaluator agent template dictionary
        self._evaluator_templates = {
            "step2_quality_assessment": self._get_step2_quality_assessment_template,
            "step3_clarification_generation": self._get_step3_clarification_generation_template,
        }
        
        # Patient agent template dictionary
        self._patient_templates = {
            "step4_clarification_response": self._get_step4_patient_response_template,
            "step4_patient_response": self._get_step4_patient_response_template,  # Add alias
        }
        
        # Expert intent template dictionary - maps question intent to template method
        self._expert_templates = {
            "Diagnosis": self._get_diagnosis_prompt_templates,
            "Management": self._get_treatment_prompt_templates,
            "Knowledge": self._get_knowledge_prompt_templates,
            "Mechanism": self._get_mechanism_prompt_templates,
            "Ethics": self._get_ethics_prompt_templates,
            "Tests": self._get_test_prompt_templates,
            "Prevention": self._get_prevention_prompt_templates,
            "Generic_CoT": self._get_generic_cot_prompt_templates  # Generic CoT template
        }
        
        # Specialty clarification question template library
        self._specialty_clarification_templates = {
            "Gynecology": ["Menstrual history questions", "Pregnancy history questions", "Sexual history questions"],
            "Neurology": ["Consciousness assessment", "Neurological exam results", "Cranial imaging"],
            "Cardiology": ["Chest pain characteristics", "Exercise tolerance assessment", "ECG results"],
            "Urology": ["Urinary symptoms detail", "Urinary system examination", "History of urinary diseases"],
            "消化科": ["消化道symptom", "diethabits", "既往消化disease史"],
            "呼吸科": ["呼吸symptom详情", "吸烟史", "胸部imagingChecking "],
            "内分泌科": ["代谢相关symptom", "家族遗传史", "激素水平Checking "],
            "骨科": ["pain性质and部", "外伤史", "imaging学Checking "],
            "皮肤科": ["皮损特征", "allergy史", "接触史"],
            "眼科": ["视力变化", "眼部symptom", "眼科Checking "],
            "耳鼻喉科": ["听力变化", "鼻咽symptom", "specialtyChecking "],
            "精神科": ["精神status", "睡眠情况", "生活社会因素"],
            "急诊科": ["急性symptom", "生命sign", "紧急Processing "],
            "儿科": ["生长发育", "疫苗接史", "儿童特hassymptom"],
            "老年科": ["多病共存", "medication史", "functionalitystatusevaluation"]
        }
        
        # experttype映射
        self.expert_type_mapping = {
            "Cardiovascular Disease": "Cardiology",
            "Pediatrics": "Pediatrics", 
            "General Infectious Disease": "Infectious Disease",
            "Gastroenterology": "Gastroenterology",
            "Neurology": "Neurology",
            "Endocrinology": "Endocrinology",
            "Pulmonology": "Pulmonology",
            "Nephrology": "Nephrology",
            "Hematology": "Hematology",
            "Oncology": "Oncology",
            "Rheumatology": "Rheumatology",
            "Dermatology": "Dermatology",
            "Psychiatry": "Psychiatry",
            "Ophthalmology": "Ophthalmology",
            "Otolaryngology": "ENT",
            "Orthopedics": "Orthopedics",
            "Urology": "Urology",
            "Gynecology": "Gynecology",
            "Emergency Medicine": "Emergency Medicine",
            "Anesthesiology": "Anesthesiology",
            "Radiology": "Radiology",
            "Pathology": "Pathology",
            "Surgery": "Surgery"
        }

        # 意graphtype映射
        self.intent_mapping = {
            "Diagnosis": "diagnosis",
            "Management": "treatment", 
            "Treatment": "treatment",
            "Knowledge": "knowledge",
            "Mechanism": "mechanism",
            "Ethics": "ethics",
            "Tests": "test",
            "Prevention": "prevention"
        }

        # 质量evaluationDimension
        self.quality_dimensions = {
            "clinical_relevance": "clinical相关性",
            "diagnostic_clarity": "diagnosis清晰度", 
            "treatment_specificity": "treatment特异性",
            "evidence_completeness": "evidencecomplete性",
            "risk_assessment": "风险evaluation",
            "patient_context": "patient背景",
            "temporal_progression": "Time进展",
            "differential_analysis": "鉴别analysis"
        }

    def _get_diagnosis_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Diagnosis intent template (English) - Optimized
        Focus: pathophysiology linkage, specific features, time-course, and evidence-weighted differential.
        """
        system_prompt = """You are a {specialty_name} diagnosis expert. Task: assign probability scores to the four options for the "most likely diagnosis". 

Core constraints:
- Highest score must be ≥ 0.5 (clear preference)
- A + B + C + D = 1.0 (normalized probabilities)
- Differentiate by evidence strength; never average scores

Output format:
<thinking>
1. Pathophysiology linkage: connect mechanisms to clinical presentation and findings
2. Specific features: identify diagnosis-specific symptoms, signs, and key tests
3. Temporal pattern: analyze onset, progression, and natural history
4. Option analysis:
   - Option A: [fit to mechanism/features/timing] → Probability: X.X
   - Option B: [fit to mechanism/features/timing] → Probability: X.X
   - Option C: [fit to mechanism/features/timing] → Probability: X.X
   - Option D: [fit to mechanism/features/timing] → Probability: X.X
5. Evidence integration: use guidelines, criteria, and authoritative references
6. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}
"""
        return system_prompt, user_prompt

    def _get_generic_cot_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Generic Chain of Thought template for ablation study
        Unified reasoning template without specialty-specific guidance
        """
        system_prompt = """You are a medical expert. Task: assign probability scores to the four options for the "most likely diagnosis". 

 CRITICAL: You MUST output the JSON scores . 
<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>
"""

        user_prompt = """
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}
"""
        return system_prompt, user_prompt

    def _get_treatment_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Management/Treatment intent template (English) - Optimized
        Focus: severity, goals, individualized factors, evidence, and benefit-risk.
        """
        system_prompt = """You are a {specialty_name} management and treatment expert. Task: select the "best next management/treatment step" and score each option. 

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by evidence; never average scores

Output format:
<thinking>
1. Severity and urgency: evaluate acuity, instability, and threats to life
2. Treatment goals: short-term stabilization, medium-term improvement, long-term outcomes
3. Individualized factors: age, comorbidities, contraindications, tolerance, adherence
4. Option analysis:
   - Option A: [indications/contraindications/benefit-risk/applicability] → Probability: X.X
   - Option B: [indications/contraindications/benefit-risk/applicability] → Probability: X.X
   - Option C: [indications/contraindications/benefit-risk/applicability] → Probability: X.X
   - Option D: [indications/contraindications/benefit-risk/applicability] → Probability: X.X
5. Evidence integration: guidelines, trials, and clinical standards
6. Outcome projection: expected benefits and risks for each option
7. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """Current case:
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}"""
        return system_prompt, user_prompt

    def _get_knowledge_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Knowledge intent template (English) - Optimized
        专门Processing 非机制知识question
        """
        system_prompt = """You are a {specialty_name} expert. Task: answer a medical knowledge question and score each option.

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by evidence; never average scores

Output format:
<thinking>
1. Guidelines/Standards: relevant clinical guidelines, practice standards, and recommendations
2. Evidence summary: systematic reviews, meta-analyses, and high-quality studies
3. Clinical context: patient population, setting, and practical considerations
4. Option analysis:
   - Option A: [guideline alignment/evidence support/clinical applicability] → Probability: X.X
   - Option B: [guideline alignment/evidence support/clinical applicability] → Probability: X.X
   - Option C: [guideline alignment/evidence support/clinical applicability] → Probability: X.X
   - Option D: [guideline alignment/evidence support/clinical applicability] → Probability: X.X
5. Fact matching: direct alignment between options and established medical facts
6. Clinical applicability: real-world relevance and implementation
7. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """Current case:
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}"""
        return system_prompt, user_prompt

    def _get_mechanism_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Mechanism intent template (English) - Optimized
        专门Processing 机制相关question
        """
        system_prompt = """You are a {specialty_name} expert. Task: answer a pharmacology/pathophysiology mechanism question and score each option. 

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by evidence; never average scores

Output format:
<thinking>
1. Molecular mechanism: receptor/target → signaling → cellular response → organ effect
2. Pharmacokinetics/dynamics: ADME and clinical significance
3. Pathophysiology link: explain clinical findings via mechanistic basis
4. Option analysis:
   - Option A: [mechanistic accuracy/evidence/clinical relevance] → Probability: X.X
   - Option B: [mechanistic accuracy/evidence/clinical relevance] → Probability: X.X
   - Option C: [mechanistic accuracy/evidence/clinical relevance] → Probability: X.X
   - Option D: [mechanistic accuracy/evidence/clinical relevance] → Probability: X.X
5. Evidence integration: authoritative references, drug labels, guidelines
6. Clinical relevance: applicability and limitations in practice
7. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """Current case:
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}"""
        return system_prompt, user_prompt

    def _get_ethics_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Ethics/Professionalism intent template (English) - Optimized
        """
        system_prompt = """You are a {specialty_name} expert.Task: analyze based on ethical principles and score each option. 

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by strength of ethical justification

Output format:
<thinking>
1. Ethical principles: autonomy, beneficence, non-maleficence, justice (priority and application)
2. Stakeholders: patient, family, clinicians, institution, society
3. Conflicts and balance: identify conflicts and prioritize values
4. Option analysis:
   - Option A: [principle alignment/legal compliance/social fit/long-term impact] → Probability: X.X
   - Option B: [principle alignment/legal compliance/social fit/long-term impact] → Probability: X.X
   - Option C: [principle alignment/legal compliance/social fit/long-term impact] → Probability: X.X
   - Option D: [principle alignment/legal compliance/social fit/long-term impact] → Probability: X.X
5. Legal and policy: relevant laws, regulations, and professional standards
6. Cultural and social context: impact on ethical decision-making
7. Long-term consequences: effects on patient, relationships, and system
8. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}

Ethics focus:
1. Ethical principles and moral considerations in each option
2. Balance of patient rights and professional responsibilities
3. Legal/regulatory compliance
4. Best recommendation grounded in ethics

Return strictly in the specified JSON format."""
        return system_prompt, user_prompt

    def _get_test_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Tests intent template (English) - Optimized
        """
        system_prompt = """You are a {specialty_name} expert. Task: select the most appropriate test and interpret its value; score each option.

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by test performance and clinical utility

Output format:
<thinking>
1. Clinical integration: synthesize symptoms, signs, history, labs
2. Test performance: sensitivity, specificity, PPV, NPV, most accurate vs. initial
3. Indications/contraindications: appropriateness and safety
4. Option analysis:
   - Option A: [performance/indication/cost/access/diagnostic value] → Probability: X.X
   - Option B: [performance/indication/cost/access/diagnostic value] → Probability: X.X
   - Option C: [performance/indication/cost/access/diagnostic value] → Probability: X.X
   - Option D: [performance/indication/cost/access/diagnostic value] → Probability: X.X
5. Cost-effectiveness and accessibility: practicality and patient tolerance
6. Diagnostic pathway: evidence-based sequencing (Bayesian reasoning)
7. Result interpretation: impact on diagnosis and management decisions
8. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """
Current case:
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}

Testing focus:
1. Indications and clinical value of each test
2. Sensitivity, specificity, and expected findings
3. Timing, sequencing, cost-effectiveness
4. Best test recommendation for this patient

Return strictly in the specified JSON format."""
        return system_prompt, user_prompt

    def _get_prevention_prompt_templates(self, case_data: Dict[str, Any]) -> tuple:
        """
        Prevention intent template (English) - Optimized
        """
        system_prompt = """You are a {specialty_name} expert. Task: propose individualized prevention strategies and score each option. 

Core constraints:
- Highest score must be ≥ 0.5
- A + B + C + D = 1.0
- Differentiate by evidence and feasibility

Output format:
<thinking>
1. Multidimensional risk assessment: genetic, environmental, behavioral, socioeconomic
2. Prevention tiers: primary (cause-level), secondary (screening/early detection), tertiary (complication prevention)
3. Evidence integration: guidelines, epidemiology, trials
4. Option analysis:
   - Option A: [tier/evidence/risk-match/cost/feasibility] → Probability: X.X
   - Option B: [tier/evidence/risk-match/cost/feasibility] → Probability: X.X
   - Option C: [tier/evidence/risk-match/cost/feasibility] → Probability: X.X
   - Option D: [tier/evidence/risk-match/cost/feasibility] → Probability: X.X
5. Individualized risk stratification: tools and models
6. Cost-effectiveness and feasibility: adherence and resource considerations
7. Long-term benefits: health outcomes and quality of life
8. Top choice rationale: why highest ≥ 0.5
[Ensure sums=1.0 and max≥0.5]
</thinking>

 CRITICAL: You MUST output the JSON scores after thinking. Do NOT skip the JSON block!

<json>
{{
    "scores": {{
        "A": <0.0-1.0>,
        "B": <0.0-1.0>,
        "C": <0.0-1.0>,
        "D": <0.0-1.0>
    }}
}}
</json>

Let's think step by step."""

        user_prompt = """
Current case:
Patient: {patient_age} years, {patient_gender}
Case background: {original_context}
Clarification: {clarification_context}
Question: {question}
Options: {options}

Prevention focus:
1. Patient risk factors and prevention needs
2. Indications/contraindications for each preventive option
3. Prioritization and implementation strategy
4. Individualized plan for this patient

Return strictly in the specified JSON format."""
        return system_prompt, user_prompt

    def get_expert_template(self, expert_name: str, case_data: Dict[str, Any], options: Dict[str, str], expert_info: Dict[str, Any], intent: str = "Diagnosis") -> List[Dict[str, str]]:
        """
        Getting experttemplate - OptimizationVersion
        """
        # 消融实验：如果Enable 通用CoT，Force 使用通用template
        if self.force_generic_cot:
            intent = "Generic_CoT"
        
        # 映射意graphtype
        intent_key = self.intent_mapping.get(intent, "diagnosis")
        if intent == "Generic_CoT":
            intent_key = "generic_cot"
        
        # Getting for应templateFunction
        template_method = getattr(self, f"_get_{intent_key}_prompt_templates", None)
        if not template_method:
            raise ValueError(f"Unknown intent: {intent}")
            
        # Getting template
        system_prompt, user_prompt = template_method(case_data)
        
        # 格式化template
        specialty_name = expert_info.get('specialty', expert_name)
        system_prompt = system_prompt.format(specialty_name=specialty_name)
        
        # 构建合并后澄清载荷：澄清for话 + 事实（如has）
        clar_dialogue = self._format_clarification_context(case_data.get('clarification_context', []))
        facts_text = self._extract_atomic_facts(case_data)
        has_facts = bool(facts_text and facts_text.strip() and facts_text.strip() != "No specific atomic facts provided.")
        has_clarification = clar_dialogue.strip() != "No clarification dialogue available."
        
        # 统一构造用户提示，per需插入 Clarification and Facts
        patient_age = case_data.get('patient', {}).get('age', 'Unknown')
        patient_gender = case_data.get('patient', {}).get('gender', 'Unknown')
        original_context = case_data.get('context', [''])[0] if isinstance(case_data.get('context', []), list) and case_data.get('context', []) else case_data.get('context', '')
        question = case_data.get('question', '')
        options_text = self._format_options(options)
        
        prompt_lines = [
            f"Patient: {patient_age} years, {patient_gender}",
            f"Case background: {original_context}",
        ]
        
        if has_clarification:
            # 仅确has澄清for话时插入 Clarification；Facts 作astheSegment补充
            if has_facts:
                prompt_lines.append(f"Clarification: {clar_dialogue}\n\nFacts:\n{facts_text}")
            else:
                prompt_lines.append(f"Clarification: {clar_dialogue}")
        else:
            # no澄清for话时，不插入 Clarification Segment，单独呈现 Facts
            if has_facts:
                prompt_lines.append(f"Facts:\n{facts_text}")
        
        prompt_lines.append(f"Question: {question}")
        prompt_lines.append(f"Options: {options_text}")
        
        formatted_user_prompt = "\n".join(prompt_lines)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]
    
    def get_evaluator_template(self, template_name: str, **kwargs) -> str:
        """
        Getting Evaluator Agenttemplate
        
        Args:
            template_name: templatename
                - "step2_quality_assessment": step二Information质量evaluation
                - "step3_clarification_generation": step三Information澄清Generating 
            **kwargs: templateParameter
            
        Returns:
            str: 格式化后templateContent
            
        Raises:
            KeyError: 当templatenamedoes not exist时
        """
        if template_name not in self._evaluator_templates:
            available_templates = list(self._evaluator_templates.keys())
            raise KeyError(f"Evaluator Agenttemplate '{template_name}' does not exist。availabletemplate: {available_templates}")
        
        return self._evaluator_templates[template_name](**kwargs)
    
    def get_patient_template(self, template_name: str, **kwargs) -> str:
        """
        Getting Patient Agenttemplate
        
        Args:
            template_name: templatename
                - "step4_clarification_response": step四patient澄清answer
            **kwargs: templateParameter
            
        Returns:
            str: 格式化后templateContent
            
        Raises:
            KeyError: 当templatenamedoes not exist时
        """
        if template_name not in self._patient_templates:
            available_templates = list(self._patient_templates.keys())
            raise KeyError(f"Patient Agenttemplate '{template_name}' does not exist。availabletemplate: {available_templates}")
        
        return self._patient_templates[template_name](**kwargs)
    
    def _get_step2_quality_assessment_template(self, case_info: str) -> str:
        """
        step二：Evaluator AgentInformation质量evaluation提示词template
        强化JSONCodeBlockOutput格式，严格LimitOutputContent。
        """
        return f"""You are a senior medical expert responsible for evaluating the quality of medical case reports.

**CRITICAL OUTPUT REQUIREMENTS:**
1. You MUST output ONLY a JSON code block using ```json markers
2. NO additional text, explanations, greetings, or reasoning outside the JSON block
3. NO markdown formatting except the required ```json code block
4. The JSON object must contain EXACTLY the 5 specified keys with integer values 0-10

**Case Information:**
{case_info}

**Evaluation Dimensions (0-10 scale):**
- **basic_score**: Patient demographics, chief complaint, medical history completeness
- **symptom_score**: Comprehensive symptom details (location, characteristics, duration, etc.)
- **exam_score**: Physical examination findings and diagnostic test results completeness  
- **timeline_score**: Clear chronological progression of illness
- **logic_score**: Internal consistency of all information provided

**REQUIRED OUTPUT FORMAT (EXACTLY as shown):**

```json
{{
    "basic_score": <integer_0_to_10>,
    "symptom_score": <integer_0_to_10>,
    "exam_score": <integer_0_to_10>,
    "timeline_score": <integer_0_to_10>,
    "logic_score": <integer_0_to_10>
}}
```

**STOP IMMEDIATELY after the closing ``` - no additional text allowed.**"""

    def _get_step3_clarification_generation_template(self, assessment_result: Dict[str, Any], case_data: Dict[str, Any]) -> str:
        """
        Step 3: Clarification question generation based on question-option combinations
        Generate 1-2 specific questions targeting option differences, output as question string list
        """
        context = case_data.get('context', [])
        question = case_data.get('question', '')
        options = case_data.get('options', {})
        patient_info = case_data.get('patient', {})
        
        # Get initial case overview (context[0])
        initial_context = context[0] if context else ""
        
        # Get patient age and gender
        age = patient_info.get('age', 'Unknown')
        gender = patient_info.get('gender', 'Unknown')
        
        # Build options list
        options_text = ""
        for key, value in options.items():
            options_text += f"{key}. {value}\n"
        
        # Extract known facts to guide question generation
        atomic_facts = case_data.get('atomic_facts', [])
        facts = case_data.get('facts', [])
        available_facts = atomic_facts if atomic_facts else facts
        
        facts_text = ""
        if available_facts:
            # 使用allfacts：atomic_facts优先，否则facts；不再截断前10
            facts_text = "\n".join([f"- {fact}" for fact in available_facts])
        
        return f"""You are a medical expert generating targeted clarification questions based on diagnostic options and known facts.

**OUTPUT MUST BE ENGLISH ONLY**

**STRICT OUTPUT REQUIREMENT:**
Output ONLY a JSON array of question strings. No explanations, no numbering inside strings, no extra text.

**Case Information:**
Question: {question}

**Diagnostic Options:**
{options_text}

**Initial Case Description:**
{initial_context}

**Patient Demographics:**
- Age: {age}
- Gender: {gender}

**Known Facts (use these to guide your questions):**
{facts_text}

**Task:**
1. Analyze the key differences between diagnostic options based on known facts
2. Generate up to 3 specific questions that can help distinguish between options
3. Focus on information gaps that are NOT already covered in the known facts
4. Prioritize questions about discriminative features that can differentiate the options
5. Consider patient demographics and pathophysiology differences

**DEDUPLICATION & SPECIFICITY CONSTRAINTS:**
- AVOID asking about information already provided in known facts
- AVOID repetitive generic questions like "What is the family history?" or "Are there other lesions?"
- English only; no Chinese; no meta-questions.
- FOCUS on distinguishing features that differentiate options:
  * Location characteristics: Specific anatomical location, distribution pattern
  * Morphological features: Size, shape, color, texture, surface characteristics
  * Timeline: Onset time, progression speed, periodicity, seasonal patterns
  * Triggers: Specific precipitating factors, environmental factors, medications
  * Associated symptoms: Pain, itching, functional impairment, systemic symptoms
  * Physical examination: Palpation findings, mobility, consistency, temperature
- PRIORITIZE questions that can rule out/confirm specific options based on pathophysiology

**Output Format (EXACT FORMAT REQUIRED):**
["Question 1", "Question 2", "Question 3"]

**Examples of good targeted questions:**
- "What is the consistency and mobility of the lesion on palpation?"
- "Are there any associated systemic symptoms like fever or weight loss?"
- "What specific triggers or medications preceded the onset of symptoms?"
- "What are the exact morphological characteristics of the lesion surface?"
- "How does the lesion respond to pressure or manipulation?"

**IMPORTANT:** Output ONLY the list of questions as shown above. No additional text, explanations, or formatting."""

    

    def _get_step4_patient_response_template(self, clarification_questions: str, case_data: Dict[str, Any]) -> str:
        """
        Step 4: Patient clarification response - English-only answers strictly based on context and facts.
        """
        context_str = '\n'.join(case_data.get('context', []))
        patient_info = case_data.get('patient', {})
        
        # Prioritize atomic_facts, fallback to facts if not available
        atomic_facts = case_data.get('atomic_facts', [])
        facts = case_data.get('facts', [])
        available_facts = atomic_facts if atomic_facts else facts
        
        facts_text = "\n".join([f"- {fact}" for fact in available_facts]) if available_facts else "No detailed fact information available"
        
        # Handle question list
        if isinstance(clarification_questions, list):
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(clarification_questions)])
        else:
            # If it's a string, try to parse as list
            try:
                import ast
                if clarification_questions.startswith('[') and clarification_questions.endswith(']'):
                    questions_list = ast.literal_eval(clarification_questions)
                    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
                else:
                    questions_text = str(clarification_questions)
            except:
                questions_text = str(clarification_questions)

        return f"""You are a patient answering clarification questions based strictly on your case information.

**OUTPUT MUST BE ENGLISH ONLY**

**STRICT OUTPUT REQUIREMENT:**
Output ONLY a JSON array of answer strings. No explanations, no numbering inside strings, no extra text.

**Your Case Information:**
{context_str}

**Available Facts (PRIMARY SOURCE FOR ANSWERS):**
{facts_text}

**Clarification Questions:**
{questions_text}

**STRICT ANSWERING RULES:**
1. **FACT-BASED ONLY**: Answer ONLY based on information explicitly stated in your case context and available facts
2. **NO INFERENCE**: Do NOT invent, assume, or infer any information not directly stated
3. **PRIORITIZE FACTS**: When available facts contain relevant information, use them as the primary source
4. **MISSING INFORMATION**: If information is not available in facts or context, answer exactly "Not provided"
5. **BE SPECIFIC**: When facts provide specific details (numbers, dates, test results), include them in your answer
6. **ONE ANSWER PER QUESTION**: Provide exactly one answer for each question

**Answer Decision Process:**
- Check available facts first for relevant information
- If facts contain the answer: Provide specific fact-based answer
- If facts partially address the question: State what is known from facts + "Not provided" for missing parts
- If facts do not address the question: Check case context
- If neither facts nor context contain the information: Answer "Not provided"

**Output Format (EXACT FORMAT REQUIRED):**
["Answer 1", "Answer 2"]

**Examples of correct fact-based answers:**
- "According to the record, hemoglobin is 8.5 g/dL."
- "Low-grade fever is mentioned; inflammatory markers are Not provided."
- "Not provided for whether the relevant test was performed."

**CRITICAL:** 
- Output ONLY the list format shown above
- No additional explanations or text
- Ensure answer count matches question count
- Use "Not provided" when information is unavailable"""

    def list_all_templates(self) -> Dict[str, List[str]]:
        """
        列出allavailabletemplate
        
        Returns:
            Dict[str, List[str]]: pertype分templateList
        """
        return {
            "evaluator_templates": list(self._evaluator_templates.keys()),
            "patient_templates": list(self._patient_templates.keys()),
            "expert_templates": [self.default_template_name],
            "specialty_clarification_templates": list(self._specialty_clarification_templates.keys())
        }

    def _extract_atomic_facts(self, case_data: Dict[str, Any]) -> str:
        """提取原子事实 - 统一字SegmentProcessing 
        
        PriorityProcessing ：
        1. 首先Checking is否存 facts 字Segment（来自 all_craft_md.jsonl）
        2. 如果没has facts，再Checking  atomic_facts 字Segment（来自 all_dev_convo.jsonl）
        3. 如果都没has，returnDefault提示
        """
        # 优先Checking  facts 字Segment（all_craft_md.jsonl Data集）
        facts = case_data.get('facts', [])
        
        # 如果没has facts 字Segment，Checking  atomic_facts 字Segment（all_dev_convo.jsonl Data集）
        if not facts:
            facts = case_data.get('atomic_facts', [])
        
        # 如果两字Segment都没hasoras空，returnDefault提示
        if not facts:
            return "No specific atomic facts provided."
        
        # 统一格式化Processing 
        formatted_facts = []
        for i, fact in enumerate(facts, 1):
            # Checking 事实is否已经withNumber开头（已经has编号）
            fact_str = str(fact).strip()
            if re.match(r'^\d+\.', fact_str):
                # 如果已经has编号，直接使用
                formatted_facts.append(fact_str)
            else:
                # 如果没has编号，添加编号
                formatted_facts.append(f"{i}. {fact_str}")
        
        return "\n".join(formatted_facts)

    def _format_clarification_context(self, clarification_context: List[Dict[str, Any]]) -> str:
        """
        格式化澄清for话Contextas自然语言格式
        
        Args:
            clarification_context: 澄清for话RecordList，eachElement包含questionandanswer
            
        Returns:
            str: 格式化后自然语言for话Record，多轮澄清用空行分开
        """
        if not clarification_context:
            return "No clarification dialogue available."
        
        formatted_dialogue = []
        
        for i, dialogue in enumerate(clarification_context):
            if isinstance(dialogue, dict):
                question = dialogue.get('question', '')
                answer = dialogue.get('answer', '')
                
                if question and answer:
                    # Format dialogue in English-only labels
                    dialogue_text = f"Doctor: {question}\nPatient: {answer}"
                    formatted_dialogue.append(dialogue_text)
            elif isinstance(dialogue, str):
                # 如果isString，直接添加
                formatted_dialogue.append(dialogue)
        
        # 多轮澄清用空行分开
        return "\n\n".join(formatted_dialogue)

    def _format_options(self, options: Dict[str, str]) -> str:
        """格式化Option"""
        formatted_options = []
        for key, value in options.items():
            formatted_options.append(f"{key}. {value}")
        return "\n".join(formatted_options)


# GlobalInstance
_global_template_manager = None

def get_unified_expert_template_manager(force_generic_cot: bool = False) -> UnifiedExpertTemplateManager:
    """Getting Globaltemplate管理器Instance"""
    global _global_template_manager
    if _global_template_manager is None or _global_template_manager.force_generic_cot != force_generic_cot:
        _global_template_manager = UnifiedExpertTemplateManager(force_generic_cot=force_generic_cot)
    return _global_template_manager


if __name__ == "__main__":
    # testOptimization后template管理器
    manager = get_unified_expert_template_manager()
    print("=== Optimization后Unified expert template managertest ===")
    print("✓ template管理器Initializing Successfully ")
    print("✓ 消除Duplicatestructure")
    print("✓ 思维链已移to<thinking>")
    print("✓ 添加JSONOutput提醒")
    print("✓ 添加'Let's think step by step'")