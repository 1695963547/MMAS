"""
API配置管理模块
管理不同LLM API的密钥和配置信息

所有密钥通过环境变量传入，请勿硬编码。
环境变量示例：
    export DEEPSEEK_API_KEY="your_key_here"
    export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
    export GLM_API_KEY="your_key_here"
    export GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
"""

import os
from typing import Dict, Optional

# API配置字典
# 论文中使用两个API模型（OpenAI兼容格式）：
#   - DeepSeek-V3.2 (671B)：官方API https://api.deepseek.com/v1
#   - GLM-5.1 (754B)：官方API https://open.bigmodel.cn/api/paas/v4/
API_CONFIGS = {
    "deepseek-chat": {
        "provider": "openai",
        "model_name": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "max_tokens": 8192,
        "temperature": 0.7,
        "timeout": 120
    },
    "glm-5.1": {
        "provider": "openai",
        "model_name": "glm-5.1",
        "api_key": os.getenv("GLM_API_KEY", ""),
        "base_url": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        "max_tokens": 8192,
        "temperature": 0.7,
        "timeout": 120
    },
}

def get_api_config(model_name: str) -> Optional[Dict]:
    """获取指定模型的API配置"""
    return API_CONFIGS.get(model_name)

def get_all_api_models() -> list:
    """获取所有可用的API模型列表"""
    return list(API_CONFIGS.keys())

def update_api_key(model_name: str, api_key: str) -> bool:
    """更新指定模型的API密钥"""
    if model_name in API_CONFIGS:
        API_CONFIGS[model_name]["api_key"] = api_key
        return True
    return False

def validate_api_config(model_name: str) -> bool:
    """验证API配置是否完整"""
    config = get_api_config(model_name)
    if not config:
        return False
    
    required_fields = ["provider", "model_name", "api_key", "base_url"]
    return all(config.get(field) for field in required_fields)
