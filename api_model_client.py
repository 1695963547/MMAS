"""
API模型客户端
用于调用不同的LLM API服务
"""

import requests
import json
import time
import logging
from typing import Dict, Any, Optional, List, Generator
from api_config import get_api_config, get_all_api_models, validate_api_config

logger = logging.getLogger(__name__)

class APIModelClient:
    """API模型客户端"""
    
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_BASE_DELAY = 2  # 基础重试延迟（秒）
    
    def __init__(self):
        self.session = requests.Session()
    
    def generate_with_messages(self, 
                              model_name: str,
                              messages: List[Dict[str, str]],
                              max_new_tokens: int = 2048,
                              temperature: float = 0.7,
                              do_sample: bool = True,
                              top_p: float = 0.9,
                              top_k: int = 50,
                              repetition_penalty: float = 1.1,
                              length_penalty: float = 1.0,
                              **kwargs) -> Dict[str, Any]:
        """
        使用消息格式生成文本响应
        
        Args:
            model_name: 模型名称
            messages: 消息列表，格式为 [{"role": "system/user", "content": "..."}]
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            do_sample: 是否采样
            top_p: top-p采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            
        Returns:
            包含生成结果的字典
        """
        config = get_api_config(model_name)
        if not config:
            raise ValueError(f"未找到模型 {model_name} 的配置")
        
        if not validate_api_config(model_name):
            raise ValueError(f"模型 {model_name} 的配置无效")
        
        try:
            # 根据不同的API提供商调用相应的方法
            provider = config.get('provider', '').lower()
            if provider == 'groq' or "llama" in model_name.lower():
                return self._call_groq_api_with_messages(config, messages, max_new_tokens, temperature, **kwargs)
            elif provider in ('openai', 'custom') or "gpt" in model_name.lower():
                # OpenAI兼容格式（包括自定义API网关）
                return self._call_openai_api_with_messages(config, messages, max_new_tokens, temperature, **kwargs)
            else:
                # 默认使用OpenAI兼容格式
                logger.info(f"未知provider '{provider}'，使用OpenAI兼容格式调用: {model_name}")
                return self._call_openai_api_with_messages(config, messages, max_new_tokens, temperature, **kwargs)
                
        except Exception as e:
            logger.error(f"API调用失败 {model_name}: {e}")
            raise

    def generate(self, 
                 model_name: str,
                 prompt: str,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.7,
                 do_sample: bool = True,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.1,
                 length_penalty: float = 1.0,
                 **kwargs) -> Dict[str, Any]:
        """
        生成文本响应
        
        Args:
            model_name: 模型名称
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            do_sample: 是否采样
            top_p: top-p采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            
        Returns:
            包含生成结果的字典
        """
        config = get_api_config(model_name)
        if not config:
            raise ValueError(f"未找到模型 {model_name} 的配置")
        
        if not validate_api_config(model_name):
            raise ValueError(f"模型 {model_name} 的配置无效")
        
        try:
            # 根据不同的API提供商调用相应的方法
            provider = config.get('provider', '').lower()
            if provider == 'groq' or "llama" in model_name.lower():
                return self._call_groq_api(config, prompt, max_new_tokens, temperature, **kwargs)
            elif provider in ('openai', 'custom') or "gpt" in model_name.lower():
                # OpenAI兼容格式（包括自定义API网关）
                return self._call_openai_api(config, prompt, max_new_tokens, temperature, **kwargs)
            else:
                # 默认使用OpenAI兼容格式
                logger.info(f"未知provider '{provider}'，使用OpenAI兼容格式调用: {model_name}")
                return self._call_openai_api(config, prompt, max_new_tokens, temperature, **kwargs)
                
        except Exception as e:
            logger.error(f"API调用失败 {model_name}: {e}")
            raise
    
    def _call_openai_api(self, 
                        config: Dict[str, Any], 
                        prompt: str, 
                        max_tokens: int, 
                        temperature: float,
                        **kwargs) -> Dict[str, Any]:
        """调用OpenAI API，带指数退避重试"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": config['model_name'],
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        # GLM-5.x 系列默认开启思考链(reasoning_content)，消耗大量token且干扰输出
        # 通过 thinking: {type: disabled} 关闭，仅对 GLM-5 模型生效
        if "glm-5" in config['model_name'].lower():
            data["thinking"] = {"type": "disabled"}
        
        url = f"{config['base_url']}/chat/completions"
        timeout = config.get('timeout', 60)
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API错误响应: {response.status_code}")
                    logger.error(f"请求URL: {url}")
                    logger.error(f"响应内容: {response.text}")
                    if response.status_code in (429, 500, 502, 503, 504):
                        delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(f"API返回{response.status_code}，第{attempt+1}次重试，等待{delay}秒...")
                        time.sleep(delay)
                        continue
                
                response.raise_for_status()
                
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                return {
                    "generated_text": generated_text,
                    "full_response": generated_text,
                    "usage": result.get("usage", {}),
                    "model": config['model_name'],
                    "api_provider": "openai"
                }
                
            except requests.exceptions.Timeout as e:
                last_error = e
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"API请求超时，第{attempt+1}次重试，等待{delay}秒...")
                time.sleep(delay)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"API连接错误，第{attempt+1}次重试，等待{delay}秒...")
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI API请求失败: {e}")
                raise
            except KeyError as e:
                logger.error(f"OpenAI API响应格式错误: {e}")
                raise
        
        logger.error(f"API调用在{self.MAX_RETRIES}次重试后仍然失败")
        raise last_error or RuntimeError(f"API调用失败，已重试{self.MAX_RETRIES}次")
    
    def _call_groq_api_with_messages(self, 
                                    config: Dict[str, Any], 
                                    messages: List[Dict[str, str]], 
                                    max_tokens: int, 
                                    temperature: float,
                                    **kwargs) -> Dict[str, Any]:
        """使用消息格式调用Groq API"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config['model_name'],
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
            # 移除强制JSON模式，让模型按照模板要求输出thinking块和JSON
        }
        
        url = f"{config['base_url']}/chat/completions"
        
        try:
            response = self.session.post(
                url, 
                headers=headers, 
                json=data, 
                timeout=config.get('timeout', 60)
            )
            
            # 打印详细的错误信息用于调试
            if response.status_code != 200:
                logger.error(f"Groq API请求失败: {response.status_code}")
                logger.error(f"请求URL: {url}")
                logger.error(f"请求数据: {data}")
                logger.error(f"响应状态码: {response.status_code}")
                logger.error(f"响应内容: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            
            # 提取生成的文本
            generated_text = result["choices"][0]["message"]["content"]
            
            # 构建返回格式，模拟本地模型的输出格式
            return {
                "generated_text": generated_text,
                "full_response": generated_text,
                "usage": result.get("usage", {}),
                "model": result.get("model", config['model_name'])
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API请求异常: {e}")
            raise
        except Exception as e:
            logger.error(f"Groq API调用失败: {e}")
            raise

    def _call_openai_api_with_messages(self, 
                                      config: Dict[str, Any], 
                                      messages: List[Dict[str, str]], 
                                      max_tokens: int, 
                                      temperature: float,
                                      **kwargs) -> Dict[str, Any]:
        """使用消息格式调用OpenAI API，带指数退避重试"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config['model_name'],
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        # GLM-5.x 系列默认开启思考链(reasoning_content)，消耗大量token且干扰输出
        # 通过 thinking: {type: disabled} 关闭，仅对 GLM-5 模型生效
        if "glm-5" in config['model_name'].lower():
            data["thinking"] = {"type": "disabled"}
        
        url = f"{config['base_url']}/chat/completions"
        timeout = config.get('timeout', 60)
        
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.post(
                    url, 
                    headers=headers, 
                    json=data, 
                    timeout=timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API错误响应: {response.status_code}")
                    logger.error(f"请求URL: {url}")
                    logger.error(f"响应内容: {response.text}")
                    # 5xx 服务器错误和 429 限流可重试
                    if response.status_code in (429, 500, 502, 503, 504):
                        delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(f"API返回{response.status_code}，第{attempt+1}次重试，等待{delay}秒...")
                        time.sleep(delay)
                        continue
                
                response.raise_for_status()
                
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                return {
                    "generated_text": generated_text,
                    "full_response": generated_text,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", config['model_name'])
                }
                
            except requests.exceptions.Timeout as e:
                last_error = e
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"API请求超时，第{attempt+1}次重试，等待{delay}秒...")
                time.sleep(delay)
            except requests.exceptions.ConnectionError as e:
                last_error = e
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"API连接错误，第{attempt+1}次重试，等待{delay}秒...")
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenAI API请求异常: {e}")
                raise
            except KeyError as e:
                logger.error(f"OpenAI API响应格式错误: {e}")
                raise
            except Exception as e:
                logger.error(f"OpenAI API调用失败: {e}")
                raise
        
        # 所有重试均失败
        logger.error(f"API调用在{self.MAX_RETRIES}次重试后仍然失败")
        raise last_error or RuntimeError(f"API调用失败，已重试{self.MAX_RETRIES}次")

    def _call_groq_api(self, 
                      config: Dict[str, Any], 
                      prompt: str, 
                      max_tokens: int, 
                      temperature: float,
                      **kwargs) -> Dict[str, Any]:
        """调用Groq API (用于Llama模型)"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # 构建消息格式
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": config['model_name'],
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
            # 移除强制JSON模式，让模型按照模板要求输出thinking块和JSON
        }
        
        url = f"{config['base_url']}/chat/completions"
        
        try:
            response = self.session.post(
                url, 
                headers=headers, 
                json=data, 
                timeout=config.get('timeout', 60)
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取生成的文本
            generated_text = result["choices"][0]["message"]["content"]
            
            # 构建返回格式，模拟本地模型的输出格式
            return {
                "generated_text": generated_text,
                "full_response": generated_text,
                "usage": result.get("usage", {}),
                "model": config['model_name'],
                "api_provider": "groq"
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API请求失败: {e}")
            logger.error(f"请求URL: {url}")
            logger.error(f"请求数据: {data}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"响应状态码: {e.response.status_code}")
                logger.error(f"响应内容: {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Groq API响应格式错误: {e}")
            raise
    
    def test_connection(self, model_name: str) -> bool:
        """测试API连接"""
        try:
            # 发送一个简单的测试请求
            result = self.generate(
                model_name=model_name,
                prompt="Hello, this is a test message. Please respond with 'Test successful.'",
                max_new_tokens=50,
                temperature=0.1
            )
            
            if result and "generated_text" in result:
                logger.info(f"API连接测试成功: {model_name}")
                return True
            else:
                logger.error(f"API连接测试失败: {model_name} - 无效响应")
                return False
                
        except Exception as e:
            logger.error(f"API连接测试失败: {model_name} - {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """获取所有可用的API模型"""
        return get_all_api_models()
    
    def is_api_model(self, model_name: str) -> bool:
        """检查是否为API模型"""
        return model_name in self.get_available_models()
    
    def call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, 
                 max_tokens: int = 2048, model_name: str = None, **kwargs) -> str:
        """统一的LLM调用接口（供患者智能体等使用）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            model_name: 模型名称，如果为None则使用第一个可用的API模型
            
        Returns:
            生成的文本字符串
        """
        if model_name is None:
            available = self.get_available_models()
            if not available:
                raise ValueError("没有可用的API模型")
            model_name = available[0]
        
        result = self.generate_with_messages(
            model_name=model_name,
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return result.get('generated_text', '')

# 全局API客户端实例
_api_client = None

def get_api_client() -> APIModelClient:
    """获取全局API客户端实例"""
    global _api_client
    if _api_client is None:
        _api_client = APIModelClient()
    return _api_client