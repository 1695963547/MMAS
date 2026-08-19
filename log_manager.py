#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理器 - 用于记录MMAS系统的完整运行日志

主要功能:
1. 记录完整的终端输出内容
2. 保存到指定目录下的.log文件
3. 支持实时日志写入和缓冲
4. 提供日志级别控制
5. 统一捕获print()和logging模块的输出
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO
import threading

class LogManagerFileHandler(logging.Handler):
    """自定义日志处理器，使用与LogCapture共享的文件句柄和锁，避免多句柄写同一文件导致输出交错
    
    关键修复：不能覆盖 self.lock（Handler基类的RLock，供logging框架handle()内部使用），
    否则handle()→acquire()获取锁后emit()内with self.lock再次获取同一不可重入Lock会死锁。
    改用独立属性 self._file_lock 存储LogManager共享锁。
    """
    
    def __init__(self, file_handle: TextIO, lock: threading.Lock):
        super().__init__()
        self.file_handle = file_handle
        self._file_lock = lock  # 不覆盖 self.lock（Handler基类的RLock）
    
    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            with self._file_lock:
                if self.file_handle and not self.file_handle.closed:
                    self.file_handle.write(msg)
                    self.file_handle.flush()
        except Exception:
            self.handleError(record)
    
    def close(self):
        """关闭处理器（不关闭文件句柄，由LogManager统一管理生命周期）"""
        pass


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "./log", 
                 log_prefix: str = "mmas_system",
                 capture_stdout: bool = True,
                 capture_stderr: bool = True,
                 resume_log_file: Optional[str] = None,
                 model_name: str = "",
                 dataset_name: str = ""):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录
            log_prefix: 日志文件前缀
            capture_stdout: 是否捕获标准输出
            capture_stderr: 是否捕获标准错误
            resume_log_file: 断点续传时要继续使用的日志文件路径
            model_name: 模型名称，用于日志文件名标识
            dataset_name: 数据集名称，用于日志文件名标识
        """
        self.log_dir = Path(log_dir)
        self.log_prefix = log_prefix
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定日志文件路径
        if resume_log_file and os.path.exists(resume_log_file):
            # 断点续传模式：使用现有日志文件
            self.log_file = Path(resume_log_file)
            self.is_resume_mode = True
        else:
            # 新建模式：生成新的日志文件名（包含模型名和数据集名）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 构建文件名：前缀_模型名_数据集名_时间戳.log
            name_parts = [log_prefix]
            if model_name:
                name_parts.append(model_name)
            if dataset_name:
                name_parts.append(dataset_name)
            name_parts.append(timestamp)
            file_name = "_".join(name_parts) + ".log"
            self.log_file = self.log_dir / file_name
            self.is_resume_mode = False
        
        # 保存日志文件路径供外部访问
        self.log_file_path = str(self.log_file)
        
        # 初始化日志文件
        self.log_handle: Optional[TextIO] = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 启动标志
        self.is_active = False
        
        # 保存原始的logging配置
        self.original_handlers = []
        
    def start_logging(self):
        """开始日志记录"""
        if self.is_active:
            return
            
        try:
            # 根据模式打开日志文件
            if self.is_resume_mode:
                # 断点续传模式：以追加模式打开
                self.log_handle = open(self.log_file, 'a', encoding='utf-8', buffering=1)
                # 写入续传标记
                self._write_resume_marker()
            else:
                # 新建模式：以写入模式打开
                self.log_handle = open(self.log_file, 'w', encoding='utf-8', buffering=1)
                # 写入日志头部信息
                self._write_log_header()
            
            # 重定向输出流
            if self.capture_stdout:
                sys.stdout = LogCapture(self.original_stdout, self.log_handle, self.lock)
            if self.capture_stderr:
                sys.stderr = LogCapture(self.original_stderr, self.log_handle, self.lock)
            
            # 配置logging模块，将其输出也重定向到日志文件
            self._setup_logging_handlers()
                
            self.is_active = True
            if self.is_resume_mode:
                print(f"日志记录已恢复，继续使用日志文件: {self.log_file}")
            else:
                print(f"日志记录已启动，日志文件: {self.log_file}")
            
        except Exception as e:
            print(f"启动日志记录失败: {e}")
            self._restore_streams()
    
    def _setup_logging_handlers(self):
        """设置logging模块的处理器，确保输出到日志文件
        
        关键修复：使用 LogManagerFileHandler 共享 self.log_handle 文件句柄和锁，
        不再创建独立的 FileHandler，避免两个句柄各自维护文件指针导致输出交错覆盖。
        """
        try:
            # 获取根日志记录器
            root_logger = logging.getLogger()
            
            # 保存原始处理器
            self.original_handlers = root_logger.handlers.copy()
            
            # 清除现有处理器
            root_logger.handlers.clear()
            
            # 使用自定义文件处理器，共享 self.log_handle 文件句柄和锁
            # 这样 print()（经LogCapture）和 logging 模块输出使用同一个文件句柄，
            # 通过共享锁串行化写入，彻底消除多句柄写同一文件的问题
            file_handler = LogManagerFileHandler(self.log_handle, self.lock)
            file_handler.setLevel(logging.DEBUG)
            
            # 创建控制台处理器（输出到原始stdout，不经过LogCapture，避免重复写入日志文件）
            console_handler = logging.StreamHandler(self.original_stdout)
            console_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            # 设置日志级别
            root_logger.setLevel(logging.DEBUG)
            
        except Exception as e:
            print(f"设置logging处理器失败: {e}")
    
    def stop_logging(self):
        """停止日志记录"""
        if not self.is_active:
            return
            
        try:
            # 写入日志尾部信息
            self._write_log_footer()
            
            # 恢复logging处理器
            self._restore_logging_handlers()
            
            # 恢复原始输出流
            self._restore_streams()
            
            self.is_active = False
            print(f"日志记录已停止，日志文件已保存到: {self.log_file}")
            
        except Exception as e:
            print(f"停止日志记录失败: {e}")
    
    def _restore_logging_handlers(self):
        """恢复原始的logging处理器"""
        try:
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            
            # 恢复原始处理器
            for handler in self.original_handlers:
                root_logger.addHandler(handler)
                
        except Exception as e:
            print(f"恢复logging处理器失败: {e}")
    
    def _restore_streams(self):
        """恢复原始输出流"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
    
    def _write_log_header(self):
        """写入日志头部信息"""
        if self.log_handle:
            header = f"""
{'='*80}
MMAS系统运行日志
启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
日志文件: {self.log_file}
{'='*80}

"""
            self.log_handle.write(header)
            self.log_handle.flush()
    
    def _write_resume_marker(self):
        """写入断点续传标记"""
        if self.log_handle:
            marker = f"""

{'='*80}
断点续传恢复
恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
继续处理剩余案例...
{'='*80}

"""
            self.log_handle.write(marker)
            self.log_handle.flush()
    
    def _write_log_footer(self):
        """写入日志尾部信息"""
        if self.log_handle:
            footer = f"""

{'='*80}
MMAS系统运行结束
结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
            self.log_handle.write(footer)
            self.log_handle.flush()
    
    def write_custom_log(self, message: str, level: str = "INFO"):
        """写入自定义日志消息"""
        if self.log_handle:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp} - CUSTOM - {level} - {message}\n"
            
            with self.lock:
                self.log_handle.write(log_entry)
                self.log_handle.flush()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_logging()


class LogCapture:
    """日志捕获类，用于同时输出到终端和日志文件"""
    
    def __init__(self, original_stream: TextIO, log_handle: TextIO, lock: threading.Lock):
        self.original_stream = original_stream
        self.log_handle = log_handle
        self.lock = lock
    
    def write(self, text: str):
        """写入文本到原始流和日志文件"""
        # 写入到原始流（终端）
        self.original_stream.write(text)
        self.original_stream.flush()
        
        # 写入到日志文件
        if self.log_handle:
            with self.lock:
                # 添加时间戳（仅对新行）
                if text.endswith('\n') and len(text.strip()) > 0:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    log_text = f"[{timestamp}] {text}"
                else:
                    log_text = text
                    
                self.log_handle.write(log_text)
                self.log_handle.flush()
    
    def flush(self):
        """刷新缓冲区"""
        self.original_stream.flush()
        if self.log_handle:
            self.log_handle.flush()
    
    def __getattr__(self, name):
        """代理其他属性到原始流"""
        return getattr(self.original_stream, name)


# 全局日志管理器实例
_log_manager_instance: Optional[LogManager] = None

def get_log_manager(log_dir: str = "./log", 
                   log_prefix: str = "mmas_system") -> LogManager:
    """获取全局日志管理器实例"""
    global _log_manager_instance
    
    if _log_manager_instance is None:
        _log_manager_instance = LogManager(log_dir=log_dir, log_prefix=log_prefix)
    
    return _log_manager_instance

def start_system_logging(log_dir: str = "./log", 
                        log_prefix: str = "mmas_system",
                        resume_log_file: Optional[str] = None,
                        model_name: str = "",
                        dataset_name: str = ""):
    """启动系统日志记录"""
    global _log_manager_instance
    _log_manager_instance = LogManager(log_dir, log_prefix, resume_log_file=resume_log_file,
                                       model_name=model_name, dataset_name=dataset_name)
    _log_manager_instance.start_logging()
    return _log_manager_instance

def stop_system_logging():
    """停止系统日志记录"""
    global _log_manager_instance
    if _log_manager_instance:
        _log_manager_instance.stop_logging()


if __name__ == "__main__":
    # 测试日志管理器
    print("测试日志管理器...")
    
    with LogManager() as log_manager:
        print("这是一条测试消息")
        print("这是另一条测试消息")
        log_manager.write_custom_log("这是自定义日志消息", "DEBUG")
        
    print("日志测试完成")