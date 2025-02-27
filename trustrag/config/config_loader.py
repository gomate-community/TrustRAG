#!/usr/bin/env python
# -*- coding:utf-8 _*-

import json
import os
from typing import Dict, Any

class ConfigLoader:
    """配置加载器，用于加载和管理全局配置"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """单例模式，确保只有一个配置实例"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置加载器"""
        if self._config is None:
            self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到：{config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件格式错误：{config_path}")
    
    def get_config(self, key: str = None) -> Dict[str, Any]:
        """获取配置信息
        
        Args:
            key: 配置键名，如 'models.reranker.path'
            
        Returns:
            如果指定了key，返回对应的配置值；否则返回整个配置字典
        """
        if key is None:
            return self._config
            
        keys = key.split('.')
        value = self._config
        for k in keys:
            try:
                value = value[k]
            except (KeyError, TypeError):
                raise KeyError(f"配置项未找到：{key}")
        return value

# 创建全局配置实例
config = ConfigLoader()
