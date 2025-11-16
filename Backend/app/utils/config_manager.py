"""
Configuration Manager

Handles loading and accessing configuration from multiple sources:
- Environment variables
- JSON configuration files
- Environment-specific overrides
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for the application.
    
    Loads configuration from:
    1. Base settings.json
    2. Environment-specific config files
    3. Environment variables (highest priority)
    """
    
    def __init__(self, environment: str = 'development'):
        """
        Initialize configuration manager.
        
        Args:
            environment: Environment name (development, production, testing)
        """
        self.environment = environment
        self.config_dir = Path(__file__).parent.parent.parent / 'config'
        
        # Storage for configuration data
        self._config: Dict[str, Any] = {}
        self._messages: Dict[str, Any] = {}
        
        # Load configurations
        self._load_configurations()
        
        logger.info(f"Configuration manager initialized for {environment} environment")
    
    def _load_configurations(self) -> None:
        """Load all configuration files."""
        
        # Load base settings
        self._load_json_file('settings.json', self._config)
        
        # Load environment-specific settings
        env_file = f'environments/{self.environment}.json'
        env_config = {}
        self._load_json_file(env_file, env_config)
        
        # Merge environment config (overrides base settings)
        self._deep_merge(self._config, env_config)
        
        # Load messages
        self._load_json_file('messages.json', self._messages)
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _load_json_file(self, filename: str, target_dict: Dict[str, Any]) -> None:
        """
        Load JSON configuration file.
        
        Args:
            filename: Relative path to config file
            target_dict: Dictionary to load data into
        """
        file_path = self.config_dir / filename
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    target_dict.update(data)
                    logger.debug(f"Loaded configuration from {filename}")
            else:
                logger.warning(f"Configuration file not found: {filename}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override dictionary into base dictionary.
        
        Args:
            base: Base dictionary to merge into
            override: Override values
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        
        # Map environment variables to config paths
        env_mappings = {
            'DEBUG': 'debug',
            'HOST': 'api.host',
            'PORT': 'api.port',
            'LOG_LEVEL': 'logging.level',
            'CONFIDENCE_THRESHOLD': 'detection.thresholds.confidence.default',
            'RATE_LIMIT_ENABLED': 'api.rate_limiting.enabled',
            'CACHE_ENABLED': 'api.caching.enabled',
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                self._set_nested_value(self._config, config_path, converted_value)
    
    def _convert_env_value(self, value: str) -> Union[str, bool, int, float]:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable value as string
            
        Returns:
            Converted value
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """
        Set nested configuration value using dot notation path.
        
        Args:
            config: Configuration dictionary
            path: Dot notation path (e.g., 'api.rate_limiting.enabled')
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            path: Dot notation path to configuration value
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = path.split('.')
            current = self._config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception as e:
            logger.warning(f"Error accessing config path '{path}': {e}")
            return default
    
    def get_message(self, path: str, default: str = None, **kwargs) -> str:
        """
        Get message string with optional formatting.
        
        Args:
            path: Dot notation path to message
            default: Default message if path not found
            **kwargs: Formatting arguments
            
        Returns:
            Formatted message string
        """
        try:
            keys = path.split('.')
            current = self._messages
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    message = default or f"Message not found: {path}"
                    return message.format(**kwargs) if kwargs else message
            
            if isinstance(current, str):
                return current.format(**kwargs) if kwargs else current
            else:
                message = default or f"Invalid message type at: {path}"
                return message.format(**kwargs) if kwargs else message
                
        except Exception as e:
            logger.warning(f"Error accessing message path '{path}': {e}")
            message = default or f"Error loading message: {path}"
            return message.format(**kwargs) if kwargs else message
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            True if feature is enabled
        """
        return self.get(f'detection.features.enable_{feature_name}', False)
    
    def get_threshold(self, threshold_type: str, level: str = 'default') -> float:
        """
        Get detection threshold value.
        
        Args:
            threshold_type: Type of threshold (confidence, severity, etc.)
            level: Threshold level (default, strict, lenient)
            
        Returns:
            Threshold value
        """
        return self.get(f'detection.thresholds.{threshold_type}.{level}', 0.7)
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration (for debugging).
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._config.clear()
        self._messages.clear()
        self._load_configurations()
        logger.info("Configuration reloaded")
    
    def validate(self) -> bool:
        """
        Validate configuration completeness.
        
        Returns:
            True if configuration is valid
        """
        required_keys = [
            'detection.thresholds.confidence.default',
            'api.rate_limiting',
            'logging.level'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True
