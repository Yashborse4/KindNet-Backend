"""
Response Builder Utility

Provides consistent response formatting across the API with support for:
- Success responses
- Error responses
- Validation error responses
- Rate limiting responses
"""

from datetime import datetime
from typing import Any, Dict, Optional, List
from flask import jsonify, Response, current_app
from pydantic import ValidationError

from app import get_config_manager


class ResponseBuilder:
    """Builder class for consistent API responses."""
    
    def __init__(self):
        """Initialize response builder."""
        self.config_manager = None
    
    def _ensure_config_manager(self):
        """Ensure config manager is available."""
        if self.config_manager is None:
            self.config_manager = get_config_manager()
    
    def _get_message(self, message_key: str, default: str = None, **kwargs) -> str:
        """Get message from config with formatting."""
        self._ensure_config_manager()
        
        if self.config_manager:
            return self.config_manager.get_message(message_key, default, **kwargs)
        else:
            return default or message_key
    
    def success(
        self, 
        data: Any, 
        message_key: str = None, 
        status_code: int = 200,
        **message_kwargs
    ) -> Response:
        """
        Create success response.
        
        Args:
            data: Response data
            message_key: Message key for localization
            status_code: HTTP status code
            **message_kwargs: Message formatting arguments
            
        Returns:
            Flask Response object
        """
        response_data = {
            'success': True,
            'data': data,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if message_key:
            message = self._get_message(message_key, **message_kwargs)
            response_data['message'] = message
        
        return jsonify(response_data), status_code
    
    def error(
        self, 
        message_key: str, 
        status_code: int = 400, 
        error_code: str = None,
        details: Dict[str, Any] = None,
        **message_kwargs
    ) -> Response:
        """
        Create error response.
        
        Args:
            message_key: Error message key
            status_code: HTTP status code
            error_code: Specific error code
            details: Additional error details
            **message_kwargs: Message formatting arguments
            
        Returns:
            Flask Response object
        """
        error_message = self._get_message(message_key, **message_kwargs)
        
        response_data = {
            'success': False,
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if error_code:
            response_data['error_code'] = error_code
        
        if details:
            response_data['details'] = details
        
        return jsonify(response_data), status_code
    
    def validation_error(self, validation_error: ValidationError) -> Response:
        """
        Create validation error response from Pydantic ValidationError.
        
        Args:
            validation_error: Pydantic validation error
            
        Returns:
            Flask Response object
        """
        # Extract validation errors
        error_details = []
        for error in validation_error.errors():
            error_detail = {
                'field': '.'.join(str(x) for x in error['loc']),
                'message': error['msg'],
                'type': error['type']
            }
            
            if 'input' in error:
                error_detail['input'] = error['input']
            
            error_details.append(error_detail)
        
        response_data = {
            'success': False,
            'error': 'Validation failed',
            'error_code': 'VALIDATION_ERROR',
            'details': {
                'validation_errors': error_details,
                'error_count': len(error_details)
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response_data), 400
    
    def rate_limit_exceeded(self) -> Response:
        """
        Create rate limit exceeded response.
        
        Returns:
            Flask Response object
        """
        message = self._get_message(
            'api.responses.errors.rate_limit_exceeded',
            'Rate limit exceeded. Please try again later.'
        )
        
        response_data = {
            'success': False,
            'error': message,
            'error_code': 'RATE_LIMIT_EXCEEDED',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response_data), 429
    
    def not_found(self, resource: str = 'Resource') -> Response:
        """
        Create not found response.
        
        Args:
            resource: Name of the resource that was not found
            
        Returns:
            Flask Response object
        """
        message = self._get_message(
            'api.responses.errors.not_found',
            f'{resource} not found',
            resource=resource
        )
        
        response_data = {
            'success': False,
            'error': message,
            'error_code': 'NOT_FOUND',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response_data), 404
    
    def method_not_allowed(self, method: str, allowed_methods: List[str]) -> Response:
        """
        Create method not allowed response.
        
        Args:
            method: HTTP method that was used
            allowed_methods: List of allowed methods
            
        Returns:
            Flask Response object
        """
        message = self._get_message(
            'api.responses.errors.method_not_allowed',
            f'Method {method} not allowed. Allowed methods: {", ".join(allowed_methods)}',
            method=method,
            allowed_methods=", ".join(allowed_methods)
        )
        
        response_data = {
            'success': False,
            'error': message,
            'error_code': 'METHOD_NOT_ALLOWED',
            'details': {
                'method': method,
                'allowed_methods': allowed_methods
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        response = jsonify(response_data)
        response.headers['Allow'] = ', '.join(allowed_methods)
        
        return response, 405
    
    def internal_server_error(self, error_id: str = None) -> Response:
        """
        Create internal server error response.
        
        Args:
            error_id: Error ID for tracking
            
        Returns:
            Flask Response object
        """
        message = self._get_message(
            'api.responses.errors.internal_server_error',
            'An internal server error occurred'
        )
        
        response_data = {
            'success': False,
            'error': message,
            'error_code': 'INTERNAL_SERVER_ERROR',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if error_id:
            response_data['error_id'] = error_id
        
        return jsonify(response_data), 500
    
    def service_unavailable(self, service_name: str = 'Service') -> Response:
        """
        Create service unavailable response.
        
        Args:
            service_name: Name of the unavailable service
            
        Returns:
            Flask Response object
        """
        message = self._get_message(
            'api.responses.errors.service_unavailable',
            f'{service_name} is currently unavailable',
            service_name=service_name
        )
        
        response_data = {
            'success': False,
            'error': message,
            'error_code': 'SERVICE_UNAVAILABLE',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response_data), 503
    
    def paginated_response(
        self, 
        data: List[Any], 
        page: int, 
        page_size: int, 
        total_items: int,
        message_key: str = None
    ) -> Response:
        """
        Create paginated response.
        
        Args:
            data: Paginated data
            page: Current page number
            page_size: Items per page
            total_items: Total number of items
            message_key: Success message key
            
        Returns:
            Flask Response object
        """
        total_pages = (total_items + page_size - 1) // page_size
        
        pagination_info = {
            'page': page,
            'page_size': page_size,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
        
        response_data = {
            'success': True,
            'data': data,
            'pagination': pagination_info,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if message_key:
            message = self._get_message(message_key)
            response_data['message'] = message
        
        return jsonify(response_data), 200
