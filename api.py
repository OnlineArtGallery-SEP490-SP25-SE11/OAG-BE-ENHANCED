from flask import Blueprint, jsonify, request
from image_processor import ImageProcessor
# from image_processor_update import OptimizedImageProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('api', __name__)
logger.debug("API Blueprint created")

# Initialize image processor
image_processor = ImageProcessor()
# image_processor = OptimizedImageProcessor()
@bp.route('/compare', methods=['POST'])
def compare_images():
    """
    Compare query image with target images and return similar ones
    """
    logger.debug("Received request to /compare endpoint")
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        query_url = data.get('query_url')
        target_urls = data.get('target_urls', [])
        threshold = int(float(data.get('threshold', 50)))  # Convert to int for type consistency

        if not query_url:
            return jsonify({'error': 'Query URL is required'}), 400

        if not target_urls:
            return jsonify({'error': 'Target URLs are required'}), 400

        if not isinstance(target_urls, list):
            return jsonify({'error': 'Target URLs must be a list'}), 400

        if threshold < 0 or threshold > 100:
            return jsonify({'error': 'Threshold must be between 0 and 100'}), 400

        try:
            result = image_processor.find_similar_images(
                query_url,
                target_urls,
                threshold
            )

            response = {
                'query_url': query_url,
                'similar_images': result['similar_images']
            }

            # Only include failed_urls in response if there were any failures
            if result.get('failed_urls'):
                response['failed_urls'] = result['failed_urls']

            return jsonify(response)

        except ValueError as e:
            return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.debug("Received request to /health endpoint")
    return jsonify({'status': 'healthy'})

@bp.route('/swagger.json', methods=['GET'])
def swagger_spec():
    """OpenAPI specification endpoint"""
    logger.debug("Received request to /swagger.json endpoint")
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Image Similarity Search API",
            "version": "1.0.0",
            "description": "API for finding similar images using OpenCV"
        },
        "paths": {
            "/api/compare": {  # Include /api prefix in paths
                "post": {
                    "summary": "Compare images and find similar ones",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query_url": {
                                            "type": "string",
                                            "description": "URL of the query image"
                                        },
                                        "target_urls": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            },
                                            "description": "List of target image URLs"
                                        },
                                        "threshold": {
                                            "type": "number",
                                            "description": "Similarity threshold (0-100)"
                                        }
                                    },
                                    "required": ["query_url", "target_urls"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query_url": {
                                                "type": "string"
                                            },
                                            "similar_images": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "url": {
                                                            "type": "string"
                                                        },
                                                        "similarity_score": {
                                                            "type": "number"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/health": {  # Include /api prefix in paths
                "get": {
                    "summary": "Health check endpoint",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {
                                                "type": "string"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })