import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from image_processor import ImageProcessor
# from image_processor_update import OptimizedImageProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize image processor
image_processor = ImageProcessor()
# image_processor = OptimizedImageProcessor()

# Define request and response models
class CompareRequest(BaseModel):
    query_url: str = Field(..., description="URL of the query image")
    target_urls: List[str] = Field(..., description="List of target image URLs")
    threshold: float = Field(50, description="Similarity threshold (0-100)")

    @field_validator('threshold')
    def threshold_must_be_valid(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Threshold must be between 0 and 100')
        return v

class SimilarImage(BaseModel):
    url: str
    similarity_score: float

class CompareResponse(BaseModel):
    query_url: str
    similar_images: List[SimilarImage]
    failed_urls: Optional[List[str]] = None

@router.post("/compare", response_model=CompareResponse)
async def compare_images(request: CompareRequest):
    """
    Compare query image with target images and return similar ones
    """
    logger.debug("Received request to /api/compare endpoint")
    try:
        threshold = int(request.threshold)  # Convert to int for type consistency
        
        result = image_processor.find_similar_images(
            request.query_url,
            request.target_urls,
            threshold
        )

        response = CompareResponse(
            query_url=request.query_url,
            similar_images=[SimilarImage(url=item['url'], similarity_score=item['similarity_score']) 
                           for item in result['similar_images']],
            failed_urls=result.get('failed_urls')
        )
        
        return response

    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Received request to /api/health endpoint")
    return {"status": "healthy"}

# Error handling middleware can be registered at the app level in app.py if needed

# In FastAPI, we don't need to explicitly define the swagger.json endpoint
# as it's automatically generated and available at /openapi.json