import cv2
import numpy as np
import requests
from io import BytesIO
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class ImageProcessor:
    def __init__(self, cache_size=100):
        self.image_cache = {}
        self.session = requests.Session()
        
    @lru_cache(maxsize=100)
    def load_image_from_url(self, url):
        """Load image from URL and convert to OpenCV format"""
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL format: {url}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Convert image to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"Failed to decode image from URL: {url}")

            return image
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching image from URL {url}: {str(e)}")
            raise ValueError(f"Failed to fetch image from URL {url}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing image from URL {url}: {str(e)}")
            raise ValueError(f"Failed to process image from URL {url}: {str(e)}")

    def calculate_hash(self, image, hash_size=8):
        """Calculate perceptual hash (pHash)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()
    
    def calculate_histogram(self, image):
        """Calculate color histogram for image using HSV color space"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                           [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compare_hashes(self, hash1, hash2):
        """Compare two hashes using Hamming distance"""
        return 100 - np.count_nonzero(hash1 != hash2) * 100 / len(hash1)

    def fast_compare_images(self, query_image, target_image):
        """Fast image comparison using only hash and histogram"""
        # Calculate perceptual hash
        hash1 = self.calculate_hash(query_image)
        hash2 = self.calculate_hash(target_image)
        hash_score = self.compare_hashes(hash1, hash2)
        
        # Early termination for clearly different images
        if hash_score < 40:
            return hash_score
        
        # Calculate histogram similarity
        hist1 = self.calculate_histogram(query_image)
        hist2 = self.calculate_histogram(target_image)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100
        
        # Final weighted score
        final_score = (hash_score * 0.6 + hist_score * 0.4)
        return final_score

    def process_target_image(self, query_image, target_url, threshold):
        """Process a single target image"""
        try:
            # Use cached image if available
            if target_url in self.image_cache:
                target_image = self.image_cache[target_url]
            else:
                target_image = self.load_image_from_url(target_url)
                self.image_cache[target_url] = target_image

            # Compare images
            similarity_score = self.fast_compare_images(query_image, target_image)

            if similarity_score >= threshold:
                return {
                    'url': target_url,
                    'similarity_score': round(similarity_score, 2),
                    'status': 'success'
                }
            return None

        except Exception as e:
            logging.warning(f"Error processing target image {target_url}: {str(e)}")
            return {
                'url': target_url,
                'error': str(e),
                'status': 'failed'
            }

    def find_similar_images(self, query_url, target_urls, threshold=50, max_workers=8):
        """Find similar images from a list of target URLs"""
        try:
            # Validate input
            if not isinstance(target_urls, list):
                raise ValueError("Target URLs must be a list")
            if not target_urls:
                raise ValueError("Target URLs list is empty")
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 100:
                raise ValueError("Threshold must be a number between 0 and 100")

            # Load query image
            try:
                query_image = self.load_image_from_url(query_url)
            except ValueError as e:
                raise ValueError(f"Error with query image: {str(e)}")

            results = []
            failed_urls = []

            # Process target images in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {
                    executor.submit(self.process_target_image, query_image, url, threshold): url
                    for url in target_urls
                }
                
                for future in future_to_url:
                    result = future.result()
                    if result:
                        if result['status'] == 'success':
                            results.append({
                                'url': result['url'],
                                'similarity_score': result['similarity_score']
                            })
                        else:
                            failed_urls.append({
                                'url': result['url'],
                                'error': result['error']
                            })

            return {
                'similar_images': sorted(results, key=lambda x: x['similarity_score'], reverse=True),
                'failed_urls': failed_urls
            }

        except Exception as e:
            logging.error(f"Error in find_similar_images: {str(e)}")
            raise