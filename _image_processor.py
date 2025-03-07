import cv2
import numpy as np
import requests
from io import BytesIO
import logging

class ImageProcessor:
    def __init__(self):
        self.image_cache = {}

    def load_image_from_url(self, url):
        """Load image from URL and convert to OpenCV format"""
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL format: {url}")

            response = requests.get(url)
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

    def calculate_phash(self, image, hash_size=8):
        """Calculate perceptual hash (pHash)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()

    def calculate_ahash(self, image, hash_size=8):
        """Calculate average hash (aHash)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size, hash_size))
        avg = resized.mean()
        return resized > avg

    def calculate_dhash(self, image, hash_size=8):
        """Calculate difference hash (dHash)"""
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

    def extract_sift_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def extract_orb_features(self, image):
        """Extract ORB features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def compare_hashes(self, hash1, hash2):
        """Compare two hashes using Hamming distance"""
        return 100 - np.count_nonzero(hash1 != hash2) * 100 / len(hash1)

    def compare_images(self, query_image, target_image):
        """Compare two images using multiple methods"""
        # Hash-based comparison (30%)
        phash1 = self.calculate_phash(query_image)
        phash2 = self.calculate_phash(target_image)
        phash_score = self.compare_hashes(phash1, phash2)

        ahash1 = self.calculate_ahash(query_image)
        ahash2 = self.calculate_ahash(target_image)
        ahash_score = self.compare_hashes(ahash1, ahash2)

        dhash1 = self.calculate_dhash(query_image)
        dhash2 = self.calculate_dhash(target_image)
        dhash_score = self.compare_hashes(dhash1, dhash2)

        hash_score = (phash_score + ahash_score + dhash_score) / 3

        # Histogram comparison (30%)
        hist1 = self.calculate_histogram(query_image)
        hist2 = self.calculate_histogram(target_image)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100

        # Feature matching (40%)
        # SIFT features (20%)
        kp1_sift, desc1_sift = self.extract_sift_features(query_image)
        kp2_sift, desc2_sift = self.extract_sift_features(target_image)

        # ORB features (20%)
        kp1_orb, desc1_orb = self.extract_orb_features(query_image)
        kp2_orb, desc2_orb = self.extract_orb_features(target_image)

        feature_score = 0

        if desc1_sift is not None and desc2_sift is not None:
            # SIFT matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches_sift = flann.knnMatch(desc1_sift, desc2_sift, k=2)
            good_matches_sift = 0
            try:
                for m, n in matches_sift:
                    if m.distance < 0.7 * n.distance:
                        good_matches_sift += 1
                sift_score = (good_matches_sift / len(desc1_sift)) * 100
            except ValueError:
                sift_score = 0
        else:
            sift_score = 0

        if desc1_orb is not None and desc2_orb is not None:
            # ORB matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches_orb = bf.match(desc1_orb, desc2_orb)
            matches_orb = sorted(matches_orb, key=lambda x: x.distance)

            # Calculate ORB score based on average distance of best matches
            num_good_matches = min(len(matches_orb), 50)  # Use top 50 matches
            if num_good_matches > 0:
                avg_distance = sum(m.distance for m in matches_orb[:num_good_matches]) / num_good_matches
                orb_score = max(0, 100 - (avg_distance / 2))  # Convert distance to similarity score
            else:
                orb_score = 0
        else:
            orb_score = 0

        feature_score = (sift_score + orb_score) / 2

        # Final weighted score
        final_score = (0.3 * hash_score + 0.3 * hist_score + 0.4 * feature_score)
        return max(0, min(100, final_score))  # Ensure score is between 0 and 100

    def find_similar_images(self, query_url, target_urls, threshold=50):
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

            for target_url in target_urls:
                try:
                    # Use cached image if available
                    if target_url in self.image_cache:
                        target_image = self.image_cache[target_url]
                    else:
                        target_image = self.load_image_from_url(target_url)
                        self.image_cache[target_url] = target_image

                    similarity_score = self.compare_images(query_image, target_image)

                    if similarity_score >= threshold:
                        results.append({
                            'url': target_url,
                            'similarity_score': round(similarity_score, 2)
                        })

                except Exception as e:
                    logging.warning(f"Error processing target image {target_url}: {str(e)}")
                    failed_urls.append({
                        'url': target_url,
                        'error': str(e)
                    })
                    continue

            return {
                'similar_images': sorted(results, key=lambda x: x['similarity_score'], reverse=True),
                'failed_urls': failed_urls
            }

        except Exception as e:
            logging.error(f"Error in find_similar_images: {str(e)}")
            raise