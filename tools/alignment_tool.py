import cv2
import numpy as np
from crewai.tools import BaseTool  # <-- 1. This is the correct import
import os

class AlignmentTool(BaseTool):
    name: str = "Image Alignment Tool"
    description: str = "Aligns scanned answer sheet with template using feature matching"
    
    def _run(self, template_path: str, student_sheet_path: str) -> dict:
        """
        Align student sheet to template and return transformation parameters
        """
        try:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            student = cv2.imread(student_sheet_path, cv2.IMREAD_GRAYSCALE)
            
            if template is None or student is None:
                raise Exception(f"Failed to load images. Check paths: {template_path}, {student_sheet_path}")
            
            # --- 2. We only need ONE step: Alignment ---
            # This single function handles skew, rotation, scale, and perspective.
            # All other steps were removed.
            aligned_image, transform_matrix, confidence = self._align_images(
                template, student
            )
            
            # --- 3. Better Error Handling ---
            # Check the confidence score. If it's bad, fail fast.
            if confidence < 0.5: # 50%
                raise Exception(f"Low alignment confidence: {confidence*100:.2f}%. "
                                "Features did not match well.")

            # --- Save the aligned image ---
            base_name = os.path.basename(student_sheet_path)
            file_name, file_ext = os.path.splitext(base_name)
            output_filename = f"{file_name}_aligned{file_ext}"
            
            # Create an 'outputs' dir if it doesn't exist
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, aligned_image)
            
            return {
                'success': True,
                'aligned_image_path': output_path,
                'transform_matrix': transform_matrix.tolist(),
                'confidence_score': float(confidence),
                'transformations_applied': {
                    'homography_alignment': True
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _align_images(self, template, image):
        """Align using ORB feature matching and return the aligned image"""
        
        orb = cv2.ORB_create(nfeatures=5000)
        
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(image, None)
        
        if des1 is None or des2 is None:
            raise Exception("Could not find features in one or both images.")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # --- 4. More Robust Filtering ---
        # Filter by distance, not a random percentage.
        DISTANCE_THRESHOLD = 70
        good_matches = [m for m in matches if m.distance < DISTANCE_THRESHOLD]
        
        if len(good_matches) < 10: # Need at least 4, but more is better
            raise Exception(f"Not enough good matches found ({len(good_matches)}). "
                            "Cannot compute homography.")

        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
        
        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        
        if h is None:
            raise Exception("Could not compute homography matrix.")

        confidence = np.sum(mask) / len(mask)
        
        height, width = template.shape
        aligned = cv2.warpPerspective(image, h, (width, height))
        
        return aligned, h, confidence