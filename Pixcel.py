import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import cv2
from typing import Tuple, Optional, Callable
import time

class SEEGPixelDetector:
    """
    Spooky-Entropy Early-Goal Explorer for Pixel Detection
    
    Finds pixels with specific characteristics in an image using self-limiting entropy exploration
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (height, width, channels)
        target_characteristics: dict = None,  # e.g., {'color': [255,0,0], 'intensity': >100}
        bins: int = 32,
        window_size: int = 100,
        T_phi: float = 0.1,
        learning_rate: float = 0.01,
        exploration_radius: int = 5
    ):
        self.height, self.width, self.channels = image_shape
        self.target_characteristics = target_characteristics or {}
        self.bins = bins
        self.window_size = window_size
        self.T_phi = T_phi
        self.learning_rate = learning_rate
        self.exploration_radius = exploration_radius
        
        # Current position (x, y) in image
        self.current_pos = np.array([self.width//2, self.height//2], dtype=float)
        self.position_history = []
        
        # SEEG components
        self.histogram = np.zeros(bins)
        self.graph_weights = self._compute_graph_weights()
        self.s_bar = np.log(bins) / bins  # Expected entropy density
        self.phi = np.zeros(bins)
        self.phi_bar = 0.0
        
        # Action space: movement directions
        self.action_space = np.array([
            [-1, -1], [-1, 0], [-1, 1],  # Up-left, up, up-right
            [0, -1],           [0, 1],   # Left, right  
            [1, -1], [1, 0], [1, 1]     # Down-left, down, down-right
        ])  # 8-directional movement
        
        # Policy parameters (for movement decisions)
        self.theta = torch.randn(8, requires_grad=True)  # 8 directions
        self.optimizer = torch.optim.Adam([self.theta], lr=learning_rate)
        
        # Tracking
        self.entropy_history = []
        self.phi_history = []
        self.found_targets = []
        
    def _compute_graph_weights(self) -> np.ndarray:
        """Compute graph weights for spooky field (simple adjacency)"""
        weights = np.zeros((self.bins, self.bins))
        for i in range(self.bins):
            for j in range(self.bins):
                weights[i, j] = 1.0 / (1.0 + abs(i - j))
        return weights
    
    def _pixel_to_bin(self, pixel_value: np.ndarray) -> int:
        """Convert pixel characteristics to histogram bin"""
        # Normalize pixel to 0-1 range then to bin
        if pixel_value.ndim == 1:  # Single pixel [R,G,B]
            norm_val = np.mean(pixel_value) / 255.0
        else:  # Multi-dimensional features
            norm_val = np.mean(pixel_value.flatten()) / 255.0
            
        bin_idx = int(norm_val * (self.bins - 1))
        return max(0, min(self.bins - 1, bin_idx))
    
    def _compute_entropy_proxy(self, pixel_value: np.ndarray) -> float:
        """Compute entropy proxy based on pixel value diversity"""
        bin_idx = self._pixel_to_bin(pixel_value)
        
        # Update histogram
        self.histogram[bin_idx] += 1
        
        # Apply sliding window (decay older entries)
        self.histogram *= (1.0 - 1.0 / self.window_size)
        
        # Compute entropy proxy
        total_count = np.sum(self.histogram)
        if total_count == 0:
            return 0.0
            
        probs = self.histogram / total_count
        probs = np.clip(probs, 1e-8, 1.0)
        entropy_proxy = -np.sum(probs * np.log(probs))
        
        return entropy_proxy
    
    def _compute_spooky_field(self) -> None:
        """Compute spooky inhibition field based on entropy density"""
        expected_per_bin = self.window_size / self.bins
        s = self.histogram / expected_per_bin if expected_per_bin > 0 else np.zeros(self.bins)
        
        # Compute Δ_i = Σ_j w_ij * (s_j - s̄)
        delta = np.zeros(self.bins)
        for i in range(self.bins):
            for j in range(self.bins):
                delta[i] += self.graph_weights[i, j] * (s[j] - self.s_bar)
        
        # Apply logistic: Φ_i = σ(Δ_i / T_Φ)
        self.phi = 1.0 / (1.0 + np.exp(-delta / self.T_phi))
        self.phi_bar = np.mean(self.phi)
    
    def _get_pixel_features(self, img: np.ndarray, x: int, y: int) -> np.ndarray:
        """Extract features from pixel at (x, y)"""
        # Get pixel value
        if len(img.shape) == 3:  # Color image
            pixel = img[y, x, :].astype(float)
        else:  # Grayscale
            pixel = np.array([img[y, x]]).astype(float)
        
        # Add neighborhood features
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                    if len(img.shape) == 3:
                        neighbors.extend(img[ny, nx, :])
                    else:
                        neighbors.append(img[ny, nx])
        
        # Combine pixel + neighborhood features
        features = np.concatenate([pixel, np.array(neighbors) if neighbors else pixel])
        return features
    
    def _check_target_match(self, pixel_value: np.ndarray, target_spec: dict) -> bool:
        """Check if pixel matches target characteristics"""
        if not target_spec:
            return False
            
        # Example target specifications:
        # {'color': [255, 0, 0], 'tolerance': 50} - red pixel
        # {'intensity': {'min': 200, 'max': 255}} - bright pixel  
        # {'gradient': {'min': 50}} - high gradient pixel
        
        if 'color' in target_spec:
            target_color = np.array(target_spec['color'])
            tolerance = target_spec.get('tolerance', 50)
            pixel_color = pixel_value[:3] if len(pixel_value) >= 3 else pixel_value
            distance = np.linalg.norm(pixel_color - target_color)
            if distance <= tolerance:
                return True
        
        if 'intensity' in target_spec:
            intensity_range = target_spec['intensity']
            intensity = np.mean(pixel_value) if len(pixel_value) > 0 else 0
            if 'min' in intensity_range and intensity < intensity_range['min']:
                return False
            if 'max' in intensity_range and intensity > intensity_range['max']:
                return False
            return True
            
        return False
    
    def get_action(self, img: np.ndarray) -> np.ndarray:
        """Get movement action based on current policy"""
        # Get current pixel features
        x, y = int(self.current_pos[0]), int(self.current_pos[1])
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        
        pixel_features = self._get_pixel_features(img, x, y)
        
        # Simple policy: use theta parameters to weight movement directions
        with torch.no_grad():
            action_weights = torch.softmax(self.theta, dim=0)
            action_idx = torch.multinomial(action_weights, 1).item()
        
        return self.action_space[action_idx]
    
    def update_policy(
        self, 
        img: np.ndarray, 
        x: int, 
        y: int, 
        reward: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """
        Update policy using SEEG algorithm
        
        Returns: (entropy_proxy, phi_bar, found_target)
        """
        # Get pixel features
        pixel_features = self._get_pixel_features(img, x, y)
        
        # Compute entropy proxy
        entropy_proxy = self._compute_entropy_proxy(pixel_features)
        
        # Check if this is a target pixel
        found_target = self._check_target_match(pixel_features, self.target_characteristics)
        
        # Compute reward if not provided
        if reward is None:
            if found_target:
                reward = 10.0  # High reward for finding target
                self.found_targets.append((x, y))
            else:
                reward = -0.1  # Small penalty for not finding target
        
        # Compute spooky field
        self.compute_spooky_field()
        
        # Compute surrogate gradient
        total_objective = reward + 0.5 * entropy_proxy  # Balance exploration + exploitation
        
        # Update policy with self-limiting
        self.optimizer.zero_grad()
        objective_tensor = torch.tensor(total_objective, requires_grad=True)
        
        # Simplified gradient computation (in practice would be more complex)
        fake_loss = -objective_tensor  # Minimize negative for maximization
        fake_loss.backward(retain_graph=True)
        
        # Apply self-limiting update: θ ← θ + ε · g · (1 - Φ̄)
        with torch.no_grad():
            if self.theta.grad is not None:
                self.theta.grad *= (1 - self.phi_bar)
        
        self.optimizer.step()
        
        # Track metrics
        self.entropy_history.append(entropy_proxy)
        self.phi_history.append(self.phi_bar)
        self.position_history.append((x, y))
        
        return entropy_proxy, self.phi_bar, found_target
    
    def compute_spooky_field(self):
        """Compute spooky field (separate method for clarity)"""
        expected_per_bin = self.window_size / self.bins
        s = self.histogram / expected_per_bin if expected_per_bin > 0 else np.zeros(self.bins)
        
        delta = np.zeros(self.bins)
        for i in range(self.bins):
            for j in range(self.bins):
                delta[i] += self.graph_weights[i, j] * (s[j] - self.s_bar)
        
        self.phi = 1.0 / (1.0 + np.exp(-delta / self.T_phi))
        self.phi_bar = np.mean(self.phi)
    
    def detect_target_pixels(self, img: np.ndarray, max_steps: int = 1000) -> list:
        """Main detection loop"""
        found_pixels = []
        
        for step in range(max_steps):
            # Get current position
            x, y = int(self.current_pos[0]), int(self.current_pos[1])
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            
            # Update policy and check for target
            entropy_proxy, phi_bar, found_target = self.update_policy(img, x, y)
            
            if found_target and (x, y) not in found_pixels:
                found_pixels.append((x, y))
                print(f"Found target pixel at ({x}, {y}) - Step {step}")
            
            # Get next action
            action = self.get_action(img)
            
            # Update position with action
            self.current_pos += action
            
            # Keep within image bounds
            self.current_pos[0] = np.clip(self.current_pos[0], 0, self.width - 1)
            self.current_pos[1] = np.clip(self.current_pos[1], 0, self.height - 1)
            
            # Occasionally add random exploration to avoid getting stuck
            if step % 50 == 0:
                self.current_pos += np.random.randn(2) * 2
        
        return found_pixels

# Example usage with real image data
def create_test_image():
    """Create a test image with specific target pixels"""
    # Create a 200x200 image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some background patterns
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            img[i:i+10, j:j+10, :] = [100, 100, 100]  # Gray patches
    
    # Add red target pixels (our search targets)
    target_positions = [(50, 50), (150, 150), (75, 125), (125, 75)]
    for x, y in target_positions:
        img[y-2:y+3, x-2:x+3, :] = [255, 0, 0]  # Red squares
    
    # Add some blue distractor pixels
    for x, y in [(25, 25), (175, 25), (25, 175)]:
        img[y-1:y+2, x-1:x+2, :] = [0, 0, 255]  # Blue pixels
    
    return img, target_positions

def run_pixel_detection():
    """Run SEEG pixel detection on test image"""
    print("Creating test image with target pixels...")
    test_img, true_targets = create_test_image()
    
    print(f"True target positions: {true_targets}")
    
    # Define target characteristics: red pixels
    target_spec = {
        'color': [255, 0, 0],  # Looking for red
        'tolerance': 50
    }
    
    # Initialize SEEG detector
    detector = SEEGPixelDetector(
        image_shape=test_img.shape,
        target_characteristics=target_spec,
        bins=32,
        window_size=100,
        learning_rate=0.01
    )
    
    print("Starting pixel detection with SEEG...")
    start_time = time.time()
    
    # Run detection
    detected_pixels = detector.detect_target_pixels(test_img, max_steps=500)
    
    elapsed_time = time.time() - start_time
    print(f"\nDetection completed in {elapsed_time:.2f} seconds")
    print(f"Detected {len(detected_pixels)} target pixels")
    print(f"True targets: {len(true_targets)}")
    
    # Calculate accuracy
    matched_targets = 0
    for det_x, det_y in detected_pixels:
        for true_x, true_y in true_targets:
            if abs(det_x - true_x) <= 5 and abs(det_y - true_y) <= 5:  # Within 5 pixels
                matched_targets += 1
                break
    
    accuracy = matched_targets / len(true_targets) if true_targets else 0
    print(f"Detection accuracy: {accuracy:.2f}")
    print(f"Detected pixels: {detected_pixels}")
    
    # Visualize results
    result_img = test_img.copy()
    
    # Mark true targets in green
    for x, y in true_targets:
        cv2.rectangle(result_img, (x-3, y-3), (x+3, y+3), [0, 255, 0], 1)
    
    # Mark detected pixels in yellow
    for x, y in detected_pixels:
        cv2.rectangle(result_img, (x-2, y-2), (x+2, y+2), [0, 255, 255], 1)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_img)
    axes[0].set_title('Original Image\n(Green=Targets, Blue=Distractors)')
    axes[0].axis('off')
    
    axes[1].imshow(result_img)
    axes[1].set_title(f'Detection Results\n(Yellow=Detected, Green=True)')
    axes[1].axis('off')
    
    axes[2].plot(detector.entropy_history, label='Entropy')
    axes[2].plot(detector.phi_history, label='Φ̄ (Inhibition)')
    axes[2].set_title('SEEG Metrics Over Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return detected_pixels, true_targets

# Run the actual detection
if __name__ == "__main__":
    detected, true_targets = run_pixel_detection()