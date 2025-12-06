import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2
from typing import Tuple, Optional
import time

class MammogramSEEGDetector:
    """
    SEEG for Mammogram Analysis - Detecting suspicious regions
    """
    
    def __init__(
        self,
        image_shape: Tuple[int, int],
        bins: int = 32,
        window_size: int = 100,
        T_phi: float = 0.1,
        learning_rate: float = 0.01
    ):
        self.height, self.width = image_shape
        self.bins = bins
        self.window_size = window_size
        self.T_phi = T_phi
        self.learning_rate = learning_rate
        
        # Current position in image
        self.current_pos = np.array([self.width//2, self.height//2], dtype=float)
        self.position_history = []
        
        # SEEG components
        self.histogram = np.zeros(bins)
        self.graph_weights = self._compute_graph_weights()
        self.s_bar = np.log(bins) / bins
        self.phi = np.zeros(bins)
        self.phi_bar = 0.0
        
        # Action space: movement directions
        self.action_space = np.array([
            [-2, -2], [-2, 0], [-2, 2],  # Up-left, up, up-right
            [0, -2],           [0, 2],   # Left, right  
            [2, -2], [2, 0], [2, 2]     # Down-left, down, down-right
        ])
        
        # Policy parameters
        self.theta = np.random.randn(9) * 0.1
        self.velocity = np.array([0.0, 0.0])  # Momentum for smoother movement
        
        # Tracking
        self.entropy_history = []
        self.phi_history = []
        self.suspicious_regions = []
        self.intensity_values = []
        
    def _compute_graph_weights(self) -> np.ndarray:
        weights = np.zeros((self.bins, self.bins))
        for i in range(self.bins):
            for j in range(self.bins):
                weights[i, j] = 1.0 / (1.0 + abs(i - j))
        return weights
    
    def _pixel_to_bin(self, intensity: float) -> int:
        """Convert pixel intensity to histogram bin"""
        # Normalize intensity (0-1) to bin (0-bins-1)
        norm_intensity = np.clip(intensity, 0.0, 1.0)
        bin_idx = int(norm_intensity * (self.bins - 1))
        return max(0, min(self.bins - 1, bin_idx))
    
    def _compute_entropy_proxy(self, intensity: float) -> float:
        """Compute entropy proxy based on intensity diversity"""
        bin_idx = self._pixel_to_bin(intensity)
        
        # Update histogram
        self.histogram[bin_idx] += 1
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
        """Compute spooky inhibition field"""
        expected_per_bin = self.window_size / self.bins
        s = self.histogram / expected_per_bin if expected_per_bin > 0 else np.zeros(self.bins)
        
        delta = np.zeros(self.bins)
        for i in range(self.bins):
            for j in range(self.bins):
                delta[i] += self.graph_weights[i, j] * (s[j] - self.s_bar)
        
        self.phi = 1.0 / (1.0 + np.exp(-delta / self.T_phi))
        self.phi_bar = np.mean(self.phi)
    
    def _is_suspicious_region(self, img: np.ndarray, x: int, y: int) -> Tuple[bool, float]:
        """
        Medical criteria for suspicious regions in mammograms:
        - High intensity (white spots - calcifications)
        - High local variance (texture changes - masses)
        - High gradient (sharp edges - tumors)
        """
        x, y = int(x), int(y)
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        
        # Get local region
        region_size = 15
        y_start = max(0, y - region_size//2)
        y_end = min(img.shape[0], y + region_size//2)
        x_start = max(0, x - region_size//2)
        x_end = min(img.shape[1], x + region_size//2)
        
        local_region = img[y_start:y_end, x_start:x_end]
        
        if local_region.size == 0:
            return False, 0.0
        
        # Feature 1: Intensity (calcifications appear white)
        local_mean = np.mean(local_region)
        
        # Feature 2: Variance (masses have different texture)
        local_var = np.var(local_region)
        
        # Feature 3: Gradient magnitude (tumor edges are sharp)
        grad_x = np.gradient(local_region, axis=1)
        grad_y = np.gradient(local_region, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        local_grad = np.mean(grad_mag)
        
        # Combined suspiciousness score
        intensity_score = max(0, (local_mean - 0.3) / 0.7)  # High intensity
        variance_score = min(1.0, local_var * 10)  # High variance
        gradient_score = min(1.0, local_grad * 5)  # High gradient
        
        suspiciousness = 0.4 * intensity_score + 0.3 * variance_score + 0.3 * gradient_score
        
        # Threshold for suspicious region
        is_suspicious = suspiciousness > 0.6
        
        return is_suspicious, suspiciousness
    
    def get_action(self, img: np.ndarray) -> np.ndarray:
        """Get movement action based on current position and image features"""
        x, y = int(self.current_pos[0]), int(self.current_pos[1])
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        
        # Get current pixel intensity
        current_intensity = img[y, x]
        
        # Compute gradients around current position
        grad_x = grad_y = 0
        if 0 < x < img.shape[1]-1 and 0 < y < img.shape[0]-1:
            grad_x = img[y, x+1] - img[y, x-1]
            grad_y = img[y+1, x] - img[y-1, x]
        
        # Weight actions based on gradients and policy parameters
        action_weights = np.copy(self.theta)
        
        # Bias toward high-gradient directions (edges of suspicious regions)
        for i, action in enumerate(self.action_space):
            grad_alignment = grad_x * action[0] + grad_y * action[1]
            action_weights[i] += 0.1 * grad_alignment
        
        # Normalize weights
        action_weights = np.exp(action_weights)  # Softmax-like
        action_weights = action_weights / np.sum(action_weights)
        
        # Choose action probabilistically
        action_idx = np.random.choice(len(self.action_space), p=action_weights)
        return self.action_space[action_idx]
    
    def update_policy(self, img: np.ndarray, x: int, y: int) -> Tuple[float, float, bool, float]:
        """
        Update policy using SEEG algorithm
        
        Returns: (entropy_proxy, phi_bar, is_suspicious, suspiciousness_score)
        """
        x, y = int(x), int(y)
        x = max(0, min(x, img.shape[1] - 1))
        y = max(0, min(y, img.shape[0] - 1))
        
        # Get pixel intensity
        intensity = img[y, x]
        self.intensity_values.append(intensity)
        
        # Compute entropy proxy
        entropy_proxy = self._compute_entropy_proxy(intensity)
        
        # Check if suspicious
        is_suspicious, suspiciousness_score = self._is_suspicious_region(img, x, y)
        
        # Compute reward
        if is_suspicious:
            reward = 5.0 * suspiciousness_score  # High reward for suspicious regions
            if (x, y) not in [pos for pos, _ in self.suspicious_regions]:
                self.suspicious_regions.append(((x, y), suspiciousness_score))
        else:
            reward = -0.1  # Small penalty for normal tissue
        
        # Compute spooky field
        self._compute_spooky_field()
        
        # Simple parameter update with entropy and spooky field
        # The spooky field prevents getting stuck exploring normal tissue patterns
        exploration_boost = (1 - self.phi_bar)  # Higher when entropy needs to increase
        
        # Update policy parameters
        gradient = reward * exploration_boost * 0.01
        self.theta += gradient * (np.random.randn(len(self.theta)) * 0.1 + 0.5)
        self.theta = np.clip(self.theta, -1.0, 1.0)
        
        # Track metrics
        self.entropy_history.append(entropy_proxy)
        self.phi_history.append(self.phi_bar)
        self.position_history.append((x, y))
        
        return entropy_proxy, self.phi_bar, is_suspicious, suspiciousness_score
    
    def analyze_mammogram(self, img: np.ndarray, max_steps: int = 2000) -> list:
        """Analyze mammogram for suspicious regions"""
        print("Starting mammogram analysis with SEEG...")
        
        for step in range(max_steps):
            # Get current position
            x, y = int(self.current_pos[0]), int(self.current_pos[1])
            x = max(0, min(x, img.shape[1] - 1))
            y = max(0, min(y, img.shape[0] - 1))
            
            # Update policy and check for suspicious regions
            entropy_proxy, phi_bar, is_suspicious, suspiciousness = self.update_policy(img, x, y)
            
            if is_suspicious and suspiciousness > 0.7:
                print(f"Suspicious region detected at ({x}, {y}), score: {suspiciousness:.3f}")
            
            # Get next action
            action = self.get_action(img)
            
            # Update position with momentum
            self.velocity = 0.7 * self.velocity + 0.3 * action  # Momentum
            self.current_pos += self.velocity
            
            # Keep within image bounds
            self.current_pos[0] = np.clip(self.current_pos[0], 0, img.shape[1] - 1)
            self.current_pos[1] = np.clip(self.current_pos[1], 0, img.shape[0] - 1)
            
            # Occasionally add random exploration to avoid getting stuck
            if step % 100 == 0:
                self.current_pos += np.random.randn(2) * 3
        
        return self.suspicious_regions

def create_realistic_mammogram():
    """
    Create a realistic mammogram based on medical characteristics:
    - Breast tissue density variations
    - Fibroglandular patterns
    - Potential calcifications (bright spots)
    - Masses (dense regions)
    """
    height, width = 512, 512
    img = np.zeros((height, width), dtype=np.float32)
    
    # Create breast shape (semi-realistic)
    center_x, center_y = width // 2, height // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Elliptical breast shape
    breast_mask = ((x_coords - center_x)**2 / (width*0.4)**2 + 
                   (y_coords - center_y)**2 / (height*0.6)**2) <= 1.0
    
    # Add breast tissue with varying density
    base_tissue = np.random.random((height, width)) * 0.3
    base_tissue = gaussian_filter(base_tissue, sigma=2)
    
    # Add fibroglandular patterns (more dense tissue)
    fibro_pattern = np.random.random((height, width)) * 0.4
    fibro_pattern = gaussian_filter(fibro_pattern, sigma=1)
    fibro_pattern = np.clip(fibro_pattern, 0, 0.4)
    
    # Combine tissues
    img = base_tissue + 0.3 * fibro_pattern
    img = img * breast_mask.astype(float)  # Apply breast shape mask
    
    # Add realistic mammographic patterns
    # Vascular structures
    for _ in range(20):
        x_start = np.random.randint(width//4, 3*width//4)
        y_start = np.random.randint(height//4, 3*height//4)
        length = np.random.randint(20, 80)
        angle = np.random.random() * 2 * np.pi
        
        for i in range(length):
            x = int(x_start + i * np.cos(angle))
            y = int(y_start + i * np.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = min(0.8, img[y, x] + 0.2)
    
    # Add calcifications (small bright spots)
    for _ in range(15):
        x = np.random.randint(width//3, 2*width//3)
        y = np.random.randint(height//3, 2*height//3)
        radius = np.random.randint(1, 4)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 <= radius**2:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        img[ny, nx] = min(1.0, img[ny, nx] + 0.7)
    
    # Add a mass-like structure (dense area)
    mass_center_x = width // 3
    mass_center_y = 2 * height // 3
    mass_radius = 30
    
    for y in range(max(0, mass_center_y - mass_radius), min(height, mass_center_y + mass_radius)):
        for x in range(max(0, mass_center_x - mass_radius), min(width, mass_center_x + mass_radius)):
            dist = np.sqrt((x - mass_center_x)**2 + (y - mass_center_y)**2)
            if dist <= mass_radius:
                # Create gradient effect for mass
                intensity_increase = (1 - dist / mass_radius) * 0.5
                img[y, x] = min(1.0, img[y, x] + intensity_increase)
    
    # Add noise to make it look realistic
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    # Apply final smoothing
    img = gaussian_filter(img, sigma=0.5)
    
    return img

def run_mammogram_analysis():
    """Run SEEG analysis on realistic mammogram"""
    print("Creating realistic mammogram simulation...")
    mammogram = create_realistic_mammogram()
    
    print(f"Mammogram shape: {mammogram.shape}")
    print(f"Intensity range: {mammogram.min():.3f} - {mammogram.max():.3f}")
    
    # Initialize SEEG detector
    detector = MammogramSEEGDetector(
        image_shape=mammogram.shape,
        bins=32,
        window_size=100,
        learning_rate=0.01
    )
    
    print("\nStarting mammogram analysis with SEEG...")
    start_time = time.time()
    
    # Analyze mammogram
    suspicious_regions = detector.analyze_mammogram(mammogram, max_steps=1000)
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(suspicious_regions)} suspicious regions")
    
    # Sort by suspiciousness score
    suspicious_regions.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 suspicious regions:")
    for i, ((x, y), score) in enumerate(suspicious_regions[:5]):
        print(f"  {i+1}. Position: ({x}, {y}), Score: {score:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original mammogram
    im1 = axes[0,0].imshow(mammogram, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('Simulated Mammogram')
    axes[0,0].axis('off')
    plt.colorbar(im1, ax=axes[0,0])
    
    # With detected regions marked
    marked_img = mammogram.copy()
    for (x, y), score in suspicious_regions[:20]:  # Mark top 20
        cv2.circle(marked_img, (x, y), 3, 1.0, 1)
    
    im2 = axes[0,1].imshow(marked_img, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title('Detected Suspicious Regions')
    axes[0,1].axis('off')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Path taken by SEEG
    path_img = mammogram.copy()
    if detector.position_history:
        path_x, path_y = zip(*detector.position_history[::10])  # Every 10th point
        axes[0,2].imshow(path_img, cmap='gray', vmin=0, vmax=1)
        axes[0,2].scatter(path_x, path_y, c='red', s=1, alpha=0.6)
        axes[0,2].plot(path_x, path_y, 'r-', alpha=0.3, linewidth=0.5)
        axes[0,2].set_title('SEEG Exploration Path')
        axes[0,2].axis('off')
    else:
        axes[0,2].text(0.5, 0.5, 'No path recorded', ha='center', va='center')
        axes[0,2].set_title('Exploration Path')
        axes[0,2].axis('off')
    
    # Entropy over time
    axes[1,0].plot(detector.entropy_history)
    axes[1,0].set_title('Entropy Proxy Over Time')
    axes[1,0].set_xlabel('Step')
    axes[1,0].set_ylabel('Entropy')
    axes[1,0].grid(True)
    
    # Phi bar (spooky inhibition) over time
    axes[1,1].plot(detector.phi_history)
    axes[1,1].set_title('Spooky Inhibition (Φ̄) Over Time')
    axes[1,1].set_xlabel('Step')
    axes[1,1].set_ylabel('Φ̄')
    axes[1,1].grid(True)
    
    # Intensity histogram
    axes[1,2].hist(detector.intensity_values, bins=50, alpha=0.7)
    axes[1,2].set_title('Pixel Intensity Distribution')
    axes[1,2].set_xlabel('Intensity')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return suspicious_regions, mammogram

# Run the mammogram analysis
if __name__ == "__main__":
    results, img = run_mammogram_analysis()
    
    print(f"\nSEEG Analysis Summary:")
    print(f"- Total suspicious regions found: {len(results)}")
    print(f"- Most suspicious region score: {results[0][1]:.3f}" if results else "- No suspicious regions found")
    print(f"- Average suspiciousness: {np.mean([score for _, score in results]):.3f}" if results else "- N/A")