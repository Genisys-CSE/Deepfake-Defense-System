import torch
import numpy as np

def create_high_freq_mask(shape, radius):
    """
    Creates a high-frequency mask for a given shape (H, W).
    radius: relative threshold (0.0 to 1.0)
    """
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    dist_from_center = np.sqrt((Y - center[0])**2 + (X - center[1])**2)
    
    max_dist = np.sqrt(center[0]**2 + center[1]**2)
    mask = dist_from_center > (radius * max_dist)
    
    # Soft tapering
    taper = np.clip((dist_from_center - (radius * max_dist)) / (0.1 * max_dist), 0, 1)
    return taper

def protect(face_tensor: torch.Tensor, orig_image_tensor: torch.Tensor, device: torch.device, params: dict) -> torch.Tensor:
    """
    face_tensor: CxHxW in [0,1], float32
    orig_image_tensor: full-image or same face tensor (for perceptual losses)
    returns: protected face tensor same shape, values clipped to [0,1]
    """
    epsilon_freq = params.get('epsilon_freq', 2.0 / 255.0)
    freq_radius = params.get('freq_radius', 0.3)
    
    face_np = face_tensor.detach().cpu().numpy() # (C, H, W)
    face_np = np.transpose(face_np, (1, 2, 0)) # (H, W, C)
    
    H, W, C = face_np.shape
    mask = create_high_freq_mask((H, W), freq_radius)
    
    protected_np = np.zeros_like(face_np)
    
    for c in range(C):
        channel = face_np[:, :, c]
        
        # 2D FFT
        F = np.fft.fft2(channel)
        Fshift = np.fft.fftshift(F)
        
        # Add small complex noise to magnitude in masked region
        magnitude = np.abs(Fshift)
        random_phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=Fshift.shape))
        
        # Fixed scaling so spatial noise amplitude respects epsilon_freq budget
        num_masked = np.sum(mask) + 1e-8
        scale_factor = epsilon_freq * H * W / np.sqrt(num_masked)
        perturbation = mask * scale_factor * random_phase
        Fshift_modified = Fshift + perturbation
        
        # Inverse shift & ifft
        F_inv = np.fft.ifftshift(Fshift_modified)
        modified_channel = np.fft.ifft2(F_inv)
        
        protected_np[:, :, c] = np.real(modified_channel)
        
    protected_np = np.clip(protected_np, 0.0, 1.0)
    
    # Convert back to tensor
    protected_tensor = torch.from_numpy(np.transpose(protected_np, (2, 0, 1))).float().to(device)
    
    return protected_tensor
