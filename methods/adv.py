import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from tqdm import tqdm

class StraightThroughJPEG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        device = x.device
        res = []
        for i in range(x.shape[0]):
            img_np = (x[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            if decimg is None:
                decimg = img_np
            else:
                if decimg.shape[2] == 3:
                    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
            decimg = np.transpose(decimg, (2, 0, 1))
            res.append(torch.from_numpy(decimg).float().to(device) / 255.0)
        return torch.stack(res)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def apply_eot(x):
    # Random resizing and small rotation
    transform = T.RandomAffine(degrees=5, scale=(0.95, 1.05))
    x_transformed = transform(x)
    
    # JPEG simulation (quality 80-95)
    quality = int(torch.randint(80, 96, (1,)).item())
    x_transformed = StraightThroughJPEG.apply(x_transformed, quality)
    
    # Gaussian noise
    noise = torch.randn_like(x_transformed) * 0.02
    x_transformed = torch.clamp(x_transformed + noise, 0.0, 1.0)
    
    return x_transformed

def protect(face_tensor: torch.Tensor, orig_image_tensor: torch.Tensor, device: torch.device, params: dict) -> torch.Tensor:
    """
    face_tensor: CxHxW in [0,1], float32
    orig_image_tensor: full-image or same face tensor (for perceptual losses)
    returns: protected face tensor same shape, values clipped to [0,1]
    """
    # Extract params
    epsilon = params.get('epsilon', 4.0 / 255.0)
    steps = params.get('steps', 150)
    lr = params.get('lr', 0.01)
    eot_samples = params.get('eot_samples', 8)
    lambda_lpips = params.get('lambda_lpips', 10.0)
    lambda_feat = params.get('lambda_feat', 5.0)
    
    # Models are passed via params to avoid reloading
    facenet = params['facenet'].to(device).eval()
    surrogate = params['surrogate'].to(device).eval()
    lpips_model = params['lpips_model'].to(device).eval()
    
    # Ensure gradients for models are off
    for p in facenet.parameters(): p.requires_grad = False
    for p in surrogate.parameters(): p.requires_grad = False
    for p in lpips_model.parameters(): p.requires_grad = False
    
    face_tensor = face_tensor.to(device)
    
    # ImageNet normalization for surrogate
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Original embeddings (no EOT for original)
    with torch.no_grad():
        face_resized_fn = F.interpolate(face_tensor.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False)
        face_resized_fn_norm = (face_resized_fn - 0.5) * 2.0
        orig_embedding = facenet(face_resized_fn_norm).detach()
        
        face_resized_surr = F.interpolate(face_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        orig_feat = surrogate(normalize(face_resized_surr)).flatten(1).detach()

    delta = torch.zeros_like(face_tensor, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    print("Running PGD+EOT adversarial attack (adv)...")
    for step in tqdm(range(steps), desc="Adv PGD"):
        optimizer.zero_grad()
        
        total_id_loss = 0.0
        total_feat_loss = 0.0
        total_lpips_loss = 0.0
        
        for _ in range(eot_samples):
            protected = torch.clamp(face_tensor + delta, 0.0, 1.0)
            
            # Apply EOT to the single image batch
            protected_eot = apply_eot(protected.unsqueeze(0))
            
            # FaceNet evaluation — minimize cosine similarity (push identity away)
            prot_resized_fn = F.interpolate(protected_eot, size=(160, 160), mode='bilinear', align_corners=False)
            prot_resized_fn_norm = (prot_resized_fn - 0.5) * 2.0
            prot_embedding = facenet(prot_resized_fn_norm)
            id_sim = F.cosine_similarity(orig_embedding, prot_embedding, dim=1).mean()
            
            # Surrogate evaluation — minimize cosine similarity (disrupt deepfake features)
            prot_resized_surr = F.interpolate(protected_eot, size=(224, 224), mode='bilinear', align_corners=False)
            prot_feat = surrogate(normalize(prot_resized_surr)).flatten(1)
            feat_sim = F.cosine_similarity(prot_feat, orig_feat, dim=1).mean()
            
            # LPIPS expects input in [-1, 1] — perceptual quality constraint
            lpips_val = lpips_model((protected.unsqueeze(0) - 0.5) * 2.0, (face_tensor.unsqueeze(0) - 0.5) * 2.0).mean()
            
            total_id_loss += id_sim
            total_feat_loss += feat_sim
            total_lpips_loss += lpips_val

        # Average over EOT samples
        total_id_loss /= eot_samples
        total_feat_loss /= eot_samples
        total_lpips_loss /= eot_samples
        
        # Minimize: id_sim (identity) + feat_sim (deepfake features) + lpips (perceptual penalty)
        loss = total_id_loss + lambda_feat * total_feat_loss + lambda_lpips * total_lpips_loss
        loss.backward()
        
        # Update delta
        optimizer.step()
        scheduler.step()
        
        # Project delta into L-inf ball (no smoothing during optimization — it destroys gradients)
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            valid_protected = torch.clamp(face_tensor + delta.data, 0.0, 1.0)
            delta.data = valid_protected - face_tensor

    # Apply light smoothing only at the very end for visual quality
    with torch.no_grad():
        final_delta = T.GaussianBlur(kernel_size=3, sigma=0.8)(delta.data.unsqueeze(0)).squeeze(0)
        final_delta = torch.clamp(final_delta, -epsilon, epsilon)
        protected_face = torch.clamp(face_tensor + final_delta, 0.0, 1.0)
    
    return protected_face
