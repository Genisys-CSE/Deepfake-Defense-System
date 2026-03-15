import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from methods.adv import apply_eot

def protect(face_tensor: torch.Tensor, orig_image_tensor: torch.Tensor, device: torch.device, params: dict) -> torch.Tensor:
    """
    face_tensor: CxHxW in [0,1], float32
    orig_image_tensor: full-image or same face tensor (for perceptual losses)
    returns: protected face tensor same shape, values clipped to [0,1]
    """
    epsilon = params.get('epsilon', 4.0 / 255.0)
    steps = params.get('steps', 150)
    lr = params.get('lr', 0.01)
    eot_samples = params.get('eot_samples', 8)
    lambda_lpips = params.get('lambda_lpips', 5.0) # Lower weight for LPIPS to emphasize identity
    
    facenet = params['facenet'].to(device).eval()
    surrogate = params['surrogate'].to(device).eval()
    lpips_model = params['lpips_model'].to(device).eval()
    
    for p in facenet.parameters(): p.requires_grad = False
    for p in surrogate.parameters(): p.requires_grad = False
    for p in lpips_model.parameters(): p.requires_grad = False
    
    face_tensor = face_tensor.to(device)
    
    with torch.no_grad():
        face_resized_fn = F.interpolate(face_tensor.unsqueeze(0), size=(160, 160), mode='bilinear', align_corners=False)
        face_resized_fn_norm = (face_resized_fn - 0.5) * 2.0
        orig_embedding = facenet(face_resized_fn_norm).detach()
        
    delta = torch.zeros_like(face_tensor, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)

    target_embedding = params.get('target_embedding', None)
    if target_embedding is not None:
        target_embedding = target_embedding.to(device)
        print("Running targeted cloaking...")
    else:
        print("Running untargeted cloaking...")

    for step in tqdm(range(steps), desc="Cloak PGD"):
        optimizer.zero_grad()
        
        total_id_loss = 0.0
        total_lpips_loss = 0.0
        
        for _ in range(eot_samples):
            protected = torch.clamp(face_tensor + delta, 0.0, 1.0)
            protected_eot = apply_eot(protected.unsqueeze(0))
            
            prot_resized_fn = F.interpolate(protected_eot, size=(160, 160), mode='bilinear', align_corners=False)
            prot_resized_fn_norm = (prot_resized_fn - 0.5) * 2.0
            prot_embedding = facenet(prot_resized_fn_norm)
            
            if target_embedding is not None:
                id_loss = -F.cosine_similarity(target_embedding, prot_embedding, dim=1).mean()
            else:
                id_loss = F.cosine_similarity(orig_embedding, prot_embedding, dim=1).mean()
            
            lpips_val = lpips_model((protected.unsqueeze(0) - 0.5) * 2.0, (face_tensor.unsqueeze(0) - 0.5) * 2.0).mean()
            
            total_id_loss += id_loss
            total_lpips_loss += lpips_val

        total_id_loss /= eot_samples
        total_lpips_loss /= eot_samples
        
        loss = total_id_loss + lambda_lpips * total_lpips_loss
        loss.backward()
        
        optimizer.step()
        
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            valid_protected = torch.clamp(face_tensor + delta.data, 0.0, 1.0)
            delta.data = valid_protected - face_tensor

    protected_face = torch.clamp(face_tensor + delta.detach(), 0.0, 1.0)
    return protected_face
