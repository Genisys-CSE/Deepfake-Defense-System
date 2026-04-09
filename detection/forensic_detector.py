"""
DeepShield - Enhanced Forensic Deepfake Detector

Hybrid detector that combines multiple forensic signals:
1. Frequency anomaly analysis (FFT radial behavior)
2. Sensor-noise consistency checks
3. Face-boundary blending artifact checks
4. Sharpness mismatch checks
5. JPEG blockiness/compression artifact checks
6. Color consistency checks (face vs surrounding region)
"""

import cv2
import numpy as np


class DeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.decision_threshold = 0.50

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {'error': f'Cannot read image: {image_path}'}

        h, w = img.shape[:2]
        if min(h, w) < 72:
            return {'error': 'Image is too small for reliable deepfake analysis'}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_box = self._largest_face(gray)
        has_face = face_box is not None

        frequency_score, spectrum_vis, frequency_meta = self._frequency_analysis(gray)
        noise_score = self._noise_consistency(gray, face_box)
        boundary_score = self._boundary_artifact_score(img, gray, face_box)
        sharpness_score = self._sharpness_mismatch(gray, face_box)
        compression_score = self._compression_artifact_score(gray)
        color_score = self._color_consistency_score(img, face_box)

        scores = {
            'frequency': self._clamp(frequency_score),
            'noise': self._clamp(noise_score),
            'boundary': self._clamp(boundary_score),
            'sharpness': self._clamp(sharpness_score),
            'compression': self._clamp(compression_score),
            'color': self._clamp(color_score),
        }

        raw_fake_probability = self._weighted_probability(scores, has_face)
        fake_probability = self._calibrate_probability(raw_fake_probability)
        label = 'FAKE' if fake_probability >= self.decision_threshold else 'REAL'
        confidence = self._confidence(fake_probability, scores, has_face)
        explanation = self._build_explanation(label, fake_probability, scores, has_face)

        return {
            'label': label,
            'confidence': round(float(confidence), 3),
            'fake_probability': round(float(fake_probability), 3),
            'analysis': {k: round(float(v), 3) for k, v in scores.items()},
            'analysis_extended': frequency_meta,
            'explanation': explanation,
            'spectrum_image': spectrum_vis,
        }

    def _largest_face(self, gray):
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
        )
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return int(fx), int(fy), int(fw), int(fh)

    def _frequency_analysis(self, gray):
        size = min(gray.shape[0], gray.shape[1])
        size = max(64, 2 ** int(np.log2(size)))
        gray_sq = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

        f_shift = np.fft.fftshift(np.fft.fft2(gray_sq))
        magnitude = np.log1p(np.abs(f_shift))
        mag_norm = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255)
        spectrum_vis = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_JET)

        profile = self._radial_profile(magnitude)
        if profile.size < 16:
            return 0.5, spectrum_vis, {'frequency_profile_quality': 'low'}

        n = profile.size
        low_band = profile[int(0.05 * n):int(0.25 * n)]
        mid_band = profile[int(0.25 * n):int(0.6 * n)]
        high_band = profile[int(0.6 * n):int(0.9 * n)]

        low_energy = float(np.mean(low_band) + 1e-6)
        mid_energy = float(np.mean(mid_band))
        high_energy = float(np.mean(high_band))

        high_low_ratio = high_energy / low_energy
        mid_peakiness = float(np.std(np.diff(mid_band)))
        periodicity = float(np.max(mid_band) - np.median(mid_band))

        score = (
            0.45 * self._scale(high_low_ratio, 0.58, 0.95) +
            0.35 * self._scale(mid_peakiness, 0.01, 0.055) +
            0.20 * self._scale(periodicity, 0.02, 0.13)
        )

        return self._clamp(score), spectrum_vis, {
            'high_low_ratio': round(high_low_ratio, 4),
            'mid_peakiness': round(mid_peakiness, 4),
            'periodicity': round(periodicity, 4),
            'mid_energy': round(mid_energy, 4),
        }

    @staticmethod
    def _radial_profile(magnitude):
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        profile = tbin / (nr + 1e-8)
        return profile[:min(h, w) // 2]

    def _noise_consistency(self, gray, face_box):
        gray_f = gray.astype(np.float32)
        residual = gray_f - cv2.GaussianBlur(gray_f, (5, 5), 1.2)
        residual_std = np.std(residual)

        if face_box is None:
            h, w = gray.shape
            quadrants = [
                residual[:h // 2, :w // 2],
                residual[:h // 2, w // 2:],
                residual[h // 2:, :w // 2],
                residual[h // 2:, w // 2:],
            ]
            quad_stds = [np.std(q) for q in quadrants if q.size > 0]
            spread = float(np.std(quad_stds)) if len(quad_stds) > 1 else 0.0
            return 0.45 * self._scale(spread, 0.2, 1.8) + 0.55 * self._scale(residual_std, 4.0, 10.0)

        fx, fy, fw, fh = face_box
        face_noise = residual[fy:fy + fh, fx:fx + fw]
        if face_noise.size == 0:
            return 0.35

        bg_mask = np.ones_like(residual, dtype=bool)
        bg_mask[fy:fy + fh, fx:fx + fw] = False
        bg_noise = residual[bg_mask]
        if bg_noise.size == 0:
            return 0.35

        face_std = float(np.std(face_noise))
        bg_std = float(np.std(bg_noise))
        ratio = abs(face_std - bg_std) / (max(face_std, bg_std) + 1e-6)
        mean_shift = abs(float(np.mean(face_noise)) - float(np.mean(bg_noise)))
        return 0.75 * self._scale(ratio, 0.06, 0.34) + 0.25 * self._scale(mean_shift, 0.2, 2.2)

    def _boundary_artifact_score(self, img, gray, face_box):
        if face_box is None:
            return 0.35

        fx, fy, fw, fh = face_box
        h, w = gray.shape
        cx, cy = fx + fw // 2, fy + fh // 2

        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_outer, (cx, cy), (int(fw * 0.62), int(fh * 0.62)), 0, 0, 360, 255, -1)
        cv2.ellipse(mask_inner, (cx, cy), (int(fw * 0.44), int(fh * 0.44)), 0, 0, 360, 255, -1)

        ring_mask = (mask_outer > 0) & (mask_inner == 0)
        inner_mask = mask_inner > 0
        if ring_mask.sum() < 50 or inner_mask.sum() < 50:
            return 0.35

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

        ring_grad = float(np.mean(grad[ring_mask]))
        inner_grad = float(np.mean(grad[inner_mask]))
        grad_ratio = ring_grad / (inner_grad + 1e-6)

        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ring_color = np.mean(ycrcb[:, :, 1:3][ring_mask], axis=0)
        inner_color = np.mean(ycrcb[:, :, 1:3][inner_mask], axis=0)
        color_jump = float(np.linalg.norm(ring_color - inner_color) / 255.0)

        return 0.72 * self._scale(grad_ratio, 1.03, 1.75) + 0.28 * self._scale(color_jump, 0.02, 0.14)

    def _sharpness_mismatch(self, gray, face_box):
        if face_box is None:
            return 0.30

        fx, fy, fw, fh = face_box
        face_region = gray[fy:fy + fh, fx:fx + fw]
        if face_region.size == 0:
            return 0.30

        face_lap = float(cv2.Laplacian(face_region, cv2.CV_64F).var())
        bg_laps = self._background_patch_laplacians(gray, face_box)
        if not bg_laps:
            return 0.30

        bg_lap = float(np.mean(bg_laps))
        ratio = abs(face_lap - bg_lap) / (max(face_lap, bg_lap) + 1e-6)
        return self._scale(ratio, 0.08, 0.42)

    @staticmethod
    def _background_patch_laplacians(gray, face_box):
        h, w = gray.shape
        fx, fy, fw, fh = face_box
        pw = max(24, min(fw, w // 3))
        ph = max(24, min(fh, h // 3))
        coords = [
            (0, 0),
            (w - pw, 0),
            (0, h - ph),
            (w - pw, h - ph),
            (w // 2 - pw // 2, 0),
            (w // 2 - pw // 2, h - ph),
        ]

        laps = []
        for x, y in coords:
            x = int(max(0, min(w - pw, x)))
            y = int(max(0, min(h - ph, y)))
            patch = gray[y:y + ph, x:x + pw]
            if patch.size == 0:
                continue
            overlap_x = max(0, min(x + pw, fx + fw) - max(x, fx))
            overlap_y = max(0, min(y + ph, fy + fh) - max(y, fy))
            if overlap_x * overlap_y > 0:
                continue
            laps.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
        return laps

    def _compression_artifact_score(self, gray):
        gray_f = gray.astype(np.float32)
        if gray_f.shape[0] < 16 or gray_f.shape[1] < 16:
            return 0.35

        left_v = gray_f[:, 8::8]
        right_v = gray_f[:, 7::8]
        cols = min(left_v.shape[1], right_v.shape[1])
        if cols == 0:
            return 0.35
        v_bound = np.abs(left_v[:, :cols] - right_v[:, :cols])

        top_h = gray_f[8::8, :]
        bottom_h = gray_f[7::8, :]
        rows = min(top_h.shape[0], bottom_h.shape[0])
        if rows == 0:
            return 0.35
        h_bound = np.abs(top_h[:rows, :] - bottom_h[:rows, :])

        left_inner = gray_f[:, 5::8]
        right_inner = gray_f[:, 4::8]
        cols_inner = min(left_inner.shape[1], right_inner.shape[1])
        if cols_inner == 0:
            return 0.35
        v_inner = np.abs(left_inner[:, :cols_inner] - right_inner[:, :cols_inner])

        top_inner = gray_f[5::8, :]
        bottom_inner = gray_f[4::8, :]
        rows_inner = min(top_inner.shape[0], bottom_inner.shape[0])
        if rows_inner == 0:
            return 0.35
        h_inner = np.abs(top_inner[:rows_inner, :] - bottom_inner[:rows_inner, :])

        boundary_energy = float(np.mean(v_bound) + np.mean(h_bound))
        inner_energy = float(np.mean(v_inner) + np.mean(h_inner) + 1e-6)
        blockiness = (boundary_energy / inner_energy) - 1.0
        return self._scale(blockiness, 0.03, 0.42)

    def _color_consistency_score(self, img, face_box):
        if face_box is None:
            return 0.30

        fx, fy, fw, fh = face_box
        h, w = img.shape[:2]
        cx, cy = fx + fw // 2, fy + fh // 2

        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_outer, (cx, cy), (int(fw * 0.70), int(fh * 0.70)), 0, 0, 360, 255, -1)
        cv2.ellipse(mask_inner, (cx, cy), (int(fw * 0.48), int(fh * 0.48)), 0, 0, 360, 255, -1)
        ring_mask = (mask_outer > 0) & (mask_inner == 0)
        face_mask = mask_inner > 0

        if ring_mask.sum() < 100 or face_mask.sum() < 100:
            return 0.30

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        dists = []
        for ch in [1, 2]:
            face_hist = cv2.calcHist([lab], [ch], face_mask.astype(np.uint8), [32], [0, 256])
            ring_hist = cv2.calcHist([lab], [ch], ring_mask.astype(np.uint8), [32], [0, 256])
            cv2.normalize(face_hist, face_hist)
            cv2.normalize(ring_hist, ring_hist)
            dists.append(float(cv2.compareHist(face_hist, ring_hist, cv2.HISTCMP_BHATTACHARYYA)))

        sat = hsv[:, :, 1].astype(np.float32)
        face_var = float(np.var(sat[face_mask]))
        ring_var = float(np.var(sat[ring_mask]))
        sat_delta = abs(face_var - ring_var) / (max(face_var, ring_var) + 1e-6)

        hist_dist = float(np.mean(dists))
        return 0.72 * self._scale(hist_dist, 0.06, 0.30) + 0.28 * self._scale(sat_delta, 0.10, 0.55)

    def _weighted_probability(self, scores, has_face):
        if has_face:
            weights = {
                'frequency': 0.23,
                'noise': 0.18,
                'boundary': 0.18,
                'sharpness': 0.14,
                'compression': 0.15,
                'color': 0.12,
            }
        else:
            weights = {
                'frequency': 0.44,
                'noise': 0.24,
                'boundary': 0.08,
                'sharpness': 0.08,
                'compression': 0.16,
                'color': 0.00,
            }
        return sum(scores[k] * w for k, w in weights.items())

    @staticmethod
    def _calibrate_probability(raw_score):
        logit = (raw_score - 0.50) * 6.5
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(prob, 0.0, 1.0))

    def _confidence(self, fake_probability, scores, has_face):
        distance = abs(fake_probability - 0.5) * 2.0
        signal_values = np.array(list(scores.values()), dtype=np.float32)
        top_strength = float(np.mean(np.sort(signal_values)[-3:]))
        agreement = float(1.0 - np.std(signal_values))
        confidence = 0.55 * distance + 0.30 * top_strength + 0.15 * max(0.0, agreement)
        if not has_face:
            confidence *= 0.88
        return self._clamp(confidence)

    def _build_explanation(self, label, fake_probability, scores, has_face):
        rank = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = rank[:3]
        top_names = [self._pretty_signal_name(k) for k, _ in top]

        if label == 'FAKE':
            summary = (
                f"Likely FAKE: strongest anomalies are in {top_names[0]} and "
                f"{top_names[1]} (fake probability {fake_probability * 100:.1f}%)."
            )
        else:
            summary = (
                f"Likely REAL: manipulation indicators stay comparatively low "
                f"(fake probability {fake_probability * 100:.1f}%)."
            )

        reasons = []
        for key, value in top:
            reasons.append(self._reason_line(key, value))

        if not has_face:
            reasons.append(
                "No strong face region was detected, so verdict is based mostly on global forensic texture."
            )

        return {
            'summary': summary,
            'reasons': reasons,
            'face_detected': has_face,
            'top_signals': [
                {'name': self._pretty_signal_name(name), 'score': round(float(score), 3)}
                for name, score in top
            ],
        }

    def _reason_line(self, signal_key, score):
        strength = "high" if score >= 0.68 else "moderate" if score >= 0.50 else "low"
        templates = {
            'frequency': "Frequency profile shows {strength} synthetic spectral irregularity.",
            'noise': "Sensor-noise consistency shows {strength} mismatch between regions.",
            'boundary': "Face boundary transitions show {strength} blending-seam evidence.",
            'sharpness': "Face/background sharpness relation indicates {strength} inconsistency.",
            'compression': "Compression grid response indicates {strength} recomposition artifacts.",
            'color': "Color distribution around the face shows {strength} chroma inconsistency.",
        }
        template = templates.get(signal_key, "Signal indicates {strength} anomaly.")
        return f"{self._pretty_signal_name(signal_key)}: {template.format(strength=strength)} (score {score * 100:.1f}%)."

    @staticmethod
    def _pretty_signal_name(signal_key):
        names = {
            'frequency': 'Frequency',
            'noise': 'Noise',
            'boundary': 'Boundary',
            'sharpness': 'Sharpness',
            'compression': 'Compression',
            'color': 'Color',
        }
        return names.get(signal_key, signal_key.title())

    @staticmethod
    def _scale(value, low, high):
        return float(np.clip((value - low) / (high - low + 1e-8), 0.0, 1.0))

    @staticmethod
    def _clamp(value):
        return float(np.clip(value, 0.0, 1.0))
