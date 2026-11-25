import os
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.hub
    import torch.nn.functional as F
    import open_clip
    from lora_fused_qkv import apply_fused_qkv_lora
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default DermaCLIP configuration
DEFAULT_CLASSES = [
    "Acne",
    "Blackheads",
    "Whiteheads",
    "Wrinkles",
    "Pigmentation",
    "Freckles",
    "Dark Circles",
    "Pores",
    "Redness",
    "Puffy Eyes"
]

DEFAULT_PROMPTS = [
    "A face with acne.",
    "A face with blackheads.",
    "A face with whiteheads.",
    "A face with wrinkles.",
    "A face with pigmentation.",
    "A face with freckles.",
    "A face with dark circles.",
    "A face with pores.",
    "A face with redness.",
    "A face with puffy eyes."
]

CANDIDATE_CHECKPOINT_PATHS = [
    Path(__file__).parent / "best_macro_f1.pt",
    Path(__file__).parent / "models" / "best_macro_f1.pt",
    Path(__file__).parent.parent / "DermaCLIP" / "DermaCLIP_H100" / "best_macro_f1.pt",
    Path(__file__).parent.parent / "DermaCLIP" / "DermaFace_v1" / "checkpoints" / "best_macro_f1.pt",
    Path(__file__).parent.parent / "DermaCLIP" / "DermaFace_Part1_OpenCLIP" / "checkpoints" / "best_macro_f1.pt",
    Path("/Users/pranaav/Desktop/Bsquare/DermaCLIP/DermaCLIP_H100/best_macro_f1.pt"),
]


class DermaCLIPAnalyzer:
    _instance = None

    @classmethod
    def get_instance(cls, checkpoint_path=None, device=None):
        if cls._instance is None:
            cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
        return cls._instance

    def __init__(self, checkpoint_path=None, device=None):
        self.device_str = device or (
            "cuda" if TORCH_AVAILABLE and torch.cuda.is_available()
            else "mps" if TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.device = torch.device(self.device_str) if TORCH_AVAILABLE else None
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.class_names = DEFAULT_CLASSES
        self.text_embeddings = None
        self.is_loaded = False

        if not TORCH_AVAILABLE:
            print("[DermaCLIP] Torch or OpenCLIP not available. Running in fallback mode.")
            return

        resolved_path = self._resolve_checkpoint(checkpoint_path)
        if resolved_path is not None and resolved_path.exists():
            self._load_model(resolved_path)
        else:
            print(f"[DermaCLIP] Checkpoint not found at {checkpoint_path}. Trying fallback OpenCLIP ViT-L-14 base.")
            self._load_base_model()

    def _resolve_checkpoint(self, given_path):
        if given_path and Path(given_path).exists():
            return Path(given_path)
        for cand in CANDIDATE_CHECKPOINT_PATHS:
            if cand.exists():
                return cand
        return None

    def _load_model(self, checkpoint_path):
        try:
            print("=" * 60)
            print("[DermaCLIP] Loading DermaCLIP checkpoint from:")
            print(f"  {checkpoint_path}")
            print(f"  Target device: {self.device_str}")
            print("=" * 60)

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            checkpoint_config = checkpoint.get("config", {})
            model_cfg = checkpoint_config.get("model", {})
            model_name = model_cfg.get("name", "ViT-L-14-quickgelu")
            pretrained = model_cfg.get("pretrained", "openai")
            lora_rank = int(model_cfg.get("lora_rank", 8))
            lora_alpha = float(model_cfg.get("lora_alpha", 16.0))

            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]

            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=pretrained
                )
            except Exception as e_base:
                print(f"[DermaCLIP] Base load ({pretrained}) failed: {e_base}. Trying direct OpenAI CDN...")
                openai_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd8d72c9937d60e95d409f6e4971c223c72b53112e698889a74a16b9f2/ViT-L-14.pt"
                cache_file = Path.home() / ".cache" / "clip" / "ViT-L-14.pt"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                if not cache_file.exists():
                    torch.hub.download_url_to_file(openai_url, str(cache_file))
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=str(cache_file)
                )

            # Apply fused-QKV LoRA
            num_blocks = apply_fused_qkv_lora(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha
            )
            print(f"[DermaCLIP] Applied LoRA to {num_blocks} vision blocks (rank={lora_rank}, alpha={lora_alpha})")

            # Load checkpoint weights
            state_dict = checkpoint["model_state_dict"]
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            print(f"[DermaCLIP] Missing keys: {len(missing)}")
            print(f"[DermaCLIP] Unexpected keys: {len(unexpected)}")

            if missing:
                print("[DermaCLIP] First 30 missing keys:")
                for key in missing[:30]:
                    print(f"  MISSING: {key}")

            if unexpected:
                print("[DermaCLIP] First 30 unexpected keys:")
                for key in unexpected[:30]:
                    print(f"  UNEXPECTED: {key}")

            self.model = self.model.to(self.device)
            self.model.eval()

            self.tokenizer = open_clip.get_tokenizer(model_name)
            self._init_text_embeddings()
            self.is_loaded = True
            print(f"[DermaCLIP] Successfully initialized DermaCLIP for classes: {self.class_names}")

        except Exception as e:
            print(f"[DermaCLIP] Error loading checkpoint: {e}. Falling back to base model.")
            self._load_base_model()

    def _load_base_model(self):
        try:
            model_name = "ViT-B-32"
            print(f"[DermaCLIP] Initializing base OpenCLIP model: {model_name}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained="openai"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self._init_text_embeddings()
            self.is_loaded = True
            print("[DermaCLIP] Base model loaded successfully.")
        except Exception as e:
            print(f"[DermaCLIP] Failed to load base OpenCLIP: {e}")
            self.is_loaded = False

    def _init_text_embeddings(self):
        prompts = [
            f"A face with {c.lower()}." for c in self.class_names
        ]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens, normalize=False)
            self.text_embeddings = F.normalize(text_features.float(), dim=-1)

    def analyze(self, image_input):
        """
        Analyze an image (cv2 BGR, RGB, or PIL Image) with DermaCLIP.
        Returns:
          probabilities: dict of {class_name: float}
          semantic_features: dict mapped to dermatology concepts
          predicted_class: str
        """
        if not self.is_loaded or self.model is None or image_input is None:
            return self._fallback_scores(image_input)

        try:
            # Convert to PIL Image in RGB
            if isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    return self._fallback_scores(image_input)
                # Lighting enhancement via CLAHE
                enhanced = self._apply_clahe(image_input)
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                return self._fallback_scores(image_input)

            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                image_features = self.model.encode_image(tensor, normalize=False)
                image_features = F.normalize(image_features.float(), dim=-1)

                logits = image_features @ self.text_embeddings.T
                if hasattr(self.model, "logit_scale"):
                    logits = logits * self.model.logit_scale.exp()

                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

            probs_dict = {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            }

            semantic = self._map_to_semantic_concepts(probs_dict)
            best_idx = int(np.argmax(probs))
            predicted_class = self.class_names[best_idx]

            return {
                "probabilities": probs_dict,
                "semantic_features": semantic,
                "predicted_class": predicted_class,
                "confidence": float(probs[best_idx])
            }

        except Exception as e:
            print(f"[DermaCLIP] Inference error: {e}")
            return self._fallback_scores(image_input)

    def _apply_clahe(self, bgr_img):
        try:
            lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            merged = cv2.merge((l, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return bgr_img

    def _map_to_semantic_concepts(self, probs):
        acne_val = probs.get("Acne", 0.0)
        blackheads_val = probs.get("Blackheads", 0.0)
        whiteheads_val = probs.get("Whiteheads", 0.0)
        wrinkles_val = probs.get("Wrinkles", 0.0)
        pigment_val = probs.get("Pigmentation", 0.0)
        freckles_val = probs.get("Freckles", 0.0)
        dark_circles_val = probs.get("Dark Circles", 0.0)
        pores_val = probs.get("Pores", 0.0)
        redness_val = probs.get("Redness", 0.0)
        puffy_val = probs.get("Puffy Eyes", 0.0)

        # Compound clinical concept scores
        sem_acne = float(np.clip(acne_val + 0.5 * blackheads_val + 0.5 * whiteheads_val, 0.0, 1.0))
        sem_pigment = float(np.clip(pigment_val + 0.5 * freckles_val, 0.0, 1.0))
        sem_wrinkles = float(np.clip(wrinkles_val, 0.0, 1.0))
        sem_dark = float(np.clip(dark_circles_val + 0.5 * puffy_val, 0.0, 1.0))
        sem_redness = float(np.clip(redness_val, 0.0, 1.0))
        sem_texture = float(np.clip(pores_val + 0.3 * blackheads_val, 0.0, 1.0))

        # Health score represents absence of acute dermatological severity
        overall_flaw = (sem_acne * 0.3 + sem_pigment * 0.25 + sem_wrinkles * 0.2 + sem_dark * 0.15 + sem_redness * 0.1)
        sem_healthy = float(np.clip(1.0 - overall_flaw, 0.05, 1.0))

        return {
            "semantic_acne": sem_acne,
            "semantic_pigmentation": sem_pigment,
            "semantic_wrinkles": sem_wrinkles,
            "semantic_dark_circles": sem_dark,
            "semantic_redness": sem_redness,
            "semantic_texture": sem_texture,
            "semantic_healthy": sem_healthy
        }

    def _fallback_scores(self, image_input):
        probs_dict = {c: 0.1 for c in self.class_names}
        return {
            "probabilities": probs_dict,
            "semantic_features": {
                "semantic_acne": 0.1,
                "semantic_pigmentation": 0.1,
                "semantic_wrinkles": 0.1,
                "semantic_dark_circles": 0.1,
                "semantic_redness": 0.1,
                "semantic_texture": 0.1,
                "semantic_healthy": 0.8
            },
            "predicted_class": "Healthy",
            "confidence": 0.8
        }
