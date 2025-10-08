"""
Vocal separation service for the audio transcription pipeline.

This module provides source separation capabilities using Demucs models
to separate vocals from background music and noise.
"""

import numpy as np
import torch
import torchaudio
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

from ..core.config import settings
from ..core.logging import get_logger
from ..utils.timing import time_operation, time_context
from ..utils.device_utils import get_device, get_device_info
from ..utils.audio_utils import load_audio_as_tensor, save_wav_from_tensor, resample_audio_tensor

logger = get_logger(__name__)


@dataclass
class SeparationResult:
    """Result of audio source separation."""
    
    vocals: np.ndarray
    accompaniment: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, Any]
    
    @property
    def has_vocals(self) -> bool:
        """Check if vocals were detected."""
        return np.max(np.abs(self.vocals)) > 1e-6


class DemucsSeparator:
    """
    Demucs-based source separation implementation.
    
    Uses Facebook's Demucs models for high-quality source separation.
    """
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """
        Initialize Demucs separator.
        
        Args:
            model_name: Demucs model name (htdemucs, htdemucs_ft, etc.)
            device: Device to run model on (auto-detect if None)
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.model = None
        self.sample_rate = 44100  # Demucs default sample rate
        
        logger.info(f"Initializing Demucs separator with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load Demucs model."""
        try:
            import demucs.api
            
            # Load the model
            self.model = demucs.api.load_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Demucs model '{self.model_name}' loaded successfully on {self.device}")
            
        except ImportError:
            logger.error("Demucs not available. Install with: pip install demucs")
            raise
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise
    
    @time_operation("demucs_separation")
    def separate_sources(self, audio: np.ndarray, 
                        sample_rate: int,
                        target_sources: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Separate audio sources using Demucs.
        
        Args:
            audio: Audio array (1D or 2D, float32)
            sample_rate: Audio sample rate
            target_sources: List of sources to extract (default: ['vocals', 'drums', 'bass', 'other'])
            
        Returns:
            Dictionary mapping source names to separated audio arrays
        """
        if self.model is None:
            raise RuntimeError("Demucs model not loaded")
        
        if target_sources is None:
            target_sources = ['vocals', 'drums', 'bass', 'other']
        
        # Ensure audio is 2D (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Resample to Demucs sample rate if needed
        if sample_rate != self.sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
            # Convert to tensor for resampling, then back to numpy
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled_tensor = resample_audio_tensor(audio_tensor, sample_rate, self.sample_rate)
            audio = resampled_tensor.squeeze().numpy()
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        with time_context("demucs_inference"):
            try:
                # Run separation
                with torch.no_grad():
                    separated = self.model(audio_tensor)
                
                # Convert back to numpy and extract target sources
                result = {}
                source_names = ['drums', 'bass', 'other', 'vocals']  # Demucs output order
                
                for i, source_name in enumerate(source_names):
                    if source_name in target_sources:
                        source_audio = separated[0, i].cpu().numpy()  # Take first batch item
                        result[source_name] = source_audio
                
                logger.info(f"Demucs separation completed for sources: {list(result.keys())}")
                return result
                
            except Exception as e:
                logger.error(f"Demucs separation failed: {e}")
                raise
    
    def separate_vocals(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract vocals from audio.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            
        Returns:
            Separated vocals array
        """
        separated = self.separate_sources(audio, sample_rate, ['vocals'])
        return separated.get('vocals', np.zeros_like(audio))


class SpectralSubtractionSeparator:
    """
    Fallback spectral subtraction method for basic noise reduction.
    
    Used when Demucs is not available or fails.
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 0.01):
        """
        Initialize spectral subtraction separator.
        
        Args:
            alpha: Over-subtraction factor
            beta: Spectral floor factor
        """
        self.alpha = alpha
        self.beta = beta
        logger.info("Spectral subtraction separator initialized")
    
    @time_operation("spectral_subtraction")
    def separate_vocals(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply spectral subtraction for basic noise reduction.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            
        Returns:
            Noise-reduced audio array
        """
        try:
            import librosa
            import scipy.signal
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Estimate noise from first 0.5 seconds
            noise_samples = int(0.5 * sample_rate)
            noise_profile = audio[:min(noise_samples, len(audio))]
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum
            noise_stft = librosa.stft(noise_profile, n_fft=2048, hop_length=512)
            noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
            
            # Apply spectral subtraction
            gain = 1.0 - self.alpha * (noise_magnitude / (magnitude + 1e-10))
            gain = np.maximum(gain, self.beta)
            
            # Reconstruct audio
            enhanced_magnitude = magnitude * gain
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            logger.info("Spectral subtraction completed")
            return enhanced_audio.astype(np.float32)
            
        except ImportError:
            logger.warning("librosa not available for spectral subtraction")
            return audio
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio


class VocalSeparator:
    """
    Main vocal separation service.
    
    Provides a unified interface for different separation methods
    with automatic fallback and configuration management.
    """
    
    def __init__(self, method: str = "auto", **kwargs):
        """
        Initialize vocal separator.
        
        Args:
            method: Separation method ("demucs", "spectral", or "auto")
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.demucs_separator = None
        self.spectral_separator = None
        self.device = get_device()
        
        logger.info(f"Initializing vocal separator with method: {method}")
        self._initialize_separators(**kwargs)
    
    def _initialize_separators(self, **kwargs):
        """Initialize separation methods based on configuration."""
        try:
            if self.method in ["demucs", "auto"]:
                self.demucs_separator = DemucsSeparator(
                    model_name=kwargs.get('model_name', settings.demucs_model),
                    device=self.device
                )
                logger.info("Demucs separator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Demucs separator: {e}")
            self.demucs_separator = None
        
        # Always initialize spectral subtraction as fallback
        self.spectral_separator = SpectralSubtractionSeparator(
            alpha=kwargs.get('alpha', 2.0),
            beta=kwargs.get('beta', 0.01)
        )
        logger.info("Spectral subtraction separator initialized")
    
    @time_operation("vocal_separation")
    def separate_vocals(self, audio: np.ndarray, 
                       sample_rate: int,
                       method: Optional[str] = None,
                       **kwargs) -> SeparationResult:
        """
        Separate vocals from audio.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            method: Override separation method
            **kwargs: Additional arguments for separation methods
            
        Returns:
            SeparationResult with separated vocals and metadata
        """
        if not settings.enable_separation_by_default:
            logger.info("Vocal separation disabled, returning original audio")
            return SeparationResult(
                vocals=audio,
                accompaniment=np.zeros_like(audio),
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                metadata={"method": "none", "enabled": False}
            )
        
        # Determine which method to use
        use_method = method or self.method
        
        metadata = {
            "method": use_method,
            "enabled": True,
            "device": str(self.device)
        }
        
        try:
            if use_method == "demucs" and self.demucs_separator:
                return self._separate_with_demucs(audio, sample_rate, metadata, **kwargs)
            elif use_method == "spectral" and self.spectral_separator:
                return self._separate_with_spectral(audio, sample_rate, metadata, **kwargs)
            elif use_method == "auto":
                # Try Demucs first, fallback to spectral subtraction
                if self.demucs_separator:
                    try:
                        return self._separate_with_demucs(audio, sample_rate, metadata, **kwargs)
                    except Exception as e:
                        logger.warning(f"Demucs separation failed, falling back to spectral: {e}")
                
                if self.spectral_separator:
                    return self._separate_with_spectral(audio, sample_rate, metadata, **kwargs)
            
            raise RuntimeError(f"Separation method '{use_method}' not available")
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {e}")
            # Return original audio as fallback
            return SeparationResult(
                vocals=audio,
                accompaniment=np.zeros_like(audio),
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                metadata={**metadata, "error": str(e), "fallback": True}
            )
    
    def _separate_with_demucs(self, audio: np.ndarray, sample_rate: int, 
                             metadata: Dict[str, Any], **kwargs) -> SeparationResult:
        """Separate vocals using Demucs."""
        separated = self.demucs_separator.separate_sources(audio, sample_rate, **kwargs)
        
        vocals = separated.get('vocals', np.zeros_like(audio))
        accompaniment = np.sum([
            separated.get(source, np.zeros_like(audio)) 
            for source in ['drums', 'bass', 'other']
        ], axis=0)
        
        # Resample back to original sample rate if needed
        if sample_rate != self.demucs_separator.sample_rate:
            # Convert to tensors for resampling
            vocals_tensor = torch.from_numpy(vocals).unsqueeze(0)
            accompaniment_tensor = torch.from_numpy(accompaniment).unsqueeze(0)
            
            vocals_resampled = resample_audio_tensor(vocals_tensor, self.demucs_separator.sample_rate, sample_rate)
            accompaniment_resampled = resample_audio_tensor(accompaniment_tensor, self.demucs_separator.sample_rate, sample_rate)
            
            vocals = vocals_resampled.squeeze().numpy()
            accompaniment = accompaniment_resampled.squeeze().numpy()
        
        metadata.update({
            "model": self.demucs_separator.model_name,
            "demucs_sample_rate": self.demucs_separator.sample_rate
        })
        
        return SeparationResult(
            vocals=vocals,
            accompaniment=accompaniment,
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate,
            metadata=metadata
        )
    
    def _separate_with_spectral(self, audio: np.ndarray, sample_rate: int,
                               metadata: Dict[str, Any], **kwargs) -> SeparationResult:
        """Separate vocals using spectral subtraction."""
        enhanced_audio = self.spectral_separator.separate_vocals(audio, sample_rate)
        
        metadata.update({
            "alpha": self.spectral_separator.alpha,
            "beta": self.spectral_separator.beta
        })
        
        return SeparationResult(
            vocals=enhanced_audio,
            accompaniment=audio - enhanced_audio,  # Approximate accompaniment
            sample_rate=sample_rate,
            duration=len(audio) / sample_rate,
            metadata=metadata
        )
    
    def separate_sources(self, audio: np.ndarray, sample_rate: int,
                        target_sources: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Separate multiple audio sources.
        
        Args:
            audio: Audio array (1D, float32)
            sample_rate: Audio sample rate
            target_sources: List of sources to extract
            
        Returns:
            Dictionary mapping source names to separated audio arrays
        """
        if target_sources is None:
            target_sources = ['vocals', 'drums', 'bass', 'other']
        
        if self.demucs_separator:
            return self.demucs_separator.separate_sources(audio, sample_rate, target_sources)
        else:
            # Fallback: only vocals available with spectral subtraction
            result = self.separate_vocals(audio, sample_rate)
            return {'vocals': result.vocals}
    
    def get_separation_quality_metrics(self, original: np.ndarray, 
                                     separated: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics for separation results.
        
        Args:
            original: Original audio array
            separated: Separated audio array
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Signal-to-noise ratio
            signal_power = np.mean(separated ** 2)
            noise_power = np.mean((original - separated) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Spectral distance
            import librosa
            orig_stft = librosa.stft(original, n_fft=2048)
            sep_stft = librosa.stft(separated, n_fft=2048)
            
            orig_mag = np.abs(orig_stft)
            sep_mag = np.abs(sep_stft)
            
            spectral_distance = np.mean(np.abs(orig_mag - sep_mag))
            
            return {
                "snr_db": snr,
                "spectral_distance": spectral_distance,
                "signal_power": signal_power,
                "noise_power": noise_power
            }
        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")
            return {"error": str(e)}


# Global separator instance
_separator_instance = None


def get_separator_instance() -> VocalSeparator:
    """Get or create global separator instance."""
    global _separator_instance
    if _separator_instance is None:
        _separator_instance = VocalSeparator()
    return _separator_instance
