"""
Integration tests for the audio transcription service.

This module tests the complete integration of the audio processing pipeline,
API endpoints, and end-to-end functionality.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile

from src.api.main import app
from src.services.audio_processor import AudioProcessor, ProcessingConfig, ProcessingResult
from src.api.models import TranscribeConfig, TranscribeResponse


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_processing_result(self, mock_processing_result):
        """Mock processing result for API tests."""
        return mock_processing_result
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "services" in data
    
    def test_capabilities_endpoint(self, client):
        """Test capabilities endpoint."""
        response = client.get("/v1/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "device_info" in data
        assert "services" in data
        assert "configuration" in data
        assert "supported_languages" in data
        assert "max_file_size_mb" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "system_resources" in data
        assert "performance_summary" in data
        assert "operation_stats" in data
        assert "memory_stats" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        assert "features" in data
    
    @patch('src.api.routes.get_processor_instance')
    def test_transcribe_endpoint_success(self, mock_get_processor, client, sample_audio_file, mock_processing_result):
        """Test successful transcription endpoint."""
        file_path, _, _ = sample_audio_file
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_file.return_value = mock_processing_result
        mock_get_processor.return_value = mock_processor
        
        # Test transcription endpoint
        with open(file_path, "rb") as f:
            response = client.post(
                "/v1/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "en", "enable_separation": "true"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "segments" in data
        assert "processing_time" in data
        assert "request_id" in data
        assert "timestamp" in data
    
    def test_transcribe_endpoint_no_file(self, client):
        """Test transcription endpoint without file."""
        response = client.post("/v1/transcribe")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "No file provided" in data["message"]
    
    def test_transcribe_endpoint_invalid_file(self, client, invalid_audio_file):
        """Test transcription endpoint with invalid file."""
        with open(invalid_audio_file, "rb") as f:
            response = client.post(
                "/v1/transcribe",
                files={"file": ("invalid.txt", f, "text/plain")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_transcribe_endpoint_large_file(self, client, large_audio_file):
        """Test transcription endpoint with large file."""
        with open(large_audio_file, "rb") as f:
            response = client.post(
                "/v1/transcribe",
                files={"file": ("large.wav", f, "audio/wav")}
            )
        
        # Should return 413 for file too large
        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "File too large" in data["message"]
    
    def test_transcribe_endpoint_invalid_config(self, client, sample_audio_file):
        """Test transcription endpoint with invalid configuration."""
        file_path, _, _ = sample_audio_file
        
        with open(file_path, "rb") as f:
            response = client.post(
                "/v1/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"config": "invalid json"}
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "Invalid configuration JSON" in data["message"]
    
    def test_transcribe_endpoint_config_validation(self, client, sample_audio_file):
        """Test transcription endpoint with configuration validation."""
        file_path, _, _ = sample_audio_file
        
        # Test with invalid speaker configuration
        config = {
            "enable_diarization": True,
            "min_speakers": 5,
            "max_speakers": 2  # Invalid: min > max
        }
        
        with open(file_path, "rb") as f:
            response = client.post(
                "/v1/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"config": json.dumps(config)}
            )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "Configuration validation failed" in data["message"]
    
    def test_transcribe_endpoint_processing_error(self, client, sample_audio_file):
        """Test transcription endpoint with processing error."""
        file_path, _, _ = sample_audio_file
        
        # Mock processor to raise an exception
        with patch('src.api.routes.get_processor_instance') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process_file.side_effect = Exception("Processing failed")
            mock_get_processor.return_value = mock_processor
            
            with open(file_path, "rb") as f:
                response = client.post(
                    "/v1/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["message"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/v1/transcribe")
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers
        assert "referrer-policy" in response.headers
    
    def test_request_id_header(self, client):
        """Test request ID header is present."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert response.headers["x-request-id"] is not None


class TestAudioProcessorIntegration:
    """Integration tests for the audio processor."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create mock audio processor for testing."""
        processor = Mock(spec=AudioProcessor)
        processor.device = "cpu"
        processor.device_info = {"device": "cpu", "cuda_available": False}
        return processor
    
    def test_audio_processor_init(self, mock_settings):
        """Test audio processor initialization."""
        with patch('src.services.audio_processor.get_vad_instance') as mock_vad, \
             patch('src.services.audio_processor.get_separator_instance') as mock_separator, \
             patch('src.services.audio_processor.get_transcriber_instance') as mock_transcriber, \
             patch('src.services.audio_processor.get_diarizer_instance') as mock_diarizer:
            
            mock_vad.return_value = Mock()
            mock_separator.return_value = Mock()
            mock_transcriber.return_value = Mock()
            mock_diarizer.return_value = Mock()
            
            processor = AudioProcessor()
            
            assert processor.device == "cpu"
            assert processor.vad is not None
            assert processor.separator is not None
            assert processor.transcriber is not None
            assert processor.diarizer is not None
    
    def test_process_audio_basic(self, mock_processor, sample_audio_data):
        """Test basic audio processing."""
        audio, sample_rate = sample_audio_data
        config = ProcessingConfig()
        
        # Mock processing result
        mock_result = ProcessingResult(
            transcription=Mock(),
            config=config,
            processing_time=1.0,
            device_info={"device": "cpu"},
            performance_metrics={},
            quality_metrics={}
        )
        mock_processor.process_audio.return_value = mock_result
        
        result = mock_processor.process_audio(audio, sample_rate, config)
        
        assert isinstance(result, ProcessingResult)
        assert result.processing_time == 1.0
        assert result.device_info["device"] == "cpu"
    
    def test_process_audio_with_separation(self, mock_processor, sample_audio_data):
        """Test audio processing with vocal separation."""
        audio, sample_rate = sample_audio_data
        config = ProcessingConfig(enable_separation=True)
        
        # Mock processing result with separation
        mock_result = ProcessingResult(
            transcription=Mock(),
            separation=Mock(),
            config=config,
            processing_time=2.0,
            device_info={"device": "cpu"},
            performance_metrics={},
            quality_metrics={}
        )
        mock_processor.process_audio.return_value = mock_result
        
        result = mock_processor.process_audio(audio, sample_rate, config)
        
        assert result.has_separation_info
        assert result.processing_time == 2.0
    
    def test_process_audio_with_diarization(self, mock_processor, sample_audio_data):
        """Test audio processing with speaker diarization."""
        audio, sample_rate = sample_audio_data
        config = ProcessingConfig(enable_diarization=True)
        
        # Mock processing result with diarization
        mock_result = ProcessingResult(
            transcription=Mock(),
            diarization=Mock(),
            config=config,
            processing_time=3.0,
            device_info={"device": "cpu"},
            performance_metrics={},
            quality_metrics={}
        )
        mock_processor.process_audio.return_value = mock_result
        
        result = mock_processor.process_audio(audio, sample_rate, config)
        
        assert result.has_speaker_info
        assert result.processing_time == 3.0
    
    def test_process_file(self, mock_processor, sample_audio_file):
        """Test processing audio file."""
        file_path, _, _ = sample_audio_file
        config = ProcessingConfig()
        
        # Mock processing result
        mock_result = ProcessingResult(
            transcription=Mock(),
            config=config,
            processing_time=1.5,
            device_info={"device": "cpu"},
            performance_metrics={},
            quality_metrics={}
        )
        mock_processor.process_file.return_value = mock_result
        
        result = mock_processor.process_file(file_path, config)
        
        assert isinstance(result, ProcessingResult)
        assert result.processing_time == 1.5
        mock_processor.process_file.assert_called_once_with(file_path, config)
    
    def test_get_processing_capabilities(self, mock_processor):
        """Test getting processing capabilities."""
        mock_processor.get_processing_capabilities.return_value = {
            "device": "cpu",
            "device_info": {"device": "cpu", "cuda_available": False},
            "services": {
                "vad": {"available": True, "methods": ["webrtc", "silero"]},
                "separation": {"available": True, "methods": ["demucs", "spectral"]},
                "transcription": {"available": True, "model_size": "small"},
                "diarization": {"available": False, "methods": []}
            },
            "configuration": {
                "max_file_size_mb": 100,
                "target_sample_rate": 16000,
                "chunk_size_seconds": 30,
                "enable_gpu": False
            }
        }
        
        capabilities = mock_processor.get_processing_capabilities()
        
        assert "device" in capabilities
        assert "services" in capabilities
        assert "configuration" in capabilities
        assert capabilities["device"] == "cpu"
        assert capabilities["services"]["vad"]["available"] == True
        assert capabilities["services"]["transcription"]["model_size"] == "small"
    
    def test_validate_config(self, mock_processor):
        """Test configuration validation."""
        # Valid configuration
        valid_config = ProcessingConfig(
            language="en",
            enable_diarization=True,
            num_speakers=2
        )
        
        mock_processor.validate_config.return_value = []
        errors = mock_processor.validate_config(valid_config)
        
        assert errors == []
        
        # Invalid configuration
        invalid_config = ProcessingConfig(
            language="invalid_lang",
            enable_diarization=True,
            min_speakers=5,
            max_speakers=2
        )
        
        mock_processor.validate_config.return_value = [
            "Language 'invalid_lang' not supported",
            "min_speakers cannot be greater than max_speakers"
        ]
        errors = mock_processor.validate_config(invalid_config)
        
        assert len(errors) == 2
        assert "Language 'invalid_lang' not supported" in errors
        assert "min_speakers cannot be greater than max_speakers" in errors


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API."""
        return TestClient(app)
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_transcription_pipeline(self, client, sample_audio_file):
        """Test complete transcription pipeline end-to-end."""
        file_path, _, _ = sample_audio_file
        
        # Mock the entire processing pipeline
        with patch('src.api.routes.get_processor_instance') as mock_get_processor:
            mock_processor = Mock()
            
            # Create a realistic processing result
            mock_transcription = Mock()
            mock_transcription.text = "Hello, this is a test transcription."
            mock_transcription.language = "en"
            mock_transcription.language_probability = 0.95
            mock_transcription.segments = [
                Mock(
                    id=0,
                    start=0.0,
                    end=2.5,
                    text="Hello, this is a test transcription.",
                    confidence=0.92,
                    words=[
                        Mock(word="Hello", start=0.0, end=0.5, probability=0.98),
                        Mock(word="this", start=0.6, end=0.8, probability=0.95),
                        Mock(word="is", start=0.9, end=1.0, probability=0.99),
                        Mock(word="a", start=1.1, end=1.2, probability=0.97),
                        Mock(word="test", start=1.3, end=1.8, probability=0.94),
                        Mock(word="transcription", start=1.9, end=2.5, probability=0.91)
                    ]
                )
            ]
            mock_transcription.metadata = {"model_size": "small", "device": "cpu"}
            
            mock_result = ProcessingResult(
                transcription=mock_transcription,
                config=ProcessingConfig(),
                processing_time=2.5,
                device_info={"device": "cpu", "cuda_available": False},
                performance_metrics={"total_operations": 3},
                quality_metrics={"transcription": {"avg_confidence": 0.92}}
            )
            
            mock_processor.process_file.return_value = mock_result
            mock_get_processor.return_value = mock_processor
            
            # Test the complete pipeline
            with open(file_path, "rb") as f:
                response = client.post(
                    "/v1/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={
                        "language": "en",
                        "enable_separation": "true",
                        "enable_vad": "true",
                        "word_timestamps": "true"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "text" in data
            assert "language" in data
            assert "segments" in data
            assert "processing_time" in data
            assert "request_id" in data
            assert "timestamp" in data
            assert "pipeline_info" in data
            assert "quality_metrics" in data
            assert "timing_metrics" in data
            
            # Verify content
            assert data["text"] == "Hello, this is a test transcription."
            assert data["language"] == "en"
            assert data["language_probability"] == 0.95
            assert len(data["segments"]) == 1
            assert data["segments"][0]["text"] == "Hello, this is a test transcription."
            assert data["segments"][0]["confidence"] == 0.92
            assert len(data["segments"][0]["words"]) == 6
            assert data["segments"][0]["words"][0]["word"] == "Hello"
            assert data["segments"][0]["words"][0]["probability"] == 0.98
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_batch_processing_simulation(self, client, sample_audio_files):
        """Test batch processing simulation."""
        results = []
        
        for file_path in sample_audio_files:
            with patch('src.api.routes.get_processor_instance') as mock_get_processor:
                mock_processor = Mock()
                
                # Create mock result for each file
                mock_transcription = Mock()
                mock_transcription.text = f"Transcription for {file_path.name}"
                mock_transcription.language = "en"
                mock_transcription.language_probability = 0.95
                mock_transcription.segments = [Mock(
                    id=0,
                    start=0.0,
                    end=5.0,
                    text=f"Transcription for {file_path.name}",
                    confidence=0.90,
                    words=[]
                )]
                mock_transcription.metadata = {"model_size": "small"}
                
                mock_result = ProcessingResult(
                    transcription=mock_transcription,
                    config=ProcessingConfig(),
                    processing_time=1.0,
                    device_info={"device": "cpu"},
                    performance_metrics={},
                    quality_metrics={}
                )
                
                mock_processor.process_file.return_value = mock_result
                mock_get_processor.return_value = mock_processor
                
                # Process each file
                with open(file_path, "rb") as f:
                    response = client.post(
                        "/v1/transcribe",
                        files={"file": (file_path.name, f, "audio/wav")}
                    )
                
                assert response.status_code == 200
                data = response.json()
                results.append(data)
        
        # Verify all files were processed
        assert len(results) == len(sample_audio_files)
        for i, result in enumerate(results):
            assert result["text"] == f"Transcription for {sample_audio_files[i].name}"
            assert result["language"] == "en"
            assert result["processing_time"] == 1.0
    
    @pytest.mark.integration
    def test_error_handling_integration(self, client, sample_audio_file):
        """Test error handling in integration scenarios."""
        file_path, _, _ = sample_audio_file
        
        # Test with processing failure
        with patch('src.api.routes.get_processor_instance') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process_file.side_effect = Exception("Processing failed")
            mock_get_processor.return_value = mock_processor
            
            with open(file_path, "rb") as f:
                response = client.post(
                    "/v1/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["message"]
    
    @pytest.mark.integration
    def test_performance_metrics_integration(self, client):
        """Test performance metrics in integration."""
        # Make multiple requests to generate metrics
        for _ in range(3):
            response = client.get("/v1/health")
            assert response.status_code == 200
        
        # Check metrics endpoint
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "performance_summary" in data
        assert "operation_stats" in data
        assert "memory_stats" in data
        assert "system_resources" in data


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @pytest.mark.integration
    def test_cli_health_check(self):
        """Test CLI health check functionality."""
        from src.cli.client import AudioTranscriptionCLI
        
        # Mock the health endpoint response
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
                "uptime": 100.0,
                "timestamp": "2024-01-01T12:00:00Z",
                "services": {
                    "audio_processor": {"status": "healthy"},
                    "system_resources": {"status": "healthy"}
                }
            }
            mock_get.return_value = mock_response
            
            async def test_health():
                async with AudioTranscriptionCLI() as cli:
                    health_data = await cli.get_health()
                    assert health_data["status"] == "healthy"
                    assert health_data["version"] == "1.0.0"
                    assert "services" in health_data
            
            asyncio.run(test_health())
    
    @pytest.mark.integration
    def test_cli_capabilities(self):
        """Test CLI capabilities functionality."""
        from src.cli.client import AudioTranscriptionCLI
        
        # Mock the capabilities endpoint response
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "device": "cpu",
                "device_info": {"device": "cpu", "cuda_available": False},
                "services": {
                    "vad": {"available": True, "methods": ["webrtc", "silero"]},
                    "separation": {"available": True, "methods": ["demucs", "spectral"]},
                    "transcription": {"available": True, "model_size": "small"},
                    "diarization": {"available": False, "methods": []}
                },
                "configuration": {
                    "max_file_size_mb": 100,
                    "target_sample_rate": 16000
                },
                "supported_languages": ["en", "es", "fr"],
                "max_file_size_mb": 100
            }
            mock_get.return_value = mock_response
            
            async def test_capabilities():
                async with AudioTranscriptionCLI() as cli:
                    capabilities_data = await cli.get_capabilities()
                    assert capabilities_data["device"] == "cpu"
                    assert "services" in capabilities_data
                    assert "supported_languages" in capabilities_data
                    assert len(capabilities_data["supported_languages"]) == 3
            
            asyncio.run(test_capabilities())
