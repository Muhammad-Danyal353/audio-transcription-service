"""
Command-line interface client for the audio transcription service.

This module provides a comprehensive CLI tool for interacting with the
audio transcription API, supporting various commands and output formats.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import os

import httpx
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from ..core.config import settings
from ..core.logging import get_logger
from ..api.models import TranscribeConfig, TranscribeResponse, ErrorResponse

logger = get_logger(__name__)
console = Console()


class AudioTranscriptionCLI:
    """
    Main CLI client for the audio transcription service.
    
    Provides a comprehensive command-line interface with support for
    various commands, output formats, and configuration options.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize CLI client.
        
        Args:
            base_url: Base URL of the transcription service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    async def transcribe_file(
        self, 
        file_path: Path, 
        config: Optional[TranscribeConfig] = None,
        output_format: str = "json",
        output_file: Optional[Path] = None,
        show_progress: bool = True
    ) -> TranscribeResponse:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            config: Transcription configuration
            output_format: Output format (json, text, srt, vtt)
            output_file: Output file path (optional)
            show_progress: Whether to show progress indicator
            
        Returns:
            TranscribeResponse with transcription results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if config is None:
            config = TranscribeConfig()
        
        # Prepare form data
        files = {"file": open(file_path, "rb")}
        data = {}
        
        # Add configuration parameters
        if config.enable_separation is not None:
            data["enable_separation"] = config.enable_separation
        if config.enable_vad is not None:
            data["enable_vad"] = config.enable_vad
        if config.enable_diarization is not None:
            data["enable_diarization"] = config.enable_diarization
        if config.language:
            data["language"] = config.language
        if config.task:
            data["task"] = config.task
        if config.word_timestamps is not None:
            data["word_timestamps"] = config.word_timestamps
        if config.separation_method:
            data["separation_method"] = config.separation_method
        if config.vad_method:
            data["vad_method"] = config.vad_method
        if config.vad_threshold is not None:
            data["vad_threshold"] = config.vad_threshold
        if config.num_speakers:
            data["num_speakers"] = config.num_speakers
        if config.min_speakers:
            data["min_speakers"] = config.min_speakers
        if config.max_speakers:
            data["max_speakers"] = config.max_speakers
        if config.chunk_long_audio is not None:
            data["chunk_long_audio"] = config.chunk_long_audio
        if config.chunk_size_seconds:
            data["chunk_size_seconds"] = config.chunk_size_seconds
        if config.chunk_overlap_seconds:
            data["chunk_overlap_seconds"] = config.chunk_overlap_seconds
        
        try:
            # Show progress if requested
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Transcribing audio...", total=None)
                    
                    response = await self.client.post(
                        f"{self.base_url}/v1/transcribe",
                        files=files,
                        data=data
                    )
            else:
                response = await self.client.post(
                    f"{self.base_url}/v1/transcribe",
                    files=files,
                    data=data
                )
            
            # Handle response
            if response.status_code == 200:
                result = TranscribeResponse(**response.json())
                
                # Save output if requested
                if output_file:
                    await self._save_output(result, output_format, output_file)
                
                return result
            else:
                error_data = response.json()
                raise Exception(f"Transcription failed: {error_data.get('message', 'Unknown error')}")
                
        finally:
            files["file"].close()
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        response = await self.client.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return response.json()
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get service capabilities."""
        response = await self.client.get(f"{self.base_url}/v1/capabilities")
        response.raise_for_status()
        return response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        response = await self.client.get(f"{self.base_url}/v1/metrics")
        response.raise_for_status()
        return response.json()
    
    async def _save_output(
        self, 
        result: TranscribeResponse, 
        output_format: str, 
        output_file: Path
    ) -> None:
        """Save transcription result to file in specified format."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.dict(), f, indent=2, ensure_ascii=False)
        
        elif output_format == "text":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.text)
        
        elif output_format == "srt":
            srt_content = self._format_as_srt(result)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
        
        elif output_format == "vtt":
            vtt_content = self._format_as_vtt(result)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(vtt_content)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _format_as_srt(self, result: TranscribeResponse) -> str:
        """Format transcription result as SRT subtitle file."""
        srt_lines = []
        
        for i, segment in enumerate(result.segments, 1):
            start_time = self._format_srt_time(segment.start)
            end_time = self._format_srt_time(segment.end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")  # Empty line between segments
        
        return "\n".join(srt_lines)
    
    def _format_as_vtt(self, result: TranscribeResponse) -> str:
        """Format transcription result as VTT subtitle file."""
        vtt_lines = ["WEBVTT", ""]
        
        for segment in result.segments:
            start_time = self._format_vtt_time(segment.start)
            end_time = self._format_vtt_time(segment.end)
            
            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(segment.text.strip())
            vtt_lines.append("")  # Empty line between segments
        
        return "\n".join(vtt_lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def create_config_from_args(args: argparse.Namespace) -> TranscribeConfig:
    """Create TranscribeConfig from command line arguments."""
    config_dict = {}
    
    # Feature flags
    if args.enable_separation is not None:
        config_dict["enable_separation"] = args.enable_separation
    if args.enable_vad is not None:
        config_dict["enable_vad"] = args.enable_vad
    if args.enable_diarization is not None:
        config_dict["enable_diarization"] = args.enable_diarization
    
    # Processing parameters
    if args.language:
        config_dict["language"] = args.language
    if args.task:
        config_dict["task"] = args.task
    if args.word_timestamps is not None:
        config_dict["word_timestamps"] = args.word_timestamps
    
    # Method parameters
    if args.separation_method:
        config_dict["separation_method"] = args.separation_method
    if args.vad_method:
        config_dict["vad_method"] = args.vad_method
    if args.vad_threshold is not None:
        config_dict["vad_threshold"] = args.vad_threshold
    
    # Diarization parameters
    if args.num_speakers:
        config_dict["num_speakers"] = args.num_speakers
    if args.min_speakers:
        config_dict["min_speakers"] = args.min_speakers
    if args.max_speakers:
        config_dict["max_speakers"] = args.max_speakers
    
    # Chunking parameters
    if args.chunk_long_audio is not None:
        config_dict["chunk_long_audio"] = args.chunk_long_audio
    if args.chunk_size:
        config_dict["chunk_size_seconds"] = args.chunk_size
    if args.chunk_overlap:
        config_dict["chunk_overlap_seconds"] = args.chunk_overlap
    
    return TranscribeConfig(**config_dict)


def display_transcription_result(result: TranscribeResponse, format_type: str = "rich"):
    """Display transcription result in specified format."""
    
    if format_type == "json":
        console.print(Syntax(json.dumps(result.dict(), indent=2), "json"))
    
    elif format_type == "text":
        console.print(result.text)
    
    elif format_type == "rich":
        # Create a rich panel with transcription results
        panel_content = f"[bold green]Transcription Results[/bold green]\n\n"
        panel_content += f"[bold]Text:[/bold] {result.text}\n"
        panel_content += f"[bold]Language:[/bold] {result.language} (confidence: {result.language_probability:.2%})\n"
        panel_content += f"[bold]Processing Time:[/bold] {result.processing_time:.2f}s\n"
        panel_content += f"[bold]Segments:[/bold] {len(result.segments)}\n"
        
        if result.has_speaker_info:
            panel_content += f"[bold]Speakers:[/bold] {len(set(seg.speaker for seg in result.speaker_segments))}\n"
        
        if result.has_word_timestamps:
            panel_content += f"[bold]Word Timestamps:[/bold] Yes\n"
        
        console.print(Panel(panel_content, title="Transcription Complete", border_style="green"))
        
        # Show segments table
        if result.segments:
            table = Table(title="Transcription Segments")
            table.add_column("ID", style="cyan")
            table.add_column("Start", style="magenta")
            table.add_column("End", style="magenta")
            table.add_column("Duration", style="yellow")
            table.add_column("Text", style="white")
            table.add_column("Confidence", style="green")
            
            for segment in result.segments:
                table.add_row(
                    str(segment.id),
                    f"{segment.start:.2f}s",
                    f"{segment.end:.2f}s",
                    f"{segment.duration:.2f}s",
                    segment.text[:50] + "..." if len(segment.text) > 50 else segment.text,
                    f"{segment.confidence:.2%}"
                )
            
            console.print(table)
    
    elif format_type == "table":
        # Simple table format
        table = Table(title="Transcription Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Text", result.text)
        table.add_row("Language", f"{result.language} ({result.language_probability:.2%})")
        table.add_row("Processing Time", f"{result.processing_time:.2f}s")
        table.add_row("Segments", str(len(result.segments)))
        table.add_row("Word Timestamps", "Yes" if result.has_word_timestamps else "No")
        table.add_row("Speaker Info", "Yes" if result.has_speaker_info else "No")
        
        console.print(table)


def display_health_status(health_data: Dict[str, Any]):
    """Display health status in a rich format."""
    
    status_color = "green" if health_data["status"] == "healthy" else "red"
    
    panel_content = f"[bold {status_color}]Status:[/bold {status_color}] {health_data['status'].upper()}\n"
    panel_content += f"[bold]Version:[/bold] {health_data['version']}\n"
    panel_content += f"[bold]Uptime:[/bold] {health_data['uptime']:.2f}s\n"
    panel_content += f"[bold]Timestamp:[/bold] {health_data['timestamp']}\n"
    
    console.print(Panel(panel_content, title="Service Health", border_style=status_color))
    
    # Show services status
    if "services" in health_data:
        table = Table(title="Service Components")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="yellow")
        
        for service_name, service_info in health_data["services"].items():
            status = service_info.get("status", "unknown")
            status_color = "green" if status == "healthy" else "red" if status == "unhealthy" else "yellow"
            
            details = ""
            if "error" in service_info:
                details = service_info["error"]
            elif "device" in service_info:
                details = f"Device: {service_info['device']}"
            
            table.add_row(
                service_name.replace("_", " ").title(),
                f"[{status_color}]{status}[/{status_color}]",
                details
            )
        
        console.print(table)


def display_capabilities(capabilities_data: Dict[str, Any]):
    """Display service capabilities in a rich format."""
    
    panel_content = f"[bold]Device:[/bold] {capabilities_data['device']}\n"
    panel_content += f"[bold]Max File Size:[/bold] {capabilities_data['max_file_size_mb']}MB\n"
    panel_content += f"[bold]Supported Languages:[/bold] {len(capabilities_data['supported_languages'])}\n"
    
    console.print(Panel(panel_content, title="Service Capabilities", border_style="blue"))
    
    # Show services
    if "services" in capabilities_data:
        table = Table(title="Available Services")
        table.add_column("Service", style="cyan")
        table.add_column("Available", style="white")
        table.add_column("Methods", style="yellow")
        
        for service_name, service_info in capabilities_data["services"].items():
            available = "Yes" if service_info.get("available", False) else "No"
            methods = ", ".join(service_info.get("methods", []))
            
            table.add_row(
                service_name.replace("_", " ").title(),
                available,
                methods or "N/A"
            )
        
        console.print(table)
    
    # Show supported languages
    if capabilities_data["supported_languages"]:
        languages = capabilities_data["supported_languages"][:20]  # Show first 20
        languages_text = ", ".join(languages)
        if len(capabilities_data["supported_languages"]) > 20:
            languages_text += f" ... and {len(capabilities_data['supported_languages']) - 20} more"
        
        console.print(f"\n[bold]Supported Languages:[/bold] {languages_text}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Transcription Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python -m src.cli.client --file audio.wav
  
  # Transcription with vocal separation
  python -m src.cli.client --file audio.wav --enable-separation
  
  # Transcription with speaker diarization
  python -m src.cli.client --file audio.wav --enable-diarization --num-speakers 2
  
  # Save as SRT subtitle file
  python -m src.cli.client --file audio.wav --output-format srt --output-file subtitles.srt
  
  # Check service health
  python -m src.cli.client --health
  
  # Get service capabilities
  python -m src.cli.client --capabilities
        """
    )
    
    # Main command
    parser.add_argument("--file", "-f", type=Path, help="Audio file to transcribe")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Request timeout in seconds")
    
    # Output options
    parser.add_argument("--output-format", "-o", choices=["json", "text", "srt", "vtt", "rich", "table"], 
                       default="rich", help="Output format")
    parser.add_argument("--output-file", type=Path, help="Output file path")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress indicator")
    
    # Feature flags
    parser.add_argument("--enable-separation", action="store_true", help="Enable vocal separation")
    parser.add_argument("--disable-separation", action="store_true", help="Disable vocal separation")
    parser.add_argument("--enable-vad", action="store_true", help="Enable voice activity detection")
    parser.add_argument("--disable-vad", action="store_true", help="Disable voice activity detection")
    parser.add_argument("--enable-diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--disable-diarization", action="store_true", help="Disable speaker diarization")
    
    # Processing parameters
    parser.add_argument("--language", "-l", help="Language code (e.g., en, es, fr)")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", help="Task type")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--no-word-timestamps", action="store_true", help="Disable word-level timestamps")
    
    # Method parameters
    parser.add_argument("--separation-method", choices=["demucs", "spectral", "auto"], help="Separation method")
    parser.add_argument("--vad-method", choices=["webrtc", "silero", "auto"], help="VAD method")
    parser.add_argument("--vad-threshold", type=float, help="VAD threshold (0.0-1.0)")
    
    # Diarization parameters
    parser.add_argument("--num-speakers", type=int, help="Expected number of speakers")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    
    # Chunking parameters
    parser.add_argument("--chunk-long-audio", action="store_true", help="Enable chunking for long audio")
    parser.add_argument("--no-chunk-long-audio", action="store_true", help="Disable chunking for long audio")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in seconds")
    parser.add_argument("--chunk-overlap", type=float, help="Chunk overlap in seconds")
    
    # Service commands
    parser.add_argument("--health", action="store_true", help="Check service health")
    parser.add_argument("--capabilities", action="store_true", help="Get service capabilities")
    parser.add_argument("--metrics", action="store_true", help="Get service metrics")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.file, args.health, args.capabilities, args.metrics]):
        parser.error("Must specify either --file, --health, --capabilities, or --metrics")
    
    if args.file and not args.file.exists():
        parser.error(f"File not found: {args.file}")
    
    # Handle conflicting arguments
    if args.enable_separation and args.disable_separation:
        parser.error("Cannot specify both --enable-separation and --disable-separation")
    if args.enable_vad and args.disable_vad:
        parser.error("Cannot specify both --enable-vad and --disable-vad")
    if args.enable_diarization and args.disable_diarization:
        parser.error("Cannot specify both --enable-diarization and --disable-diarization")
    if args.word_timestamps and args.no_word_timestamps:
        parser.error("Cannot specify both --word-timestamps and --no-word-timestamps")
    if args.chunk_long_audio and args.no_chunk_long_audio:
        parser.error("Cannot specify both --chunk-long-audio and --no-chunk-long-audio")
    
    try:
        async with AudioTranscriptionCLI(args.url, args.timeout) as cli:
            
            if args.health:
                health_data = await cli.get_health()
                display_health_status(health_data)
            
            elif args.capabilities:
                capabilities_data = await cli.get_capabilities()
                display_capabilities(capabilities_data)
            
            elif args.metrics:
                metrics_data = await cli.get_metrics()
                console.print(Syntax(json.dumps(metrics_data, indent=2), "json"))
            
            elif args.file:
                # Create configuration
                config = create_config_from_args(args)
                
                # Transcribe file
                result = await cli.transcribe_file(
                    args.file,
                    config,
                    args.output_format,
                    args.output_file,
                    not args.no_progress
                )
                
                # Display result
                display_transcription_result(result, args.output_format)
                
                if args.output_file:
                    console.print(f"\n[green]Output saved to: {args.output_file}[/green]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
