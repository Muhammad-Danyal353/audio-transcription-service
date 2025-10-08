"""
Advanced CLI client with click for better user experience.

This module provides an enhanced command-line interface using click
with better argument parsing, help text, and interactive features.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import os

import click
import httpx
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


class AdvancedAudioTranscriptionCLI:
    """
    Advanced CLI client with enhanced features.
    
    Provides a comprehensive command-line interface with support for
    configuration files, batch processing, and interactive modes.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize advanced CLI client.
        
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
        """Transcribe an audio file with enhanced error handling."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if config is None:
            config = TranscribeConfig()
        
        # Prepare form data
        files = {"file": open(file_path, "rb")}
        data = {}
        
        # Add configuration parameters
        for field_name, field_value in config.dict().items():
            if field_value is not None:
                data[field_name] = field_value
        
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
    
    async def batch_transcribe(
        self,
        file_paths: List[Path],
        config: Optional[TranscribeConfig] = None,
        output_format: str = "json",
        output_dir: Optional[Path] = None,
        show_progress: bool = True
    ) -> List[TranscribeResponse]:
        """Transcribe multiple files in batch."""
        results = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Batch transcribing...", total=len(file_paths))
                
                for i, file_path in enumerate(file_paths):
                    progress.update(task, description=f"Processing {file_path.name}")
                    
                    try:
                        result = await self.transcribe_file(
                            file_path, config, output_format, None, False
                        )
                        results.append(result)
                        
                        # Save individual output if output directory specified
                        if output_dir:
                            output_file = output_dir / f"{file_path.stem}.{output_format}"
                            await self._save_output(result, output_format, output_file)
                        
                    except Exception as e:
                        console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                        results.append(None)
                    
                    progress.advance(task)
        else:
            for file_path in file_paths:
                try:
                    result = await self.transcribe_file(
                        file_path, config, output_format, None, False
                    )
                    results.append(result)
                    
                    if output_dir:
                        output_file = output_dir / f"{file_path.stem}.{output_format}"
                        await self._save_output(result, output_format, output_file)
                        
                except Exception as e:
                    console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                    results.append(None)
        
        return results
    
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


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config_file(config: TranscribeConfig, config_path: Path) -> None:
    """Save configuration to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.dict(), f, indent=2, ensure_ascii=False)


def create_config_from_options(**kwargs) -> TranscribeConfig:
    """Create TranscribeConfig from click options."""
    # Filter out None values and convert to proper types
    config_dict = {k: v for k, v in kwargs.items() if v is not None}
    
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


# Click CLI Commands

@click.group()
@click.option('--url', '-u', default='http://localhost:8000', help='Service URL')
@click.option('--timeout', '-t', type=int, default=300, help='Request timeout in seconds')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, url, timeout, verbose):
    """Audio Transcription Service CLI - Advanced interface for audio transcription."""
    ctx.ensure_object(dict)
    ctx.obj['url'] = url
    ctx.obj['timeout'] = timeout
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('file', type=click.Path(exists=True, path_type=Path))
@click.option('--output-format', '-o', 
              type=click.Choice(['json', 'text', 'srt', 'vtt', 'rich', 'table']), 
              default='rich', help='Output format')
@click.option('--output-file', type=click.Path(path_type=Path), help='Output file path')
@click.option('--language', '-l', help='Language code (e.g., en, es, fr)')
@click.option('--task', type=click.Choice(['transcribe', 'translate']), 
              default='transcribe', help='Task type')
@click.option('--enable-separation', is_flag=True, help='Enable vocal separation')
@click.option('--enable-vad', is_flag=True, help='Enable voice activity detection')
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--num-speakers', type=int, help='Expected number of speakers')
@click.option('--word-timestamps', is_flag=True, help='Include word-level timestamps')
@click.option('--no-progress', is_flag=True, help='Disable progress indicator')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file path')
@click.pass_context
async def transcribe(ctx, file, output_format, output_file, language, task, 
                    enable_separation, enable_vad, enable_diarization, 
                    num_speakers, word_timestamps, no_progress, config):
    """Transcribe an audio file to text."""
    
    # Load configuration from file if provided
    config_dict = {}
    if config:
        config_dict = load_config_file(config)
    
    # Override with command line options
    if language:
        config_dict['language'] = language
    if task:
        config_dict['task'] = task
    if enable_separation:
        config_dict['enable_separation'] = True
    if enable_vad:
        config_dict['enable_vad'] = True
    if enable_diarization:
        config_dict['enable_diarization'] = True
    if num_speakers:
        config_dict['num_speakers'] = num_speakers
    if word_timestamps:
        config_dict['word_timestamps'] = True
    
    transcribe_config = TranscribeConfig(**config_dict)
    
    try:
        async with AdvancedAudioTranscriptionCLI(ctx.obj['url'], ctx.obj['timeout']) as cli_client:
            result = await cli_client.transcribe_file(
                file, transcribe_config, output_format, output_file, not no_progress
            )
            
            display_transcription_result(result, output_format)
            
            if output_file:
                console.print(f"\n[green]Output saved to: {output_file}[/green]")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--output-format', '-o', 
              type=click.Choice(['json', 'text', 'srt', 'vtt']), 
              default='json', help='Output format')
@click.option('--output-dir', type=click.Path(path_type=Path), 
              help='Output directory for batch results')
@click.option('--language', '-l', help='Language code (e.g., en, es, fr)')
@click.option('--enable-separation', is_flag=True, help='Enable vocal separation')
@click.option('--enable-vad', is_flag=True, help='Enable voice activity detection')
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--num-speakers', type=int, help='Expected number of speakers')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file path')
@click.pass_context
async def batch(ctx, files, output_format, output_dir, language, 
               enable_separation, enable_vad, enable_diarization, 
               num_speakers, config):
    """Transcribe multiple audio files in batch."""
    
    if not files:
        console.print("[red]No files specified[/red]")
        sys.exit(1)
    
    # Load configuration from file if provided
    config_dict = {}
    if config:
        config_dict = load_config_file(config)
    
    # Override with command line options
    if language:
        config_dict['language'] = language
    if enable_separation:
        config_dict['enable_separation'] = True
    if enable_vad:
        config_dict['enable_vad'] = True
    if enable_diarization:
        config_dict['enable_diarization'] = True
    if num_speakers:
        config_dict['num_speakers'] = num_speakers
    
    transcribe_config = TranscribeConfig(**config_dict)
    
    try:
        async with AdvancedAudioTranscriptionCLI(ctx.obj['url'], ctx.obj['timeout']) as cli_client:
            results = await cli_client.batch_transcribe(
                list(files), transcribe_config, output_format, output_dir
            )
            
            # Display summary
            successful = sum(1 for r in results if r is not None)
            failed = len(results) - successful
            
            console.print(f"\n[green]Batch processing completed: {successful} successful, {failed} failed[/green]")
            
            if output_dir:
                console.print(f"[green]Output saved to: {output_dir}[/green]")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
async def health(ctx):
    """Check service health status."""
    try:
        async with AdvancedAudioTranscriptionCLI(ctx.obj['url'], ctx.obj['timeout']) as cli_client:
            health_data = await cli_client.get_health()
            display_health_status(health_data)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
async def capabilities(ctx):
    """Get service capabilities and configuration."""
    try:
        async with AdvancedAudioTranscriptionCLI(ctx.obj['url'], ctx.obj['timeout']) as cli_client:
            capabilities_data = await cli_client.get_capabilities()
            display_capabilities(capabilities_data)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
async def metrics(ctx):
    """Get service performance metrics."""
    try:
        async with AdvancedAudioTranscriptionCLI(ctx.obj['url'], ctx.obj['timeout']) as cli_client:
            metrics_data = await cli_client.get_metrics()
            console.print(Syntax(json.dumps(metrics_data, indent=2), "json"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output configuration file path')
@click.option('--language', '-l', help='Language code (e.g., en, es, fr)')
@click.option('--enable-separation', is_flag=True, help='Enable vocal separation')
@click.option('--enable-vad', is_flag=True, help='Enable voice activity detection')
@click.option('--enable-diarization', is_flag=True, help='Enable speaker diarization')
@click.option('--num-speakers', type=int, help='Expected number of speakers')
@click.option('--word-timestamps', is_flag=True, help='Include word-level timestamps')
def config(output, language, enable_separation, enable_vad, enable_diarization, 
          num_speakers, word_timestamps):
    """Generate configuration file with specified options."""
    
    config_dict = {}
    if language:
        config_dict['language'] = language
    if enable_separation:
        config_dict['enable_separation'] = True
    if enable_vad:
        config_dict['enable_vad'] = True
    if enable_diarization:
        config_dict['enable_diarization'] = True
    if num_speakers:
        config_dict['num_speakers'] = num_speakers
    if word_timestamps:
        config_dict['word_timestamps'] = True
    
    config = TranscribeConfig(**config_dict)
    
    if output:
        save_config_file(config, output)
        console.print(f"[green]Configuration saved to: {output}[/green]")
    else:
        console.print(Syntax(json.dumps(config.dict(), indent=2), "json"))


def main():
    """Main CLI entry point."""
    # Convert async commands to sync for click
    for command in [transcribe, batch, health, capabilities, metrics]:
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback
            command.callback = lambda *args, **kwargs: asyncio.run(original_callback(*args, **kwargs))
    
    cli()


if __name__ == "__main__":
    main()
