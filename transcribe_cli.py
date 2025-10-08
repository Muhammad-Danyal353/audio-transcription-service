#!/usr/bin/env python3
"""
Simple CLI script for audio transcription.

This script provides a simple command-line interface for the audio transcription service.
It can be used as a standalone script or imported as a module.

Usage:
    python transcribe_cli.py --file audio.wav
    python transcribe_cli.py --file audio.wav --enable-separation
    python transcribe_cli.py --health
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.client import AudioTranscriptionCLI, create_config_from_args, display_transcription_result, display_health_status, display_capabilities


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Transcription Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python transcribe_cli.py --file audio.wav
  
  # Transcription with vocal separation
  python transcribe_cli.py --file audio.wav --enable-separation
  
  # Transcription with speaker diarization
  python transcribe_cli.py --file audio.wav --enable-diarization --num-speakers 2
  
  # Save as SRT subtitle file
  python transcribe_cli.py --file audio.wav --output-format srt --output-file subtitles.srt
  
  # Check service health
  python transcribe_cli.py --health
  
  # Get service capabilities
  python transcribe_cli.py --capabilities
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
                import json
                from rich.console import Console
                from rich.syntax import Syntax
                console = Console()
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
                    from rich.console import Console
                    console = Console()
                    console.print(f"\n[green]Output saved to: {args.output_file}[/green]")
    
    except KeyboardInterrupt:
        from rich.console import Console
        console = Console()
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        from rich.console import Console
        console = Console()
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
