#!/usr/bin/env /usr/bin/python3
"""
Edge Deployment Profiling for MiniCPM-AV
Measures inference latency, memory usage, and throughput on CPU.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer

from modeling_minicpm_av import MiniCPMAV, MiniCPMAVConfig


@dataclass
class ProfileResult:
    """Container for profiling results."""
    component: str
    latency_ms: float
    memory_mb: float
    throughput_samples_per_sec: float


class EdgeProfiler:
    """
    Profiler for edge deployment feasibility.
    
    Measures:
    - Model loading time
    - Inference latency (per component)
    - Memory usage (peak and steady-state)
    - End-to-end latency
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = []
        
    def measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # For CPU, use RSS from psutil if available
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except ImportError:
                return 0.0
    
    def profile_component(
        self,
        name: str,
        fn,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
        *args,
        **kwargs
    ) -> ProfileResult:
        """
        Profile a model component.
        
        Args:
            name: Component name
            fn: Function to profile
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
            *args, **kwargs: Arguments to fn
            
        Returns:
            ProfileResult with metrics
        """
        print(f"\nProfiling: {name}")
        print(f"  Warmup: {warmup_runs} runs")
        print(f"  Benchmark: {benchmark_runs} runs")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = fn(*args, **kwargs)
        
        # Benchmark
        latencies = []
        memory_before = self.measure_memory()
        
        for _ in range(benchmark_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = fn(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        memory_after = self.measure_memory()
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        memory_used = memory_after - memory_before
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        print(f"  Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Range: [{min_latency:.2f}, {max_latency:.2f}] ms")
        print(f"  Memory: {memory_used:.2f} MB")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        
        result = ProfileResult(
            component=name,
            latency_ms=avg_latency,
            memory_mb=memory_used,
            throughput_samples_per_sec=throughput
        )
        self.results.append(result)
        
        return result
    
    def profile_model_loading(self, config: MiniCPMAVConfig) -> Dict:
        """Profile model loading time and memory."""
        print("\n" + "="*60)
        print("Profiling Model Loading")
        print("="*60)
        
        memory_before = self.measure_memory()
        start_time = time.perf_counter()
        
        # Load model
        model = MiniCPMAV(config=config)
        
        end_time = time.perf_counter()
        memory_after = self.measure_memory()
        
        load_time = (end_time - start_time) * 1000  # ms
        memory_used = memory_after - memory_before
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Loading:")
        print(f"  Time: {load_time:.2f} ms ({load_time/1000:.2f} s)")
        print(f"  Memory: {memory_used:.2f} MB")
        print(f"  Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
        
        return {
            'load_time_ms': load_time,
            'memory_mb': memory_used,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    
    def profile_audio_encoder(
        self,
        model: MiniCPMAV,
        audio_duration_sec: float = 5.0,
        sampling_rate: int = 16000
    ) -> ProfileResult:
        """Profile audio encoder component."""
        print("\n" + "="*60)
        print("Profiling Audio Encoder (Moonshine-tiny)")
        print("="*60)
        
        # Create dummy audio
        num_samples = int(audio_duration_sec * sampling_rate)
        dummy_audio = torch.randn(num_samples).to(self.device)
        
        def encode_fn():
            model.audio_encoder.eval()
            with torch.no_grad():
                return model.audio_encoder(
                    dummy_audio,
                    sampling_rate=sampling_rate,
                    return_dict=True
                )
        
        return self.profile_component(
            "audio_encoder",
            encode_fn,
            warmup_runs=5,
            benchmark_runs=20
        )
    
    def profile_audio_projection(
        self,
        model: MiniCPMAV,
        num_tokens: int = 2  # Moonshine typically outputs 2 tokens
    ) -> ProfileResult:
        """Profile audio projection component."""
        print("\n" + "="*60)
        print("Profiling Audio Projection")
        print("="*60)
        
        # Create dummy audio features
        dummy_features = torch.randn(1, num_tokens, 288).to(self.device)
        
        def project_fn():
            model.audio_projector.eval()
            with torch.no_grad():
                return model.audio_projector(dummy_features)
        
        return self.profile_component(
            "audio_projection",
            project_fn,
            warmup_runs=10,
            benchmark_runs=50
        )
    
    def profile_token_compression(
        self,
        model: MiniCPMAV,
        input_tokens: int = 2,
        output_tokens: int = 50
    ) -> ProfileResult:
        """Profile token compression component."""
        print("\n" + "="*60)
        print("Profiling Token Compression")
        print("="*60)
        
        # Create dummy projected features
        dummy_features = torch.randn(1, input_tokens, 2304).to(self.device)
        
        def compress_fn():
            if model.audio_compressor is None:
                return dummy_features
            model.audio_compressor.eval()
            with torch.no_grad():
                return model.audio_compressor(dummy_features)
        
        return self.profile_component(
            "token_compression",
            compress_fn,
            warmup_runs=10,
            benchmark_runs=50
        )
    
    def profile_end_to_end(
        self,
        model: MiniCPMAV,
        tokenizer: AutoTokenizer,
        include_audio: bool = True,
        include_vision: bool = False  # Vision is expensive on CPU
    ) -> ProfileResult:
        """Profile end-to-end inference."""
        print("\n" + "="*60)
        print("Profiling End-to-End Inference")
        print("="*60)
        print(f"  Include audio: {include_audio}")
        print(f"  Include vision: {include_vision}")
        
        # Prepare inputs
        audio = torch.randn(16000 * 5).to(self.device) if include_audio else None
        
        # For vision, we'd need actual images - skip for CPU profiling
        images = None
        
        question = "What instrument is playing?"
        
        def inference_fn():
            model.eval()
            with torch.no_grad():
                # This is a simplified version
                # Full implementation would use model.generate_with_audio
                if audio is not None:
                    audio_embeds = model.encode_audio(audio)['audio_embeds']
                return audio_embeds if audio is not None else None
        
        return self.profile_component(
            "end_to_end",
            inference_fn,
            warmup_runs=3,
            benchmark_runs=10
        )
    
    def generate_report(self) -> Dict:
        """Generate comprehensive profiling report."""
        report = {
            'device': self.device,
            'cpu_count': os.cpu_count(),
            'results': [
                {
                    'component': r.component,
                    'latency_ms': r.latency_ms,
                    'memory_mb': r.memory_mb,
                    'throughput_samples_per_sec': r.throughput_samples_per_sec
                }
                for r in self.results
            ]
        }
        
        # Calculate totals
        total_latency = sum(r.latency_ms for r in self.results)
        total_memory = sum(r.memory_mb for r in self.results)
        
        report['summary'] = {
            'total_latency_ms': total_latency,
            'total_memory_mb': total_memory,
            'end_to_end_latency_ms': self.results[-1].latency_ms if self.results else 0
        }
        
        return report


def classify_deployment_feasibility(latency_ms: float) -> str:
    """Classify deployment feasibility based on latency."""
    if latency_ms < 100:
        return "EXCELLENT - Real-time capable"
    elif latency_ms < 500:
        return "GOOD - Interactive use acceptable"
    elif latency_ms < 2000:
        return "FAIR - Noticeable delay but usable"
    elif latency_ms < 5000:
        return "POOR - Significant delay"
    else:
        return "UNACCEPTABLE - Not suitable for edge"


def print_report(report: Dict):
    """Print formatted profiling report."""
    print("\n" + "="*60)
    print("EDGE DEPLOYMENT PROFILING REPORT")
    print("="*60)
    
    print(f"\nEnvironment:")
    print(f"  Device: {report['device']}")
    print(f"  CPU Count: {report['cpu_count']}")
    
    print(f"\nComponent Breakdown:")
    print(f"  {'Component':<25} {'Latency (ms)':<15} {'Throughput':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    
    for result in report['results']:
        print(f"  {result['component']:<25} {result['latency_ms']:<15.2f} {result['throughput_samples_per_sec']:<15.2f}")
    
    print(f"\nSummary:")
    summary = report['summary']
    print(f"  Total component latency: {summary['total_latency_ms']:.2f} ms")
    print(f"  End-to-end latency: {summary['end_to_end_latency_ms']:.2f} ms")
    
    feasibility = classify_deployment_feasibility(summary['end_to_end_latency_ms'])
    print(f"\n  Feasibility: {feasibility}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if summary['end_to_end_latency_ms'] > 5000:
        print("  ⚠️  Model is too large for real-time edge deployment")
        print("  • Consider quantization (INT8/INT4)")
        print("  • Use ONNX Runtime or TensorRT for optimization")
        print("  • Reduce audio token count further")
        print("  • Consider model distillation")
    elif summary['end_to_end_latency_ms'] > 1000:
        print("  ⚡ Model is usable but has noticeable latency")
        print("  • Consider batching for throughput")
        print("  • Use quantization for 2-4x speedup")
    else:
        print("  ✅ Model is suitable for edge deployment")


def main(args):
    """Main profiling function."""
    print("="*60)
    print("MiniCPM-AV Edge Deployment Profiling")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Initialize profiler
    profiler = EdgeProfiler(device=device)
    
    # Model configuration
    config = MiniCPMAVConfig(
        use_audio_compression=args.use_compression,
        num_audio_tokens=args.num_audio_tokens
    )
    
    # Profile model loading
    loading_metrics = profiler.profile_model_loading(config)
    
    # Initialize model for component profiling
    print("\n[Initializing model for component profiling...]")
    model = MiniCPMAV(config=config)
    model.to(device)
    model.eval()
    
    # Profile components
    if args.profile_audio:
        profiler.profile_audio_encoder(model, audio_duration_sec=5.0)
    
    if args.profile_projection:
        profiler.profile_audio_projection(model)
    
    if args.profile_compression:
        profiler.profile_token_compression(model)
    
    if args.profile_end_to_end:
        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V",
            trust_remote_code=True
        )
        profiler.profile_end_to_end(model, tokenizer)
    
    # Generate report
    report = profiler.generate_report()
    report['loading'] = loading_metrics
    
    # Print report
    print_report(report)
    
    # Save report
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Profile MiniCPM-AV for edge deployment')
    
    # Model arguments
    parser.add_argument('--use_compression', action='store_true', default=True,
                        help='Use audio token compression')
    parser.add_argument('--num_audio_tokens', type=int, default=50,
                        help='Number of audio tokens')
    
    # Profiling arguments
    parser.add_argument('--profile_audio', action='store_true', default=True,
                        help='Profile audio encoder')
    parser.add_argument('--profile_projection', action='store_true', default=True,
                        help='Profile audio projection')
    parser.add_argument('--profile_compression', action='store_true', default=True,
                        help='Profile token compression')
    parser.add_argument('--profile_end_to_end', action='store_true', default=True,
                        help='Profile end-to-end inference')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default='./results/edge_profile.json',
                        help='Output file for profiling results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
