# 2D Gaussian Splatting Renderer

A high-performance Rust implementation of 2D Gaussian Splatting for image representation and reconstruction using both CPU and GPU acceleration.

## Overview

This project implements **Image-GS**, a content-adaptive image representation method that uses anisotropic 2D Gaussians for efficient, high-fidelity image compression and rendering. The system can fit a collection of 2D Gaussians to approximate any input image through an iterative training process.

### Features

- **Dual Rendering Modes**: CPU and GPU implementations for maximum compatibility
- **Content-Adaptive Initialization**: Smart Gaussian placement based on image content and edge detection
- **Adaptive Training**: Dynamic learning rates and densification strategies
- **Quality Metrics**: Comprehensive evaluation including PSNR, SSIM, and LPIPS
- **Multi-batch GPU Processing**: Handles large Gaussian collections efficiently
- **Robust Error Handling**: Automatic GPU device recovery and fallback mechanisms

## Technical Architecture

```mermaid
%%{init: {'theme':'dark', 'themeVariables': { 'primaryColor': '#0f172a', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#3b82f6', 'lineColor': '#60a5fa', 'secondaryColor': '#1e293b', 'tertiaryColor': '#334155', 'background': '#0f172a', 'mainBkg': '#0f172a', 'secondBkg': '#1e293b', 'tertiaryBkg': '#334155'}}}%%

flowchart TD
    %% Input/Output
    INPUT["🖼️ Input Image<br/>(JPEG/PNG)"] 
    OUTPUT["✨ Rendered Output<br/>(PNG)"]
    METRICS["📊 Quality Metrics<br/>(PSNR, SSIM, LPIPS)"]
    
    %% CLI Entry Point
    CLI["🚀 main.rs<br/>CLI Interface"] 
    
    %% Execution Modes
    CPU_MODE["💻 CPU Mode<br/>--cpu / -1"]
    GPU_MODE["🎮 GPU Mode<br/>--gpu / -2"] 
    COMPARE_MODE["⚖️ Compare Mode<br/>--compare / -3"]
    
    %% Core Modules
    GAUSSIAN["⚪ gaussian.rs<br/>Gaussian2D Struct"]
    IMAGIGS["🎯 ImageGS Module<br/>Main Controller"]
    ACCURACY["📈 accuracy_metrics.rs<br/>Quality Evaluation"]
    
    %% ImageGS Submodules
    INIT["🎲 initialization.rs<br/>Smart Gaussian Placement"]
    GPU_RENDER["⚡ gpu_render.rs<br/>WebGPU Compute Shaders"]
    
    %% Implementation Layer
    IMPL_BOX["📦 Implementation Layer"]
    ADAPTIVE["🔄 adaptive.rs<br/>Dynamic Training"]
    CPU_IMPL["🖥️ cpu_impl.rs<br/>Rayon Parallel Processing"]
    GPU_IMPL["🚀 gpu_impl.rs<br/>GPU Training Logic"]
    COMMON["🛠️ common.rs<br/>Shared Utilities"]
    
    %% GPU Pipeline Components
    WEBGPU["🎮 WebGPU Backend<br/>(Vulkan/DX12/GL)"]
    COMPUTE["⚙️ Compute Shaders<br/>Parallel Gaussian Eval"]
    BATCHING["📦 Multi-batch Processing<br/>Memory Management"]
    
    %% Training Process
    TRAINING["🎯 Training Loop<br/>Iterative Optimization"]
    DENSIFY["➕ Densification<br/>Add Gaussians"]
    PRUNE["✂️ Pruning<br/>Remove Ineffective"]
    
    %% Data Flow
    INPUT --> CLI
    CLI --> CPU_MODE
    CLI --> GPU_MODE
    CLI --> COMPARE_MODE
    
    CPU_MODE --> IMAGIGS
    GPU_MODE --> IMAGIGS
    COMPARE_MODE --> IMAGIGS
    
    IMAGIGS --> GAUSSIAN
    IMAGIGS --> INIT
    IMAGIGS --> IMPL_BOX
    
    IMPL_BOX --> ADAPTIVE
    IMPL_BOX --> CPU_IMPL  
    IMPL_BOX --> GPU_IMPL
    IMPL_BOX --> COMMON
    
    GPU_IMPL --> GPU_RENDER
    GPU_RENDER --> WEBGPU
    WEBGPU --> COMPUTE
    COMPUTE --> BATCHING
    
    INIT --> TRAINING
    ADAPTIVE --> TRAINING
    TRAINING --> DENSIFY
    TRAINING --> PRUNE
    
    TRAINING --> OUTPUT
    OUTPUT --> ACCURACY
    ACCURACY --> METRICS
    
    %% Styling with Black & Blue Theme
    classDef inputOutput fill:#93c5fd,stroke:#ffffff,stroke-width:2px,color:#0f172a
    classDef cliMode fill:#1e3a8a,stroke:#ffffff,stroke-width:2px,color:#ffffff
    classDef coreModule fill:#2563eb,stroke:#ffffff,stroke-width:2px,color:#ffffff
    classDef implementation fill:#3b82f6,stroke:#ffffff,stroke-width:2px,color:#ffffff
    classDef gpuPipeline fill:#60a5fa,stroke:#ffffff,stroke-width:2px,color:#0f172a
    classDef training fill:#1d4ed8,stroke:#ffffff,stroke-width:2px,color:#ffffff
    
    class INPUT,OUTPUT,METRICS inputOutput
    class CLI,CPU_MODE,GPU_MODE,COMPARE_MODE cliMode
    class GAUSSIAN,IMAGIGS,ACCURACY,INIT coreModule
    class IMPL_BOX,ADAPTIVE,CPU_IMPL,GPU_IMPL,COMMON implementation
    class GPU_RENDER,WEBGPU,COMPUTE,BATCHING gpuPipeline
    class TRAINING,DENSIFY,PRUNE training
```

### Core Components

- **Gaussian2D**: 2D Gaussian **primitives** with position, scale, rotation, and color
- **ImageGS**: Main **controller** for Gaussian collections and training
- **GpuRenderer**: **WebGPU** compute shader implementation for accelerated rendering 
- **Accuracy Metrics**: **PSNR**, **SSIM**, and **LPIPS** quality evaluation

### Key Algorithms

**Initialization**: **Three-tier** approach with 60% **grid-based** placement, 25% **edge detection**, and 15% **content-aware** random sampling.

**Training**: **Adaptive** learning rates, smart **densification** in high-error regions, **pruning** of ineffective Gaussians, and GPU render **caching** .

## Installation

### Prerequisites

- **Rust**: Latest stable version
- **GPU Support**: Intel Arc or compatible WebGPU device (for GPU mode)

### Build

```bash
cargo build --release
```

## Usage

Basic execution modes :

```bash
./target/release/gauss-render -1    # CPU mode
./target/release/gauss-render -2    # GPU mode (Intel Arc recommended)
./target/release/gauss-render -3    # Comparison mode (both CPU and GPU)
```

Custom parameters :
```bash
./target/release/gauss-render -2 --image path/to/image.jpg --iterations 500 --width 800 --height 600
```

Run `./target/release/gauss-render --help` for full usage information.

### Output Files

The system generates several outputs:
- `cpu_final_output.png` / `gpu_final_output.png`: Final rendered results
- `iterations_gpu/`: Training progress snapshots (GPU mode)
- Quality metrics printed to console

## Performance Characteristics

### GPU Acceleration

The GPU implementation  provides significant performance improvements:
- **Compute Shaders**: Parallel Gaussian evaluation using WebGPU
- **Batch Processing**: Handles large Gaussian collections through multi-batch rendering
- **Smart Caching**: Reduces redundant computations during training
- **Device Recovery**: Automatic handling of GPU device loss

### Optimization Features

- **Parallel CPU Rendering**: Rayon-based parallelization for CPU mode
- **Adaptive Rendering Frequency**: Less frequent rendering in later training phases
- **Memory Management**: Conservative buffer limits to prevent device issues
- **Early Stopping**: Automatic termination when convergence is achieved

## Project Structure

```
src/
├── main.rs                    # CLI interface and mode selection
├── lib.rs                     # Public API exports
├── gaussian.rs                # Core Gaussian2D implementation
├── accuracy_metrics.rs        # Quality evaluation metrics
└── image_gs/
    ├── mod.rs                 # ImageGS struct definition
    ├── initialization.rs      # Gaussian initialization strategies
    ├── gpu_render.rs          # WebGPU compute implementation
    └── implementation/
        ├── adaptive.rs        # Adaptive training algorithms
        ├── cpu_impl.rs        # CPU-specific implementations
        ├── gpu_impl.rs        # GPU training logic
        └── common.rs          # Shared utilities and rendering
```

## Research Background

This is my Rust implementation of 2D Gaussian Splatting based on the research paper **"Image-GS: Content-Adaptive Image Representation via 2D Gaussians"** [Research Paper](https://arxiv.org/pdf/2407.01866). The method extends traditional 3D Gaussian Splatting techniques to the 2D domain, providing an alternative to neural approaches like NeRF with explicit control over the representation and efficient rendering capabilities.
