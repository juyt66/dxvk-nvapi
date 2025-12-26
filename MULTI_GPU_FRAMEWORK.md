# Multi-GPU Framework with Vulkan Support

## Overview

This document describes a comprehensive multi-GPU rendering framework for DXVK-NVAPI with support for split-frame rendering across up to 3 NVIDIA GPUs using CUDA 12 and Vulkan.

## Architecture

### 1. GPU Device Management (`src/vk_multigpu_device.h/.cpp`)

```cpp
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace dxvk {

struct VkGpuDevice {
    uint32_t deviceId;
    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
    VkQueue graphicsQueue;
    VkQueue computeQueue;
    VkQueue transferQueue;
    VkCommandPool commandPool;
    std::string deviceName;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    uint32_t graphicsQueueFamilyIndex;
    uint32_t computeQueueFamilyIndex;
    uint32_t transferQueueFamilyIndex;
};

class VkMultiGpuManager {
public:
    VkMultiGpuManager();
    ~VkMultiGpuManager();

    // Initialization
    bool InitializeMultiGpu(VkInstance instance, uint32_t desiredGpuCount = 3);

    // GPU enumeration
    std::vector<VkGpuDevice*> GetAvailableGpus() const;
    uint32_t GetActiveGpuCount() const;
    VkGpuDevice* GetGpuByIndex(uint32_t index);

    // Device properties
    bool SupportsAsyncComputeOnGpu(uint32_t gpuIndex) const;
    bool SupportsMemorySharing() const;

    // Memory management
    VkDeviceMemory AllocateMemory(uint32_t gpuIndex, const VkMemoryAllocateInfo& allocInfo);
    void FreeMemory(uint32_t gpuIndex, VkDeviceMemory memory);

private:
    std::vector<VkGpuDevice> gpuDevices;
    VkInstance vkInstance;
    std::mutex gpuMutex;

    bool SelectQueueFamilyIndices(VkPhysicalDevice device, VkGpuDevice& gpuInfo);
    bool CreateLogicalDevice(VkPhysicalDevice physicalDevice, VkGpuDevice& gpuInfo);
};

} // namespace dxvk
```

### 2. Frame Distribution System (`src/vk_frame_distribution.h/.cpp`)

```cpp
#pragma once

#include "vk_multigpu_device.h"
#include <array>
#include <memory>

namespace dxvk {

enum class FrameDistributionMode {
    SPLIT_FRAME_HORIZONTAL,    // Divide horizontally
    SPLIT_FRAME_VERTICAL,      // Divide vertically
    SPLIT_FRAME_QUADRANTS,     // 4-way split
    AFR_ALTERNATING,           // Alternating frame rendering
    SFR_OPTIMIZED              // Optimized split frame
};

struct FrameRegion {
    uint32_t offsetX;
    uint32_t offsetY;
    uint32_t width;
    uint32_t height;
    uint32_t gpuIndex;
};

class VkFrameDistributor {
public:
    VkFrameDistributor(VkMultiGpuManager* gpuManager);
    ~VkFrameDistributor();

    // Configuration
    void SetDistributionMode(FrameDistributionMode mode);
    void SetFrameResolution(uint32_t width, uint32_t height);

    // Region computation
    std::vector<FrameRegion> ComputeFrameRegions() const;
    FrameRegion GetRegionForGpu(uint32_t gpuIndex) const;

    // Rendering distribution
    void DistributeCommandBuffers(const std::vector<VkCommandBuffer>& cmdBuffers);

    // Synchronization
    void SynchronizeFrameCompletion();
    void InsertGpuSynchronizationPoints();

private:
    VkMultiGpuManager* gpuManager;
    FrameDistributionMode distributionMode;
    uint32_t frameWidth, frameHeight;
    std::vector<FrameRegion> cachedRegions;

    std::vector<FrameRegion> ComputeHorizontalSplit() const;
    std::vector<FrameRegion> ComputeVerticalSplit() const;
    std::vector<FrameRegion> ComputeQuadrantSplit() const;
};

} // namespace dxvk
```

### 3. Synchronization Primitives (`src/vk_multigpu_sync.h/.cpp`)

```cpp
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <semaphore>

namespace dxvk {

class VkMultiGpuSynchronizer {
public:
    VkMultiGpuSynchronizer(VkMultiGpuManager* gpuManager);
    ~VkMultiGpuSynchronizer();

    // Semaphore management
    VkSemaphore CreateTimelineSemaphore(uint32_t gpuIndex, uint64_t initialValue = 0);
    void DestroySemaphore(uint32_t gpuIndex, VkSemaphore semaphore);

    // Frame synchronization
    void SignalFrameComplete(uint32_t gpuIndex, uint64_t frameId);
    void WaitForFrameComplete(uint32_t gpuIndex, uint64_t frameId);

    // Cross-GPU synchronization
    void InsertInterGpuBarrier(
        uint32_t srcGpu, uint32_t dstGpu,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage);

    // Event-based synchronization
    VkEvent CreateCrossGpuEvent(uint32_t gpuIndex);
    void SignalEvent(uint32_t gpuIndex, VkEvent event);
    void WaitForEvent(uint32_t gpuIndex, VkEvent event);

private:
    VkMultiGpuManager* gpuManager;
    std::vector<std::vector<VkSemaphore>> timelineSemaphores;
    std::vector<uint64_t> currentFrameIds;
};

} // namespace dxvk
```

### 4. Command Buffer Distribution (`src/vk_multigpu_commands.h/.cpp`)

```cpp
#pragma once

#include "vk_multigpu_device.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <queue>

namespace dxvk {

struct GpuCommandSubmission {
    uint32_t gpuIndex;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkSemaphore> signalSemaphores;
    VkFence fence;
};

class VkMultiGpuCommandDistributor {
public:
    VkMultiGpuCommandDistributor(VkMultiGpuManager* gpuManager);
    ~VkMultiGpuCommandDistributor();

    // Command buffer recording
    VkCommandBuffer BeginCommandBuffer(uint32_t gpuIndex);
    void EndCommandBuffer(VkCommandBuffer cmdBuffer);

    // Submission management
    void SubmitCommandBuffersToGpu(const GpuCommandSubmission& submission);
    void SubmitMultiGpuFrame(const std::vector<GpuCommandSubmission>& submissions);

    // Workload balancing
    void BalanceWorkloadAcrossGpus(const std::vector<uint32_t>& estimatedWorkloads);

    // Fence management
    bool WaitForGpuCompletion(uint32_t gpuIndex, uint64_t timeoutNs);

private:
    VkMultiGpuManager* gpuManager;
    std::vector<std::queue<VkCommandBuffer>> perGpuCommandQueues;
};

} // namespace dxvk
```

### 5. Texture and Memory Management (`src/vk_multigpu_memory.h/.cpp`)

```cpp
#pragma once

#include <vulkan/vulkan.h>
#include <map>
#include <memory>

namespace dxvk {

enum class MemoryPlacementStrategy {
    REPLICATED,        // Copy on all GPUs
    DISTRIBUTED,       // Split across GPUs
    PINNED_HOST,       // Host memory accessible to all
    GPU_LOCAL          // Local to specific GPU
};

class VkMultiGpuMemoryManager {
public:
    VkMultiGpuMemoryManager(VkMultiGpuManager* gpuManager);
    ~VkMultiGpuMemoryManager();

    // Buffer allocation
    VkBuffer AllocateBuffer(
        VkBufferUsageFlags usage,
        VkDeviceSize size,
        MemoryPlacementStrategy strategy,
        const std::vector<uint32_t>& targetGpus = {});

    // Image allocation
    VkImage AllocateImage(
        const VkImageCreateInfo& imageInfo,
        MemoryPlacementStrategy strategy,
        const std::vector<uint32_t>& targetGpus = {});

    // Deallocation
    void FreeBuffer(VkBuffer buffer);
    void FreeImage(VkImage image);

    // Memory transfer
    void CopyBufferBetweenGpus(uint32_t srcGpu, uint32_t dstGpu, VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void ReplicateBufferToAllGpus(VkBuffer buffer, VkDeviceSize size);

    // Persistent mapping
    void* MapMemory(VkBuffer buffer, VkDeviceSize size);
    void UnmapMemory(VkBuffer buffer);

private:
    VkMultiGpuManager* gpuManager;
    std::map<VkBuffer, MemoryPlacementStrategy> bufferStrategies;
    std::map<VkImage, MemoryPlacementStrategy> imageStrategies;
};

} // namespace dxvk
```

## Integration with DXVK-NVAPI

### Step 1: Add to `src/nvapi_gpu.cpp`

Add multi-GPU enumeration support:

```cpp
NvAPI_Status __cdecl NvAPI_GPU_GetPhysicalGpuInfoEx(
    NvU32* pCount,
    NvPhysicalGpuHandle* phPhysicalGpu,
    NV_GPU_INFO_EX* pInfo) {

    auto multiGpuManager = GetMultiGpuManager();
    auto gpus = multiGpuManager->GetAvailableGpus();

    *pCount = std::min((NvU32)gpus.size(), (NvU32)NVAPI_MAX_PHYSICAL_GPUS);

    for (uint32_t i = 0; i < *pCount; i++) {
        phPhysicalGpu[i] = reinterpret_cast<NvPhysicalGpuHandle>(gpus[i]);

        // Fill GPU info
        pInfo[i].gpuId = gpus[i]->deviceId;
        // ... populate other fields
    }

    return OK(n);
}
```

### Step 2: Add to `src/meson.build`

```meson
nvapi_sources += files(
  'vk_multigpu_device.cpp',
  'vk_frame_distribution.cpp',
  'vk_multigpu_sync.cpp',
  'vk_multigpu_commands.cpp',
  'vk_multigpu_memory.cpp',
)

nvapi_deps += dependency('vulkan')
```

## Usage Example

```cpp
// Initialize multi-GPU
VkMultiGpuManager gpuManager;
gpuManager.InitializeMultiGpu(vkInstance, 3);  // 3 GPUs

// Setup frame distribution
VkFrameDistributor distributor(&gpuManager);
distributor.SetDistributionMode(FrameDistributionMode::SPLIT_FRAME_HORIZONTAL);
distributor.SetFrameResolution(3840, 2160);  // 4K resolution

// Get frame regions for rendering
auto regions = distributor.ComputeFrameRegions();

// For each GPU, render its region
for (const auto& region : regions) {
    auto gpu = gpuManager.GetGpuByIndex(region.gpuIndex);
    // Record commands for this region...
}

// Synchronize completion
distributor.SynchronizeFrameCompletion();
```

## Performance Considerations

1. **Split-Frame Rendering (SFR)**: Divides frame into horizontal, vertical, or quadrant regions
2. 2. **Alternate Frame Rendering (AFR)**: GPUs render alternate frames
   3. 3. **PCIe Bandwidth**: Monitor inter-GPU communication overhead
      4. 4. **Memory Replication**: Replicate frequently-used data across GPUs
        
         5. ## Configuration Environment Variables
        
         6. ```bash
            # Enable multi-GPU
            export DXVK_MULTIGPU_MODE=1

            # Set distribution mode (horizontal, vertical, quadrants, afr)
            export DXVK_MULTIGPU_DIST_MODE=horizontal

            # Number of GPUs
            export DXVK_MULTIGPU_COUNT=3

            # Enable GPU workload balancing
            export DXVK_MULTIGPU_BALANCE=1
            ```

            ## Build Instructions

            ```bash
            git clone https://github.com/juyt66/dxvk-nvapi.git
            cd dxvk-nvapi
            # Add multi-GPU support files
            meson setup build
            cd build
            ninja
            ninja install
            ```

            ## Future Enhancements

            1. Dynamic workload balancing
            2. 2. Machine learning-based frame region optimization
               3. 3. Support for HDR and higher precision formats
                  4. 4. Ray-tracing distribution across GPUs
                     5. 5. PCIe P2P optimization for Nvlink/Unified Memory
                       
                        6. ---
                        7. *DXVK-NVAPI Multi-GPU Framework v1.0 - CUDA 12 Vulkan Support*
