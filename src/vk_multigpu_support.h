#pragma once

#include "vk_multigpu_device.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <memory>
#include <semaphore>
#include <map>

namespace dxvk {

// Frame Distribution Structures
enum class FrameDistributionMode {
  SPLIT_FRAME_HORIZONTAL,
  SPLIT_FRAME_VERTICAL,
  SPLIT_FRAME_QUADRANTS,
  AFR_ALTERNATING,
  SFR_OPTIMIZED
};

struct FrameRegion {
  uint32_t offsetX, offsetY, width, height;
  uint32_t gpuIndex;
};

// Frame Distribution Manager
class VkFrameDistributor {
public:
  VkFrameDistributor(VkMultiGpuManager* gpuManager);
  ~VkFrameDistributor();
  void SetDistributionMode(FrameDistributionMode mode);
  void SetFrameResolution(uint32_t width, uint32_t height);
  std::vector<FrameRegion> ComputeFrameRegions() const;
  FrameRegion GetRegionForGpu(uint32_t gpuIndex) const;
  void DistributeCommandBuffers(const std::vector<VkCommandBuffer>& cmdBuffers);
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

// Synchronization Primitives
class VkMultiGpuSynchronizer {
public:
  VkMultiGpuSynchronizer(VkMultiGpuManager* gpuManager);
  ~VkMultiGpuSynchronizer();
  VkSemaphore CreateTimelineSemaphore(uint32_t gpuIndex, uint64_t initialValue = 0);
  void DestroySemaphore(uint32_t gpuIndex, VkSemaphore semaphore);
  void SignalFrameComplete(uint32_t gpuIndex, uint64_t frameId);
  void WaitForFrameComplete(uint32_t gpuIndex, uint64_t frameId);
  void InsertInterGpuBarrier(uint32_t srcGpu, uint32_t dstGpu,
                            VkPipelineStageFlags srcStage,
                            VkPipelineStageFlags dstStage);
  VkEvent CreateCrossGpuEvent(uint32_t gpuIndex);
  void SignalEvent(uint32_t gpuIndex, VkEvent event);
  void WaitForEvent(uint32_t gpuIndex, VkEvent event);
private:
  VkMultiGpuManager* gpuManager;
  std::vector<std::vector<VkSemaphore>> timelineSemaphores;
  std::vector<uint64_t> currentFrameIds;
};

// Memory Management
enum class MemoryPlacementStrategy {
  REPLICATED, DISTRIBUTED, PINNED_HOST, GPU_LOCAL
};

class VkMultiGpuMemoryManager {
public:
  VkMultiGpuMemoryManager(VkMultiGpuManager* gpuManager);
  ~VkMultiGpuMemoryManager();
  VkBuffer AllocateBuffer(VkBufferUsageFlags usage, VkDeviceSize size,
                         MemoryPlacementStrategy strategy,
                         const std::vector<uint32_t>& targetGpus = {});
  VkImage AllocateImage(const VkImageCreateInfo& imageInfo,
                       MemoryPlacementStrategy strategy,
                       const std::vector<uint32_t>& targetGpus = {});
  void FreeBuffer(VkBuffer buffer);
  void FreeImage(VkImage image);
  void CopyBufferBetweenGpus(uint32_t srcGpu, uint32_t dstGpu,
                            VkBuffer src, VkBuffer dst, VkDeviceSize size);
  void ReplicateBufferToAllGpus(VkBuffer buffer, VkDeviceSize size);
  void* MapMemory(VkBuffer buffer, VkDeviceSize size);
  void UnmapMemory(VkBuffer buffer);
private:
  VkMultiGpuManager* gpuManager;
  std::map<VkBuffer, MemoryPlacementStrategy> bufferStrategies;
  std::map<VkImage, MemoryPlacementStrategy> imageStrategies;
};

// Command Distribution
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
  VkCommandBuffer BeginCommandBuffer(uint32_t gpuIndex);
  void EndCommandBuffer(VkCommandBuffer cmdBuffer);
  void SubmitCommandBuffersToGpu(const GpuCommandSubmission& submission);
  void SubmitMultiGpuFrame(const std::vector<GpuCommandSubmission>& submissions);
  void BalanceWorkloadAcrossGpus(const std::vector<uint32_t>& estimatedWorkloads);
  bool WaitForGpuCompletion(uint32_t gpuIndex, uint64_t timeoutNs);
private:
  VkMultiGpuManager* gpuManager;
  std::vector<std::queue<VkCommandBuffer>> perGpuCommandQueues;
};

} // namespace dxvk
