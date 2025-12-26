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
