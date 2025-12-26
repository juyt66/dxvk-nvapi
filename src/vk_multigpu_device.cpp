#include "vk_multigpu_device.h"
#include <algorithm>
#include <iostream>

namespace dxvk {

VkMultiGpuManager::VkMultiGpuManager() : vkInstance(VK_NULL_HANDLE) {}

VkMultiGpuManager::~VkMultiGpuManager() {
  std::lock_guard<std::mutex> lock(gpuMutex);
  for (auto& gpu : gpuDevices) {
    if (gpu.commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(gpu.logicalDevice, gpu.commandPool, nullptr);
    }
    if (gpu.logicalDevice != VK_NULL_HANDLE) {
      vkDestroyDevice(gpu.logicalDevice, nullptr);
    }
  }
  gpuDevices.clear();
}

bool VkMultiGpuManager::InitializeMultiGpu(VkInstance instance, uint32_t desiredGpuCount) {
    std::lock_guard<std::mutex> lock(gpuMutex);
  vkInstance = instance;

  uint32_t physicalDeviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);

  if (physicalDeviceCount == 0) {
    std::cerr << "No physical GPU devices found\n";
    return false;
  }

  std::vector<VkPhysicalDevice> devices(physicalDeviceCount);
  vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, devices.data());

  uint32_t gpusToUse = std::min(physicalDeviceCount, desiredGpuCount);
  gpuDevices.reserve(gpusToUse);

  for (uint32_t i = 0; i < gpusToUse; i++) {
    VkGpuDevice gpuInfo = {};
    gpuInfo.deviceId = i;
    gpuInfo.physicalDevice = devices[i];

    vkGetPhysicalDeviceProperties(devices[i], &gpuInfo.properties);
    vkGetPhysicalDeviceMemoryProperties(devices[i], &gpuInfo.memoryProperties);
    gpuInfo.deviceName = gpuInfo.properties.deviceName;

    if (!SelectQueueFamilyIndices(devices[i], gpuInfo)) {
      std::cerr << "Failed to find queue families for GPU " << i << "\n";
      continue;
    }

    if (!CreateLogicalDevice(devices[i], gpuInfo)) {
      std::cerr << "Failed to create logical device for GPU " << i << "\n";
      continue;
    }

    gpuDevices.push_back(gpuInfo);
  }

  return !gpuDevices.empty();
}

std::vector<VkGpuDevice*> VkMultiGpuManager::GetAvailableGpus() const {
    std::vector<VkGpuDevice*> result;
  for (auto& gpu : gpuDevices) {
    result.push_back(const_cast<VkGpuDevice*>(&gpu));
  }
  return result;
}

uint32_t VkMultiGpuManager::GetActiveGpuCount() const {
    return static_cast<uint32_t>(gpuDevices.size());
}

VkGpuDevice* VkMultiGpuManager::GetGpuByIndex(uint32_t index) {
    if (index < gpuDevices.size()) {
    return &gpuDevices[index];
    }
  return nullptr;
}

bool VkMultiGpuManager::SupportsAsyncComputeOnGpu(uint32_t gpuIndex) const {
    if (gpuIndex >= gpuDevices.size()) return false;
  return gpuDevices[gpuIndex].computeQueueFamilyIndex != UINT32_MAX;
}

bool VkMultiGpuManager::SupportsMemorySharing() const {
    return GetActiveGpuCount() >= 2;
}

VkDeviceMemory VkMultiGpuManager::AllocateMemory(uint32_t gpuIndex, const VkMemoryAllocateInfo& allocInfo) {
    if (gpuIndex >= gpuDevices.size()) return VK_NULL_HANDLE;

  VkDeviceMemory memory = VK_NULL_HANDLE;
  vkAllocateMemory(gpuDevices[gpuIndex].logicalDevice, &allocInfo, nullptr, &memory);
  return memory;
}

void VkMultiGpuManager::FreeMemory(uint32_t gpuIndex, VkDeviceMemory memory) {
    if (gpuIndex < gpuDevices.size() && memory != VK_NULL_HANDLE) {
    vkFreeMemory(gpuDevices[gpuIndex].logicalDevice, memory, nullptr);
    }
}

bool VkMultiGpuManager::SelectQueueFamilyIndices(VkPhysicalDevice device, VkGpuDevice& gpuInfo) {
    uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

  gpuInfo.graphicsQueueFamilyIndex = UINT32_MAX;
  gpuInfo.computeQueueFamilyIndex = UINT32_MAX;
  gpuInfo.transferQueueFamilyIndex = UINT32_MAX;

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      gpuInfo.graphicsQueueFamilyIndex = i;
    }
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT && gpuInfo.computeQueueFamilyIndex == UINT32_MAX) {
      gpuInfo.computeQueueFamilyIndex = i;
    }
    if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT && gpuInfo.transferQueueFamilyIndex == UINT32_MAX) {
      gpuInfo.transferQueueFamilyIndex = i;
    }
  }

  return gpuInfo.graphicsQueueFamilyIndex != UINT32_MAX;
}

bool VkMultiGpuManager::CreateLogicalDevice(VkPhysicalDevice physicalDevice, VkGpuDevice& gpuInfo) {
    float queuePriority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

  if (gpuInfo.graphicsQueueFamilyIndex != UINT32_MAX) {
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = gpuInfo.graphicsQueueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueInfo);
  }

  if (gpuInfo.computeQueueFamilyIndex != UINT32_MAX && 
          gpuInfo.computeQueueFamilyIndex != gpuInfo.graphicsQueueFamilyIndex) {
    VkDeviceQueueCreateInfo queueInfo = {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = gpuInfo.computeQueueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueInfo);
  }

  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
  deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();

  if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &gpuInfo.logicalDevice) != VK_SUCCESS) {
    return false;
  }

  vkGetDeviceQueue(gpuInfo.logicalDevice, gpuInfo.graphicsQueueFamilyIndex, 0, &gpuInfo.graphicsQueue);

  if (gpuInfo.computeQueueFamilyIndex != UINT32_MAX) {
    vkGetDeviceQueue(gpuInfo.logicalDevice, gpuInfo.computeQueueFamilyIndex, 0, &gpuInfo.computeQueue);
  }

  if (gpuInfo.transferQueueFamilyIndex != UINT32_MAX) {
    vkGetDeviceQueue(gpuInfo.logicalDevice, gpuInfo.transferQueueFamilyIndex, 0, &gpuInfo.transferQueue);
  }

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = gpuInfo.graphicsQueueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(gpuInfo.logicalDevice, &poolInfo, nullptr, &gpuInfo.commandPool) != VK_SUCCESS) {
    vkDestroyDevice(gpuInfo.logicalDevice, nullptr);
    return false;
  }

  return true;
}

} // namespace dxvk
