use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::QueueFlags;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(library, InstanceCreateInfo::default())
    .expect("failed to create instance");

    //Note this picks the first device, not the best atm
    let physical_device = instance
    .enumerate_physical_devices()
    .expect("could not enumerate devices")
    .next()
    .expect("no devices available");

    for family in physical_device.queue_family_properties() {
        println!("Found a queue family with {:?} queue(s)", family.queue_count);
    }


    let queue_family_index = physical_device
    .queue_family_properties()
    .iter()
    .enumerate()
    .position(|(_queue_family_index, queue_family_properties)| {
        queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
    })
    .expect("couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");

    //Queues are... essentially threads on the  GPU for compute tasks. 
    let queue = queues.next().unwrap();

    //Memory Buffers are used to hold information about to be transported to the GPU

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let data: i32 = 12;

    let source_content: Vec<i32> = (0..64).collect();
let source = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
        usage: BufferUsage::TRANSFER_SRC,
        ..Default::default()
    },
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_HOST
            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
    },
    source_content,
)
.expect("failed to create source buffer");

let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
let destination = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
        usage: BufferUsage::TRANSFER_DST,
        ..Default::default()
    },
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_HOST
            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        ..Default::default()
    },
    destination_content,
)
.expect("failed to create destination buffer");


//Command Buffers are used to hold commands for the GPU on how to compute a workload 
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};

let command_buffer_allocator = StandardCommandBufferAllocator::new(
    device.clone(),
    StandardCommandBufferAllocatorCreateInfo::default(),
);

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};

//Builder pattern. 
let mut builder = AutoCommandBufferBuilder::primary(
    &command_buffer_allocator,
    queue_family_index,
    CommandBufferUsage::OneTimeSubmit,
)
.unwrap();

builder
    .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
    .unwrap();
let command_buffer = builder.build().unwrap();

use vulkano::sync::{self, GpuFuture};

let future = sync::now(device.clone())
    .then_execute(queue.clone(), command_buffer)
    .unwrap()
    .then_signal_fence_and_flush() // same as signal fence, and then flush
    .unwrap();

future.wait(None).unwrap();

let src_content = source.read().unwrap();
let destination_content = destination.read().unwrap();
assert_eq!(&*src_content, &*destination_content);

println!("Everything succeeded!");


// Compute Shader Pipeline
let shader = cs::load(device.clone()).expect("failed to create shader module");

let data_iter = 0..65536u32;
let data_buffer = Buffer::from_iter(
    memory_allocator.clone(),
    BufferCreateInfo {
        usage: BufferUsage::STORAGE_BUFFER,
        ..Default::default()
    },
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
    },
    data_iter,
)
.expect("failed to create buffer");

use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo};

let cs = shader.entry_point("main").unwrap();
let stage = PipelineShaderStageCreateInfo::new(cs);
let layout = PipelineLayout::new(
    device.clone(),
    PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
        .into_pipeline_layout_create_info(device.clone())
        .unwrap(),
)
.unwrap();

let compute_pipeline = ComputePipeline::new(
    device.clone(),
    None,
    ComputePipelineCreateInfo::stage_layout(stage, layout),
)
.expect("failed to create compute pipeline");

}

use vulkano_shaders;

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}