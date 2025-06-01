use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer}, device::Device, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::vulkan_engine::MyVertex;

//Loads geometry and produces the vertex buffer. 
pub struct GeometryLoader{
    memory_allocator: Arc<vulkano::memory::allocator::GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>
}
impl GeometryLoader{
    pub fn new(device: Arc<Device>) -> Self{
        let ma = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        GeometryLoader{memory_allocator: ma}
    }


    pub fn create_vertex_buffer<T: Vertex + BufferContents>(self, vertexs: Vec<T>) -> Subbuffer<[T]>{
        
        let vertex_buffer: Subbuffer<[T]> = Buffer::from_iter(
            self.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertexs,
        )
        .unwrap();
        vertex_buffer
    }
}