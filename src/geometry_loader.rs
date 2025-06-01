use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer}, device::Device, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}};

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

    pub fn create_vertex_buffer(self) -> Subbuffer<[MyVertex]>{
        let vertex1 = MyVertex {
            position: [-0.5, -0.5],
        };
        let vertex2 = MyVertex {
            position: [0.5, 0.5],
        };
        let vertex3 = MyVertex {
            position: [0.5, -0.5],
        };
        
        let vertex4 = MyVertex {
            position: [-0.5, 0.5],
        };
        let vertex5 = MyVertex {
            position: [0.5, -0.5],
        };
        let vertex6 = MyVertex {
            position: [0.5, 0.5],
        };
        let vertex_buffer: Subbuffer<[MyVertex]> = Buffer::from_iter(
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
            vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6],
        )
        .unwrap();
        vertex_buffer
    }
}