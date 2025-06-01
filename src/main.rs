use std::io::Read;
use std::sync::Arc;
mod vulkan_engine;
mod shader_loader;
mod geometry_loader;
mod settings;
mod mesh_parser;
use std::fs;
use std::fs::File;

use mesh_parser::{read_buffer_view, serialise_gltf, BufferAccessor};
use settings::Settings;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use vulkano::swapchain::Surface;




fn main() {
    let settings = Settings::default();


    let mut fl = File::open("Untitled(2).bin").unwrap();
    let mut buf=  Vec::new();
    fl.read_to_end(&mut buf);

    let ba = BufferAccessor{bit_size: 32, buffer: 0, byteLength: 288, byteOffset: 0, target: 34962};
    read_buffer_view(&buf, ba);

    println!("{}", buf.len());

    let event_loop = EventLoop::new();
    let frames_in_flight = 4;
    let required_extensions: vulkano::instance::InstanceExtensions = Surface::required_extensions(&event_loop);
   
    //Window handle from Winit
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let settings = Settings::default();

    let mut vk_instance = vulkan_engine::VulkanInstance::new(&window, required_extensions, settings);

    let geometry_loader = geometry_loader::GeometryLoader::new(vk_instance.device.clone());

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let mut previous_fence_i = 0;

    let vb = geometry_loader.create_vertex_buffer();
    

    use vulkano::sync::GpuFuture;
    use vulkano::sync::future::FenceSignalFuture;
    use vulkano::sync;
    
    //The Event looooop -----------------------------------------------------------------------------------

    let new_dimensions: winit::dpi::PhysicalSize<u32> = window.inner_size();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
            
        }
        Event::MainEventsCleared => {
            if window_resized{
                //vk_instance.reload_objects_dependent_on_window_size(new_dimensions);
            }
            vk_instance.draw(vb.clone());
        }
        _ => (),
    });
}

