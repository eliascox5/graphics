use std::sync::Arc;
mod vulkan_engine;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use vulkano::swapchain::Surface;




fn main() {
    let event_loop = EventLoop::new();
    let frames_in_flight = 4;
    let required_extensions: vulkano::instance::InstanceExtensions = Surface::required_extensions(&event_loop);
   
    //Window handle from Winit
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let mut vk_instance = vulkan_engine::VulkanInstance::new(&window, required_extensions);


    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let mut previous_fence_i = 0;

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
            //if window_resized{
                //vk_instance.reload_objects_dependent_on_window_size(new_dimensions);
            //}

            vk_instance.draw();
        }
        _ => (),
    });
}