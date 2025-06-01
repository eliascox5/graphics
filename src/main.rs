use std::io::Read;
use std::ops::Index;
use std::sync::Arc;
mod vulkan_engine;
mod shader_loader;
mod geometry_loader;
mod settings;
mod mesh_parser;
use std::fs;
use std::fs::File;

use mesh_parser::{read_buffer_view, BufferAccessor, three_d, Vec4};
use settings::Settings;
use vulkano::buffer::Subbuffer;
use vulkano::device::Device;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use vulkano::swapchain::Surface;

//assumes every point is going to be hit. 
//O(N)
fn sort_by_indexes<T: Clone>(list: Vec<T>, indexes: Vec<u16>) -> Vec<T>{
    let mut sorted: Vec<T> = vec![];
    for index in indexes{
        let item: T = list.get(index as usize).unwrap().clone();
        sorted.push(item);
    }
    return sorted;
}

pub fn random_colours(v: &Vec<three_d>, device: Arc<Device>) -> Subbuffer<[Vec4]>{
        let geometry_loader = geometry_loader::GeometryLoader::new(device.clone());
        let vecs: Vec<Vec4>  = v.iter()
        .map(|p| Vec4{position: [rand::random::<f32>() * 255.0,rand::random::<f32>() * 255.0,rand::random::<f32>()* 255.0,rand::random::<f32>() * 255.0]})
        .collect();
        geometry_loader.create_vertex_buffer(vecs)
}

pub fn make_cube() -> Vec<three_d>{
    let mut fl = File::open("Untitled(2).bin").unwrap();
    let mut buf=  Vec::new();
    fl.read_to_end(&mut buf);

    let ba = BufferAccessor{bit_size: 32, buffer: 0, byteLength: 288, byteOffset: 0, component_type: 34962};
    let data = read_buffer_view::<f32>(&buf, ba);
    let mut vectors: Vec<three_d> = vec![];

    let v3 = data.chunks(3);


//[rand::random::<f32>() * 255.0,rand::random::<f32>() * 255.0,rand::random::<f32>()* 255.0,rand::random::<f32>() * 255.0]
    for v in v3{
        let vs:[f32; 3]  = [v[0], v[1], v[2]];
        vectors.push(three_d { position: vs, color: [255.0, 255.0,255.0,255.0]});
    }
    

    let bu = BufferAccessor{bit_size: 16, buffer: 0, byteLength: 72, byteOffset: 1056, component_type: 34963};
    let indicies = read_buffer_view::<u16>(&buf, bu);

    sort_by_indexes(vectors, indicies)

}

fn main() {
    let settings = Settings::default();

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
     let vecs  = make_cube(); 
    let vb = geometry_loader.create_vertex_buffer(vecs);

    
    println!("Made everything!");
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

