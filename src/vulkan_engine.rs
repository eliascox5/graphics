use std::collections::BTreeMap;
use std::future;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAlloc, StandardDescriptorSetAllocator};
use vulkano::descriptor_set::layout::{DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage, self};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::dpi::PhysicalSize;
use crate::mesh_parser::three_d;
use crate::{geometry_loader, make_cube, random_colours};
use crate::settings::Settings;
use crate::vulkan_engine::future::Future;
use vulkano::sync::fence::Fence;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

pub struct VulkanEngine{
    swapchain: Arc<Swapchain>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    pub device: Arc<Device>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    pipeline: Arc<GraphicsPipeline>,
    queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[three_d]>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    max_frames_in_flight: i32,
    current_frame: i32,
    previous_frame_finished_future: Option<Box<dyn GpuFuture>>,
    memory_allocator: Arc<vulkano::memory::allocator::GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>>,
    settings: Settings,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>
}
    
//notes:


impl VulkanEngine{
    //Create a new Engine
    pub fn new(window: &Arc<Window>, required_extensions: vulkano::instance::InstanceExtensions, settings: Settings) -> Self{  
        let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, device, mut queues) = create_devices(instance, &surface, device_extensions);

        //Grabs the first available queue. 
        let queue = queues.next().unwrap();

        println!("Creating Swapchain");
        let (swapchain, images) = {
            //The capabilities of the Surface.
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            //Sets the dimensions of the swapchain to the dimensions of the window
            let dimensions = window.inner_size();
            let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            //Currently just picks the first image format it finds.
            let image_format = physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        //A render pass represents what the renderer does for every image it renders.
        let render_pass: Arc<RenderPass> = create_render_pass(device.clone(), swapchain.clone());
        //The Frame buffers hold each image before they are sent to the screen
        let framebuffers: Vec<Arc<Framebuffer>> =create_framebuffers(&images, &render_pass.clone());

        //A viewport describes the reigon of the screen you're rendering too. 
        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
       
        println!("Loading object data");
        let memory_allocator: Arc<vulkano::memory::allocator::GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>> = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let geometry_loader = geometry_loader::GeometryLoader::new(device.clone());
        let vecs  = make_cube(); 
        let vertex_buffer = geometry_loader.create_vertex_buffer(vecs);


        println!("Loading Shaders");
        let vs = vs_spir::load(device.clone()).expect("failed to create shader module");
        let fs = fs_spir::load(device.clone()).expect("failed to create shader module");


        //Create Descriptor Sets
        let descriptor_set_allocator = Arc::new(StandardCommandBufferAllocator::new(device, Default::default(),));
        

        println!("Creating Pipeline");
         //Graphics pipeline
        let pipeline = create_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        println!("Creating Command Buffers");
        let command_buffer_allocator: StandardCommandBufferAllocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let mut command_buffers = create_command_buffers(
            &command_buffer_allocator, &queue, &pipeline, &framebuffers, &vertex_buffer.clone() );

        let max_frames_in_flight = images.len() as i32;
        let current_frame = 0;

        let frames_in_flight = images.len();
        let mut previous_frame_finished_future: Option<Box<dyn GpuFuture>> = Some(sync::now(device.clone()).boxed());
            
        println!("Finalising Engine Initialisation");
        Self {swapchain, command_buffers, images, render_pass, device, viewport, vs, fs, pipeline, queue, vertex_buffer, framebuffers, command_buffer_allocator, 
            max_frames_in_flight, current_frame, previous_frame_finished_future, memory_allocator, settings, descriptor_set_allocator}
        }
    
    //Recreate swapchain and stuff when required.
    pub  fn reload_objects_dependent_on_window_size(&mut self, new_dimensions: winit::dpi::PhysicalSize<u32>){
        let (new_swapchain, new_images) = self.swapchain
        .recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..self.swapchain.create_info()
        })
        .expect("failed to recreate swapchain: {e}");
        self.swapchain = new_swapchain;
        let new_framebuffers = create_framebuffers(&new_images, &self.render_pass);
        let new_pipeline = create_pipeline(
            self.device.clone(),
            self.vs.clone(),
            self. fs.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        );
        self.command_buffers = create_command_buffers(&self.command_buffer_allocator, &self.queue, &new_pipeline,&new_framebuffers, &self.vertex_buffer);
    }

    //Draw a frame
    pub fn draw<T: Vertex + BufferContents>(&mut self, vb: Subbuffer<[T]>)      {
        //Fetches the image (render target) to draw this frame on.

        //Debugging 
        if (self.settings.logging){
            println!("There are {} verticies", vb.len());
            println!("There are {} frame buffers", self.framebuffers.len());
            println!("The current frame is {}", self.current_frame);
        }
          
        let (image_i, suboptimal, acquire_future) =
        match swapchain::acquire_next_image(self.swapchain.clone(), None)
            .map_err(Validated::unwrap)
        {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                print!("Swapchain out of date oh no ");
                return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };
        
        //let new_buffer = self.create_command_buffer_for_vertex::<T>(vb.clone());
        //Syncronisation handling. 
        self.current_frame = (self.current_frame + 1) % self.framebuffers.len() as i32;
        self.create_and_flush_future(image_i, acquire_future);
        }
        
    //Registers a future on a specific image and then signals the fence once the image is ready
    //Basically executes the command_buffer once the image is available.
    fn create_and_flush_future(&mut self, image_i: u32, acquire_future: swapchain::SwapchainAcquireFuture) {
            let future =   sync::now(self.device.clone())
                .join(acquire_future)
                .then_execute(self.queue.clone(), self.command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();
    
            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    self.previous_frame_finished_future = Some(future.boxed());
                    }
                Err(VulkanError::OutOfDate) => {
                    //recreate_swapchain = true;
                    self.previous_frame_finished_future = Some(sync::now(self.device.clone()).boxed());
                }
                Err(e) => {
                    panic!("failed to flush future: {e}");
                    // previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
    }
    }
}

fn create_devices(instance: Arc<Instance>, surface: &Arc<Surface>, device_extensions: DeviceExtensions) -> (Arc<PhysicalDevice>, Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    let (physical_device, queue_family_index) =
        select_physical_device(&instance, surface, &device_extensions);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions, // new
            ..Default::default()
        },
    )
    .expect("failed to create device");
    (physical_device, device, queues)
}

//Selection Functions
pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no device available")
}

fn create_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(), // set the format the same as the swapchain
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}

//Create the framebuffers for an image. 
//Framebuffers are created once at the start of the creation process.
pub fn create_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

//Creates a pipeline, a description of the order and manner shaders and other processing will be applied.
fn create_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    //Shaders
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    //Info about how the vertex buffer maps onto the shader entry
    let vertex_input_state = three_d::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();


    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

//Creates a bog standard descriptor set layout
fn create__std_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout>{
 
    
}

fn create_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[three_d]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    //Reads "For each frame buffer( Remmeber equivelent to each image
    //Create a new command buffer.
    //The command buffer should start a render pass, bind the pipeline and vertex buffer
    //Then eend the render pass.
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        //CUSTOM GRAPHICS INSERT
                        clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
       
    }


mod vs_spir {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec4 color;

            layout(location = 0) out vec4 fragColor;

            void main() {
                gl_Position = vec4(position.x /2, position.y/2, position.z/2, 1.0);
                fragColor = color;
            }
        ",
    }
}
 
mod fs_spir {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec4 fragColor;
            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(fragColor);
            }
        ",
    }
}
