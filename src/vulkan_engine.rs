use std::future;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
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
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit::window::Window;
use winit::dpi::PhysicalSize;
use crate::vulkan_engine::future::Future;
use vulkano::sync::fence::Fence;

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub struct VulkanInstance{
    swapchain: Arc<Swapchain>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    device: Arc<Device>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    pipeline: Arc<GraphicsPipeline>,
    queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    max_frames_in_flight: i32,
    current_frame: i32,
    previous_frame_finished_future: Option<Box<dyn GpuFuture>>
}
    

impl VulkanInstance{
    pub fn new(window: &Arc<Window>, required_extensions: vulkano::instance::InstanceExtensions) -> Self{  
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

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

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

    //Grabs the first available queue. 
    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
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
    let render_pass: Arc<RenderPass> = get_render_pass(device.clone(), swapchain.clone());
    //The Frame buffers hold each image before they are sent to the screen
    let framebuffers: Vec<Arc<Framebuffer>> =get_framebuffers(&images, &render_pass.clone());

    let memory_allocator: Arc<vulkano::memory::allocator::GenericMemoryAllocator<vulkano::memory::allocator::FreeListAllocator>> = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    //Some Sample data to make a lil triangle
    let vertex1 = MyVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = MyVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = MyVertex {
        position: [0.5, -0.25],
    };
    let vertex_buffer: Subbuffer<[MyVertex]> = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )
    .unwrap();

    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");
//A viewport describes the reigon of the screen you're rendering too. 
    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };
//Graphics pipeline
    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let command_buffer_allocator: StandardCommandBufferAllocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator, &queue, &pipeline, &framebuffers, &vertex_buffer );

    let max_frames_in_flight = images.len() as i32;
    let current_frame = 0;

    let frames_in_flight = images.len();
    let mut previous_frame_finished_future: Option<Box<dyn GpuFuture>> = Some(sync::now(device.clone()).boxed());
        

    Self {swapchain, command_buffers, images, render_pass, device, viewport, vs, fs, pipeline, queue, vertex_buffer, framebuffers, command_buffer_allocator, max_frames_in_flight, current_frame, previous_frame_finished_future}
    }
    
   pub  fn reload_objects_dependent_on_window_size(&mut self, new_dimensions: winit::dpi::PhysicalSize<u32>){
        let (new_swapchain, new_images) = self.swapchain
        .recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..self.swapchain.create_info()
        })
        .expect("failed to recreate swapchain: {e}");
    self.swapchain = new_swapchain;
    let new_framebuffers = get_framebuffers(&new_images, &self.render_pass);

        let new_pipeline = get_pipeline(
            self.device.clone(),
            self.vs.clone(),
           self. fs.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        );
        self.command_buffers = get_command_buffers(&self.command_buffer_allocator, &self.queue, &self.pipeline, &self.framebuffers, &self.vertex_buffer);
    }

   pub fn draw(&mut self)      {
    let (image_i, suboptimal, acquire_future) =
    match swapchain::acquire_next_image(self.swapchain.clone(), None)
        .map_err(Validated::unwrap)
    {
        Ok(r) => r,
        Err(VulkanError::OutOfDate) => {
            print!("Swapchain out of date oh no");
            return;
        }
        Err(e) => panic!("failed to acquire next image: {e}"),
    };

    let future = self.previous_frame_finished_future
                    .take()
                    .unwrap()
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



//Selection Functions :)
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

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
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

pub fn get_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
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

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
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

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[MyVertex]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
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


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}