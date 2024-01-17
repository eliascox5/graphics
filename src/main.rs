use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};

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

}


