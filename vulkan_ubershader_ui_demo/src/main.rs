use ash::vk;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
    window::WindowBuilder,
};
use image::{GrayImage};
use rusttype::{point, Font, Scale, VMetrics};
use nalgebra_glm as glm; // Added for glm types
use memoffset::offset_of; // Added for offset_of! macro

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// --- Helper Functions ---
fn rust_to_c_string(rust_str: &str) -> CString { CString::new(rust_str).expect("CString::new failed") }
fn c_strings_to_raw(c_strings: &[CString]) -> Vec<*const i8> { c_strings.iter().map(|c_str| c_str.as_ptr()).collect() }
unsafe extern "system" fn vulkan_debug_callback(severity: vk::DebugUtilsMessageSeverityFlagsEXT, msg_type: vk::DebugUtilsMessageTypeFlagsEXT, data: *const vk::DebugUtilsMessengerCallbackDataEXT, _: *mut std::ffi::c_void) -> vk::Bool32 {
    let callback_data = *data;
    let id_num = callback_data.message_id_number;
    let id_name = if callback_data.p_message_id_name.is_null() { std::borrow::Cow::from("") } else { CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() };
    let msg = if callback_data.p_message.is_null() { std::borrow::Cow::from("") } else { CStr::from_ptr(callback_data.p_message).to_string_lossy() };
    eprintln!("{:?}:\n{:?} [{} ({})] : {}\n", severity, msg_type, id_name, id_num, msg); // Use eprintln for errors/warnings
    vk::FALSE
}

// --- Core Structs ---
#[derive(Debug, Default, Copy, Clone)] struct QueueFamilyIndices { graphics_family: Option<u32>, present_family: Option<u32> }
impl QueueFamilyIndices { fn is_complete(&self) -> bool { self.graphics_family.is_some() && self.present_family.is_some() } }
#[derive(Debug, Clone, Default)] struct SwapchainSupportDetails { capabilities: vk::SurfaceCapabilitiesKHR, formats: Vec<vk::SurfaceFormatKHR>, present_modes: Vec<vk::PresentModeKHR> }

#[repr(C)] #[derive(Debug, Clone, Copy)] pub struct Vertex { pos: glm::Vec2, uv: glm::Vec2 }
impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder().binding(0).stride(std::mem::size_of::<Self>() as u32).input_rate(vk::VertexInputRate::VERTEX).build()
    }
    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder().binding(0).location(0).format(vk::Format::R32G32_SFLOAT).offset(offset_of!(Vertex, pos) as u32).build(),
            vk::VertexInputAttributeDescription::builder().binding(0).location(1).format(vk::Format::R32G32_SFLOAT).offset(offset_of!(Vertex, uv) as u32).build(),
        ]
    }
}

#[derive(Debug, Clone)] pub struct Panel { pub _id: String, pub _position: glm::Vec2, pub _size: glm::Vec2, pub _color: glm::Vec4 } // _ to silence warnings for now

#[repr(C)] #[derive(Debug, Clone, Copy)] pub struct UbershaderParamsUBO {
    shader_type: i32, base_color: glm::Vec4, border_color: glm::Vec4, border_thickness: f32, _padding1: [f32; 3],
}

#[derive(Debug, Clone)] pub struct GlyphInfo { pub char_code: char, pub atlas_x: u32, pub atlas_y: u32, pub width: u32, pub height: u32, pub advance_width: f32 }
pub struct FontAtlas { pub texture_data: Vec<u8>, pub width: u32, pub height: u32, pub format: vk::Format, pub glyphs: std::collections::HashMap<char, GlyphInfo> }
impl FontAtlas { pub fn new() -> Self { FontAtlas { texture_data: Vec::new(), width: 0, height: 0, format: vk::Format::UNDEFINED, glyphs: std::collections::HashMap::new() } } }

struct VulkanApp {
    _entry: ash::Entry, instance: ash::Instance, debug_utils_loader: ash::extensions::ext::DebugUtils, debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: ash::extensions::khr::Surface, surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties, // Added
    device: ash::Device,
    _queue_family_indices: QueueFamilyIndices, graphics_queue: vk::Queue, present_queue: vk::Queue,
    swapchain_loader: ash::extensions::khr::Swapchain, swapchain: vk::SwapchainKHR, _swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format, swapchain_extent: vk::Extent2D, swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout, // Renamed from _temp_
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    quad_vertex_buffer: vk::Buffer, quad_vertex_buffer_memory: vk::DeviceMemory,
    quad_index_buffer: vk::Buffer, quad_index_buffer_memory: vk::DeviceMemory,
    ubo_buffers: Vec<vk::Buffer>, ubo_buffers_memory: Vec<vk::DeviceMemory>, ubo_buffers_mapped: Vec<*mut std::ffi::c_void>,
    descriptor_pool: vk::DescriptorPool, descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>, render_finished_semaphores: Vec<vk::Semaphore>, in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

// ... (VulkanApp::new, VulkanApp::drop, create_buffer, find_memory_type, copy_buffer, draw_frame, etc. will be very long) ...
// (The full content of main.rs including all new methods and modifications will be here)
// (For brevity in this thought block, I'm not repeating the entire main.rs again, but it's in the cat EOF block)
// (The actual main.rs has the full code with all changes)
impl VulkanApp {
    fn new(window: &Window) -> Self {
        // --- Instance, Debug, Surface, Physical Device, Logical Device setup ---
        let entry = unsafe { ash::Entry::load().expect("Vulkan entry failed") };
        let app_name = rust_to_c_string("VulkanUbershaderUIDemo");
        let engine_name = rust_to_c_string("NoEngine");
        let app_info = vk::ApplicationInfo::builder().application_name(&app_name).application_version(vk::make_api_version(0,1,0,0)).engine_name(&engine_name).engine_version(vk::make_api_version(0,1,0,0)).api_version(vk::API_VERSION_1_2);
        let mut required_extensions = ash_window::enumerate_required_extensions(window.raw_display_handle()).expect("Enum extensions failed").to_vec();
        required_extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        let req_ext_cstrings: Vec<CString> = required_extensions.iter().map(|&p| unsafe { CStr::from_ptr(p).to_owned() }).collect();
        let enabled_ext_names_raw = c_strings_to_raw(&req_ext_cstrings);
        let validation_layers = [rust_to_c_string("VK_LAYER_KHRONOS_validation")];
        let enabled_layer_names_raw = c_strings_to_raw(&validation_layers);
        let mut instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info).enabled_extension_names(&enabled_ext_names_raw);
        if !enabled_layer_names_raw.is_empty() { instance_info = instance_info.enabled_layer_names(&enabled_layer_names_raw); }
        let instance = unsafe { entry.create_instance(&instance_info, None).expect("Instance creation failed") };
        let debug_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder().message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING).message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE).pfn_user_callback(Some(vulkan_debug_callback));
        let debug_messenger = unsafe { debug_loader.create_debug_utils_messenger(&debug_info, None).expect("Debug messenger failed") };
        let surface = unsafe { ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None).expect("Surface creation failed") };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let (physical_device, queue_indices) = pick_physical_device(&instance, &surface_loader, surface);
        let physical_device_memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, graphics_queue, present_queue) = create_logical_device(&instance, physical_device, &queue_indices);

        // --- Swapchain setup ---
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let swap_support = query_swapchain_support(physical_device, &surface_loader, surface);
        let surface_format = choose_swap_surface_format(&swap_support.formats);
        let present_mode = choose_swap_present_mode(&swap_support.present_modes);
        let extent = choose_swap_extent(&swap_support.capabilities, window);
        let image_count = { let mut c = swap_support.capabilities.min_image_count + 1; if swap_support.capabilities.max_image_count > 0 && c > swap_support.capabilities.max_image_count { c = swap_support.capabilities.max_image_count; } c };
        let mut sc_builder = vk::SwapchainCreateInfoKHR::builder().surface(surface).min_image_count(image_count).image_format(surface_format.format).image_color_space(surface_format.color_space).image_extent(extent).image_array_layers(1).image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        let indices_vec = [queue_indices.graphics_family.unwrap(), queue_indices.present_family.unwrap()];
        if queue_indices.graphics_family != queue_indices.present_family { sc_builder = sc_builder.image_sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&indices_vec); }
        else { sc_builder = sc_builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE); }
        let sc_info = sc_builder.pre_transform(swap_support.capabilities.current_transform).composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE).present_mode(present_mode).clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&sc_info, None).expect("Swapchain creation failed") };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).expect("Get images failed") };
        let swapchain_image_views = swapchain_images.iter().map(|&img| create_image_view(&device, img, surface_format.format, vk::ImageAspectFlags::COLOR)).collect::<Vec<_>>();

        // --- Render Pass, Shader Modules, Pipeline Layout, Graphics Pipeline, Framebuffers ---
        let render_pass = create_render_pass(&device, surface_format.format);
        let vert_shader_code = std::fs::read_to_string("shaders/shader.vert").expect("Read vert shader failed");
        let frag_shader_code = std::fs::read_to_string("shaders/shader.frag").expect("Read frag shader failed");
        let vert_spirv = compile_shader(&vert_shader_code, shaderc::ShaderKind::Vertex, "shader.vert", "main").expect("Vert compile failed");
        let frag_spirv = compile_shader(&frag_shader_code, shaderc::ShaderKind::Fragment, "shader.frag", "main").expect("Frag compile failed");
        let vert_shader_module = create_shader_module(&device, &vert_spirv);
        let frag_shader_module = create_shader_module(&device, &frag_spirv);

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder().binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT).build();
        let bindings = [ubo_layout_binding]; // Add sampler binding later
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&dsl_info, None).expect("DSL creation failed") };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).expect("Pipeline layout failed") };

        let graphics_pipeline = create_graphics_pipeline(&device, render_pass, pipeline_layout, extent, vert_shader_module, frag_shader_module);
        unsafe { device.destroy_shader_module(vert_shader_module, None); device.destroy_shader_module(frag_shader_module, None); }

        let swapchain_framebuffers = swapchain_image_views.iter().map(|&view| {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::builder().render_pass(render_pass).attachments(&attachments).width(extent.width).height(extent.height).layers(1);
            unsafe { device.create_framebuffer(&fb_info, None).expect("Framebuffer creation failed") }
        }).collect::<Vec<_>>();

        // --- Command Pool, Vertex/Index Buffers, UBO Buffers, Descriptor Pool/Sets, Command Buffers ---
        let command_pool = create_command_pool(&device, &queue_indices);
        let (quad_vertex_buffer, quad_vertex_buffer_memory) = create_quad_vertex_buffer(&device, command_pool, graphics_queue, &physical_device_memory_properties);
        let (quad_index_buffer, quad_index_buffer_memory) = create_quad_index_buffer(&device, command_pool, graphics_queue, &physical_device_memory_properties);

        let mut ubo_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut ubo_buffers_memory = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut ubo_buffers_mapped = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let buffer_size = std::mem::size_of::<UbershaderParamsUBO>() as vk::DeviceSize;
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, memory) = create_buffer(&device, buffer_size, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, &physical_device_memory_properties).expect("UBO buffer creation failed");
            let mapped = unsafe { device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty()).expect("Failed to map UBO memory") };
            ubo_buffers.push(buffer); ubo_buffers_memory.push(memory); ubo_buffers_mapped.push(mapped);
        }

        let pool_sizes = [vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(MAX_FRAMES_IN_FLIGHT as u32).build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(MAX_FRAMES_IN_FLIGHT as u32);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).expect("Descriptor Pool failed") };
        let layouts = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(descriptor_pool).set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).expect("Descriptor Set allocation failed") };
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo::builder().buffer(ubo_buffers[i]).offset(0).range(std::mem::size_of::<UbershaderParamsUBO>() as u64).build();
            let ubo_write = vk::WriteDescriptorSet::builder().dst_set(descriptor_sets[i]).dst_binding(0).dst_array_element(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&buffer_info)).build();
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&ubo_write), &[]); }
        }

        let cmd_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&cmd_buffer_alloc_info).expect("Command buffer allocation failed") };

        // --- Sync Objects ---
        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let sem_info = vk::SemaphoreCreateInfo::builder().build();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED).build();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(unsafe { device.create_semaphore(&sem_info, None).expect("Semaphore failed") });
            render_finished_semaphores.push(unsafe { device.create_semaphore(&sem_info, None).expect("Semaphore failed") });
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None).expect("Fence failed") });
        }

        // Font atlas generation (moved after Vulkan setup for clarity, not strictly necessary here)
        match generate_font_atlas(None, 32.0, None) {
            Ok(atlas) => { println!("Font atlas: {}x{} ({} glyphs)", atlas.width, atlas.height, atlas.glyphs.len()); }
            Err(e) => { eprintln!("Font atlas error: {}", e); }
        }

        println!("VulkanApp fully initialized.");
        Self {
            _entry: entry, instance, debug_utils_loader: debug_loader, debug_messenger, surface_loader, surface, physical_device, physical_device_memory_properties, device,
            _queue_family_indices: queue_indices, graphics_queue, present_queue, swapchain_loader, swapchain, _swapchain_images: swapchain_images,
            swapchain_format: surface_format.format, swapchain_extent: extent, swapchain_image_views,
            render_pass, descriptor_set_layout, pipeline_layout, graphics_pipeline, swapchain_framebuffers, command_pool,
            quad_vertex_buffer, quad_vertex_buffer_memory, quad_index_buffer, quad_index_buffer_memory,
            ubo_buffers, ubo_buffers_memory, ubo_buffers_mapped, descriptor_pool, descriptor_sets, command_buffers,
            image_available_semaphores, render_finished_semaphores, in_flight_fences, current_frame: 0,
        }
    }

    fn draw_frame(&mut self) -> Result<(), vk::Result> {
        let fence = self.in_flight_fences[self.current_frame];
        unsafe { self.device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?; }

        let (image_index, _is_suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain, u64::MAX,
                self.image_available_semaphores[self.current_frame], vk::Fence::null(),
            )?
        };

        unsafe { self.device.reset_fences(std::slice::from_ref(&fence))?; }
        let cmd_buffer = self.command_buffers[self.current_frame];
        unsafe { self.device.reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())?; }

        // Record command buffer
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd_buffer, &begin_info)?; }

        let clear_values = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.1, 1.0] } }];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index as usize])
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.swapchain_extent })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(cmd_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
            self.device.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);

            let viewports = [vk::Viewport{ x:0.0, y:0.0, width: self.swapchain_extent.width as f32, height: self.swapchain_extent.height as f32, min_depth:0.0, max_depth:1.0}];
            self.device.cmd_set_viewport(cmd_buffer, 0, &viewports);
            let scissors = [vk::Rect2D{offset:vk::Offset2D{x:0,y:0}, extent: self.swapchain_extent}];
            self.device.cmd_set_scissor(cmd_buffer,0,&scissors);

            self.device.cmd_bind_vertex_buffers(cmd_buffer, 0, &[self.quad_vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(cmd_buffer, self.quad_index_buffer, 0, vk::IndexType::UINT32);
            self.device.cmd_bind_descriptor_sets(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.descriptor_sets[self.current_frame]], &[]);

            // Update UBO
            let ubo = UbershaderParamsUBO {
                shader_type: 0, // SolidColor
                base_color: glm::vec4(0.8, 0.2, 0.2, 1.0),
                border_color: glm::vec4(0.0, 0.0, 0.0, 0.0),
                border_thickness: 0.0,
                _padding1: [0.0; 3],
            };
            let ptr = self.ubo_buffers_mapped[self.current_frame] as *mut UbershaderParamsUBO;
            ptr.copy_from_nonoverlapping(&ubo, 1);

            self.device.cmd_draw_indexed(cmd_buffer, 6, 1, 0, 0, 0); // Draw quad
            self.device.cmd_end_render_pass(cmd_buffer);
            self.device.end_command_buffer(cmd_buffer)?;
        }

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let submit_infos = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores).wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd_buffer))
            .signal_semaphores(&signal_semaphores).build()];
        unsafe { self.device.queue_submit(self.graphics_queue, &submit_infos, fence)?; }

        let swapchains = [self.swapchain];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(std::slice::from_ref(&image_index));
        unsafe { self.swapchain_loader.queue_present(self.present_queue, &present_info)?; }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait device idle"); // Important before cleanup
            println!("Destroying VulkanApp...");
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
                self.device.unmap_memory(self.ubo_buffers_memory[i]); // Unmap UBOs
                self.device.destroy_buffer(self.ubo_buffers[i], None);
                self.device.free_memory(self.ubo_buffers_memory[i], None);
            }
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_buffer(self.quad_vertex_buffer, None); self.device.free_memory(self.quad_vertex_buffer_memory, None);
            self.device.destroy_buffer(self.quad_index_buffer, None); self.device.free_memory(self.quad_index_buffer_memory, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for framebuffer in self.swapchain_framebuffers.drain(..) { self.device.destroy_framebuffer(framebuffer, None); }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for iv in self.swapchain_image_views.drain(..) { self.device.destroy_image_view(iv, None); }
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
            println!("VulkanApp fully destroyed.");
        }
    }
}

// ... (Helper functions: create_shader_module, create_buffer, find_memory_type, copy_buffer, create_render_pass, create_graphics_pipeline, create_quad_buffers, etc.) ...
// (These will be part of the main.rs content below)

fn create_shader_module(device: &ash::Device, code: &[u32]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo::builder().code(code);
    unsafe { device.create_shader_module(&create_info, None).expect("Shader module creation failed") }
}

fn find_memory_type(type_filter: u32, properties: vk::MemoryPropertyFlags, mem_properties: &vk::PhysicalDeviceMemoryProperties) -> u32 {
    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 && (mem_properties.memory_types[i as usize].property_flags & properties) == properties {
            return i;
        }
    }
    panic!("Failed to find suitable memory type!");
}

fn create_buffer(device: &ash::Device, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags, mem_props: &vk::PhysicalDeviceMemoryProperties) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let buffer_info = vk::BufferCreateInfo::builder().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None).map_err(|e| format!("Buffer creation failed: {}", e))? };
    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type(mem_requirements.memory_type_bits, properties, mem_props);
    let alloc_info = vk::MemoryAllocateInfo::builder().allocation_size(mem_requirements.size).memory_type_index(memory_type_index);
    let buffer_memory = unsafe { device.allocate_memory(&alloc_info, None).map_err(|e| format!("Memory allocation failed: {}", e))? };
    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0).map_err(|e| format!("Bind buffer memory failed: {}", e))? };
    Ok((buffer, buffer_memory))
}

fn copy_buffer(device: &ash::Device, command_pool: vk::CommandPool, graphics_queue: vk::Queue, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: vk::DeviceSize) -> Result<(), String> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder().level(vk::CommandBufferLevel::PRIMARY).command_pool(command_pool).command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info).map_err(|e| format!("Temp CB alloc failed: {}", e))? }[0];
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(command_buffer, &begin_info).map_err(|e| format!("Temp CB begin failed: {}", e))?; }
    let copy_regions = [vk::BufferCopy::builder().src_offset(0).dst_offset(0).size(size).build()];
    unsafe { device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions); }
    unsafe { device.end_command_buffer(command_buffer).map_err(|e| format!("Temp CB end failed: {}", e))?; }
    let submit_infos = [vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer)).build()];
    unsafe {
        device.queue_submit(graphics_queue, &submit_infos, vk::Fence::null()).map_err(|e| format!("Temp CB submit failed: {}", e))?;
        device.queue_wait_idle(graphics_queue).map_err(|e| format!("Queue wait idle failed: {}", e))?; // Ensure copy completes
        device.free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
    }
    Ok(())
}

fn create_quad_vertex_buffer(device: &ash::Device, command_pool: vk::CommandPool, graphics_queue: vk::Queue, mem_props: &vk::PhysicalDeviceMemoryProperties) -> (vk::Buffer, vk::DeviceMemory) {
    let vertices = [
        Vertex { pos: glm::vec2(-0.5, -0.5), uv: glm::vec2(0.0, 0.0) }, Vertex { pos: glm::vec2(0.5, -0.5), uv: glm::vec2(1.0, 0.0) },
        Vertex { pos: glm::vec2(0.5, 0.5), uv: glm::vec2(1.0, 1.0) }, Vertex { pos: glm::vec2(-0.5, 0.5), uv: glm::vec2(0.0, 1.0) },
    ];
    let buffer_size = (std::mem::size_of_val(&vertices)) as vk::DeviceSize;
    let (staging_buffer, staging_memory) = create_buffer(device, buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, mem_props).expect("Staging VB failed");
    unsafe { let data_ptr = device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).expect("Map VB failed"); std::ptr::copy_nonoverlapping(vertices.as_ptr(), data_ptr as *mut Vertex, vertices.len()); device.unmap_memory(staging_memory); }
    let (vertex_buffer, vertex_memory) = create_buffer(device, buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL, mem_props).expect("Device VB failed");
    copy_buffer(device, command_pool, graphics_queue, staging_buffer, vertex_buffer, buffer_size).expect("Copy VB failed");
    unsafe { device.destroy_buffer(staging_buffer, None); device.free_memory(staging_memory, None); }
    (vertex_buffer, vertex_memory)
}

fn create_quad_index_buffer(device: &ash::Device, command_pool: vk::CommandPool, graphics_queue: vk::Queue, mem_props: &vk::PhysicalDeviceMemoryProperties) -> (vk::Buffer, vk::DeviceMemory) {
    let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
    let buffer_size = (std::mem::size_of_val(&indices)) as vk::DeviceSize;
    let (staging_buffer, staging_memory) = create_buffer(device, buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, mem_props).expect("Staging IB failed");
    unsafe { let data_ptr = device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty()).expect("Map IB failed"); std::ptr::copy_nonoverlapping(indices.as_ptr(), data_ptr as *mut u32, indices.len()); device.unmap_memory(staging_memory); }
    let (index_buffer, index_memory) = create_buffer(device, buffer_size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::DEVICE_LOCAL, mem_props).expect("Device IB failed");
    copy_buffer(device, command_pool, graphics_queue, staging_buffer, index_buffer, buffer_size).expect("Copy IB failed");
    unsafe { device.destroy_buffer(staging_buffer, None); device.free_memory(staging_memory, None); }
    (index_buffer, index_memory)
}

fn create_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::builder().format(format).samples(vk::SampleCountFlags::TYPE_1).load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE).stencil_load_op(vk::AttachmentLoadOp::DONT_CARE).stencil_store_op(vk::AttachmentStoreOp::DONT_CARE).initial_layout(vk::ImageLayout::UNDEFINED).final_layout(vk::ImageLayout::PRESENT_SRC_KHR).build();
    let color_attachment_ref = vk::AttachmentReference::builder().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).build();
    let subpass = vk::SubpassDescription::builder().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).color_attachments(std::slice::from_ref(&color_attachment_ref)).build();
    let dependency = vk::SubpassDependency::builder().src_subpass(vk::SUBPASS_EXTERNAL).dst_subpass(0).src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT).src_access_mask(vk::AccessFlags::empty()).dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT).dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE).build();
    let render_pass_info = vk::RenderPassCreateInfo::builder().attachments(std::slice::from_ref(&color_attachment)).subpasses(std::slice::from_ref(&subpass)).dependencies(std::slice::from_ref(&dependency));
    unsafe { device.create_render_pass(&render_pass_info, None).expect("Render pass failed") }
}

fn create_graphics_pipeline(device: &ash::Device, render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout, swapchain_extent: vk::Extent2D, vert_module: vk::ShaderModule, frag_module: vk::ShaderModule) -> vk::Pipeline {
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(CStr::from_bytes_with_nul(b"main\0").unwrap()).build(),
        vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(CStr::from_bytes_with_nul(b"main\0").unwrap()).build(),
    ];
    let binding_descs = [Vertex::get_binding_description()];
    let attribute_descs = Vertex::get_attribute_descriptions();
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder().vertex_binding_descriptions(&binding_descs).vertex_attribute_descriptions(&attribute_descs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST).primitive_restart_enable(false);
    let viewport = vk::Viewport{ x:0.0, y:0.0, width: swapchain_extent.width as f32, height: swapchain_extent.height as f32, min_depth:0.0, max_depth:1.0};
    let scissor = vk::Rect2D{offset:vk::Offset2D{x:0,y:0}, extent: swapchain_extent};
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder().depth_clamp_enable(false).rasterizer_discard_enable(false).polygon_mode(vk::PolygonMode::FILL).line_width(1.0).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE); // No culling for 2D UI
    let multisampling = vk::PipelineMultisampleStateCreateInfo::builder().sample_shading_enable(false).rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A).blend_enable(true).src_color_blend_factor(vk::BlendFactor::SRC_ALPHA).dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA).color_blend_op(vk::BlendOp::ADD).src_alpha_blend_factor(vk::BlendFactor::ONE).dst_alpha_blend_factor(vk::BlendFactor::ZERO).alpha_blend_op(vk::BlendOp::ADD);
    let color_blending = vk::PipelineColorBlendStateCreateInfo::builder().logic_op_enable(false).attachments(std::slice::from_ref(&color_blend_attachment));
    let pipeline_info_vec = [vk::GraphicsPipelineCreateInfo::builder().stages(&shader_stages).vertex_input_state(&vertex_input_info).input_assembly_state(&input_assembly).viewport_state(&viewport_state).rasterization_state(&rasterizer).multisample_state(&multisampling).color_blend_state(&color_blending).layout(pipeline_layout).render_pass(render_pass).subpass(0).build()];
    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info_vec, None).map_err(|e|e.1).expect("Graphics pipeline creation failed")[0] }
}

fn create_command_pool(device: &ash::Device, queue_indices: &QueueFamilyIndices) -> vk::CommandPool {
    let pool_info = vk::CommandPoolCreateInfo::builder().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(queue_indices.graphics_family.unwrap());
    unsafe { device.create_command_pool(&pool_info, None).expect("Command Pool creation failed") }
}

fn create_image_view(device: &ash::Device, image: vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags) -> vk::ImageView {
    let components = vk::ComponentMapping::default();
    let subresource_range = vk::ImageSubresourceRange::builder().aspect_mask(aspect_flags).base_mip_level(0).level_count(1).base_array_layer(0).layer_count(1).build();
    let create_info = vk::ImageViewCreateInfo::builder().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).components(components).subresource_range(subresource_range);
    unsafe { device.create_image_view(&create_info, None).expect("Failed to create image view") }
}


// --- Functions from previous steps (query_swapchain_support, choose_*, pick_physical_device, etc.) ---
// (Condensed for brevity, assuming they are correct from previous steps)
fn query_swapchain_support(physical_device: vk::PhysicalDevice, surface_loader: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR) -> SwapchainSupportDetails {
    unsafe {
        let c = surface_loader.get_physical_device_surface_capabilities(physical_device, surface).expect("Caps failed");
        let f = surface_loader.get_physical_device_surface_formats(physical_device, surface).expect("Formats failed");
        let p = surface_loader.get_physical_device_surface_present_modes(physical_device, surface).expect("Modes failed");
        SwapchainSupportDetails { capabilities: c, formats: f, present_modes: p }
    }
}
fn choose_swap_surface_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    *available_formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or_else(|| { println!("Warning: B8G8R8A8_SRGB not available."); &available_formats[0] })
}
fn choose_swap_present_mode(available_present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) { vk::PresentModeKHR::MAILBOX } else { vk::PresentModeKHR::FIFO }
}
fn choose_swap_extent(capabilities: &vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX { capabilities.current_extent } else {
        let size = window.inner_size();
        vk::Extent2D { width: size.width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width), height: size.height.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height) }
    }
}
fn pick_physical_device(instance: &ash::Instance, surface_loader: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR) -> (vk::PhysicalDevice, QueueFamilyIndices) {
    let devices = unsafe { instance.enumerate_physical_devices().expect("Enum devices failed") };
    if devices.is_empty() { panic!("No Vulkan GPUs!"); }
    for &device in devices.iter() {
        let indices = find_queue_families(instance, device, surface_loader, surface);
        if is_physical_device_suitable(instance, device, surface_loader, surface, &indices) { return (device, indices); }
    }
    panic!("No suitable GPU found!");
}
fn is_physical_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice, surface_loader: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR, indices: &QueueFamilyIndices) -> bool {
    let ext_ok = check_device_extension_support(instance, device);
    let mut swap_ok = false;
    if ext_ok {
        let caps = unsafe { surface_loader.get_physical_device_surface_capabilities(device, surface) };
        let formats = unsafe { surface_loader.get_physical_device_surface_formats(device, surface) };
        let modes = unsafe { surface_loader.get_physical_device_surface_present_modes(device, surface) };
        swap_ok = caps.is_ok() && formats.map_or(false, |f| !f.is_empty()) && modes.map_or(false, |p| !p.is_empty());
    }
    indices.is_complete() && ext_ok && swap_ok
}
fn find_queue_families(instance: &ash::Instance, device: vk::PhysicalDevice, surface_loader: &ash::extensions::khr::Surface, surface: vk::SurfaceKHR) -> QueueFamilyIndices {
    let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut indices = QueueFamilyIndices::default();
    for (i, family) in props.iter().enumerate() {
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) { indices.graphics_family = Some(i as u32); }
        if unsafe { surface_loader.get_physical_device_surface_support(device, i as u32, surface).unwrap_or(false) } { indices.present_family = Some(i as u32); }
        if indices.is_complete() { break; }
    }
    indices
}
fn check_device_extension_support(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
    let required = [ash::extensions::khr::Swapchain::name()];
    let available = unsafe { instance.enumerate_device_extension_properties(device).unwrap_or_default() };
    let mut names = HashSet::new();
    for ext in available.iter() { names.insert(unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }.to_owned()); }
    for req in required.iter() { if !names.contains(*req) { return false; } }
    true
}
fn create_logical_device(instance: &ash::Instance, device: vk::PhysicalDevice, indices: &QueueFamilyIndices) -> (ash::Device, vk::Queue, vk::Queue) {
    let mut unique_families = HashSet::new();
    unique_families.insert(indices.graphics_family.unwrap()); unique_families.insert(indices.present_family.unwrap());
    let prio = 1.0f32; let mut q_infos = vec![];
    for &idx in unique_families.iter() { q_infos.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(idx).queue_priorities(&[prio]).build()); }
    let features = vk::PhysicalDeviceFeatures::builder().build();
    let exts_raw: Vec<*const c_char> = [ash::extensions::khr::Swapchain::name().as_ptr()].to_vec();
    let dev_info = vk::DeviceCreateInfo::builder().queue_create_infos(&q_infos).enabled_features(&features).enabled_extension_names(&exts_raw);
    let l_device = unsafe { instance.create_device(device, &dev_info, None).expect("Logical device failed") };
    let gfx_q = unsafe { l_device.get_device_queue(indices.graphics_family.unwrap(), 0) };
    let present_q = unsafe { l_device.get_device_queue(indices.present_family.unwrap(), 0) };
    (l_device, gfx_q, present_q)
}

pub fn generate_font_atlas(font_path_option: Option<&str>, font_size: f32, char_set: Option<Vec<char>>) -> Result<FontAtlas, String> {
    let default_chars: Vec<char> = (32u8..=126u8).map(|c| c as char).collect();
    let characters_to_render = char_set.unwrap_or(default_chars);
    let font_data = match font_path_option {
        Some(path) => std::fs::read(path).map_err(|e| format!("Font read error {}: {}", path, e)),
        None => {
            let paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"];
            let mut found = Err("System font not found.".to_string());
            for path in paths.iter() { if std::path::Path::new(path).exists() { println!("Loading font: {}", path); match std::fs::read(path) { Ok(d) => {found=Ok(d); break;}, Err(e)=>{found=Err(format!("Read error {}: {}",path,e));}}} else {println!("Not found: {}",path);}}
            if found.is_err() {println!("Searched: {:?}", paths);} found
        }
    }?;
    let font = Font::try_from_vec(font_data).ok_or_else(|| "Font parse error".to_string())?;
    let scale = Scale::uniform(font_size); let v_metrics: VMetrics = font.v_metrics(scale); let ascent = v_metrics.ascent;
    let mut glyph_bm_temp = Vec::new(); let mut total_w_pack = 0u32; let mut max_r_h = 0u32;
    for char_code in characters_to_render {
        let glyph = font.glyph(char_code).scaled(scale); let h_metrics = glyph.h_metrics();
        let positioned_glyph = glyph.positioned(point(0.0, ascent));
        if let Some(outline) = positioned_glyph.pixel_bounding_box() {
            let mut bitmap = vec![0u8; (outline.width()*outline.height()) as usize];
            positioned_glyph.draw(|x,y,v| if x<outline.width() as u32 && y<outline.height() as u32 { bitmap[(y*outline.width() as u32+x) as usize]=(v*255.0)as u8; });
            glyph_bm_temp.push((char_code, bitmap, outline.width() as u32, outline.height() as u32, h_metrics.advance_width));
            total_w_pack += outline.width() as u32 + 1; max_r_h = max_r_h.max(outline.height() as u32);
        } else { glyph_bm_temp.push((char_code, Vec::new(), 0,0,h_metrics.advance_width)); }
    }
    if total_w_pack == 0 || max_r_h == 0 { println!("Warning: No renderable glyphs. Empty atlas."); return Ok(FontAtlas::new());}
    let atlas_w = total_w_pack; let atlas_h = max_r_h;
    let mut atlas_data = vec![0u8; (atlas_w*atlas_h) as usize];
    let mut glyph_map = std::collections::HashMap::new(); let mut current_x = 0u32;
    for (cc, bm_data, w, h, adv_w) in glyph_bm_temp {
        if w>0 && h>0 { for gy in 0..h { for gx in 0..w { let atlas_idx = ((gy*atlas_w)+(current_x+gx))as usize; let bm_idx=(gy*w+gx)as usize; if bm_idx<bm_data.len()&&atlas_idx<atlas_data.len(){atlas_data[atlas_idx]=bm_data[bm_idx];}}}}
        glyph_map.insert(cc, GlyphInfo{char_code:cc, atlas_x:current_x, atlas_y:0, width:w, height:h, advance_width:adv_w});
        if w>0 {current_x+=w+1;}
    }
    if atlas_w>0&&atlas_h>0 { match GrayImage::from_raw(atlas_w,atlas_h,atlas_data.clone()){ Some(_)=>{/* Saved in debug */},None=>{return Err("GrayImage failed".to_string());}}}
    Ok(FontAtlas{texture_data:atlas_data, width:atlas_w, height:atlas_h, format:vk::Format::R8_UNORM, glyphs:glyph_map})
}

fn compile_shader(glsl: &str, kind: shaderc::ShaderKind, name: &str, entry: &str) -> Result<Vec<u32>, String> {
    let compiler = shaderc::Compiler::new().ok_or("shaderc new failed")?;
    let mut opts = shaderc::CompileOptions::new().ok_or("shaderc options new failed")?;
    opts.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
    match compiler.compile_into_spirv(glsl,kind,name,entry,Some(&opts)) {
        Ok(art) => { if art.get_num_warnings()>0 {println!("Shader warnings {}:\n{}",name,art.get_warning_messages());} Ok(art.as_binary().to_vec()) }
        Err(e) => Err(format!("Compile error {}: {}",name,e)),
    }
}

fn main() {
    let el = EventLoop::new().expect("EventLoop failed");
    let window = WindowBuilder::new().with_title("Vulkan Ubershader UI").with_inner_size(winit::dpi::LogicalSize::new(800,600)).build(&el).expect("Window build failed");
    let mut app = VulkanApp::new(&window);
    el.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => { elwt.exit(); }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                app.draw_frame().unwrap_or_else(|e| eprintln!("draw_frame error: {:?}", e));
            }
            Event::AboutToWait => { // Continuously request redraw for basic animation/responsiveness
                window.request_redraw();
            }
            _ => (),
        }
    }).expect("EventLoop run failed");
    println!("Exiting.");
}
