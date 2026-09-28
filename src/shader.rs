// SPDX-License-Identifier: GPL-3.0-or-later

//! GPU detail shader — renders mono image data with the live density develop.
//!
//! The mono `Vec<f32>` is uploaded to the GPU once as an `R16Float` texture.
//! The pointwise density develop ([`film::Develop`]) is applied per fragment
//! from uniforms — zero CPU re-encoding, zero new `Handle` per frame. The WGSL
//! `develop()` mirrors the Rust twin exactly (parity-tested), so the detail
//! preview, the grid bakes, and exports share one tone model. See
//! `docs/raw-pipeline-rewrite.md`.

use cosmic::iced::core::{Length, Rectangle};
use cosmic::iced::wgpu::util::DeviceExt;
use cosmic::iced::widget::shader::{Pipeline, Primitive, Program, Shader, Viewport};

use crate::edit_manifest::CropMargins;
use crate::film::Develop;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// GPU-backed detail view image with live exposure adjustment.
pub struct DetailProgram {
    mono: Vec<f32>,
    width: u32,
    height: u32,
    /// The full-resolution (post-masked-border, pre-downscale) display-oriented
    /// source dimensions the crop margins are authored against, restored from
    /// the downscaled texture + the sensor's true long edge at construction
    /// (both share the same aspect, so the scale per axis is recovered). The
    /// texture is a downscale of this; the shader maps source-pixel crop
    /// margins onto the texture by `tex/src` per axis. At the native level-up
    /// (texture == source) the scale collapses to 1 and the crop is exact.
    src_w: u32,
    src_h: u32,
    /// The live pointwise density develop applied per fragment: EV gain →
    /// `log10(base/value)` density → black/white window → pivot-power contrast.
    /// The shader's [`Uniforms.dev_*`] block mirrors this exactly.
    develop: Develop,
    /// Detail-view zoom in `log2` units: 1.0 = contain fit, each +1 doubles
    /// the rendered scale (see [`DetailPrimitive::prepare`]).
    zoom: f32,
    /// Pan offset of the image center from the widget center, in logical
    /// points; converted to physical pixels on the GPU side.
    pan: cosmic::iced::Point,
    /// Live source-pixel crop margins removed from each edge of the
    /// full-resolution display-oriented frame. Applied as a uniform UV-remap
    /// (no texture re-upload): at render time [`crop_uv_geometry`] scales them
    /// onto the downscaled texture by `tex_axis/src_axis`, so the live trim and
    /// the CPU thumbnail bake (which crops the same source-pixel margins onto
    /// the print) stay in the same reference frame. A keyboard trim is instant.
    /// `CropMargins::default()` (all zero) shows the full frame.
    crop: CropMargins,
    /// User display rotation as cumulative counter-clockwise 90° quarter-turns
    /// (`0`…`3`), applied ON TOP of the EXIF-upright texture as a uniform-only
    /// UV remap (no texture re-upload). The crop sub-rectangle is authored in
    /// the EXIF-upright frame and is rotated along with the display, so the
    /// crop math is untouched. `0` = identity.
    rotation: u8,
    /// Whether the "view dimmed crop area" overlay is shown: the full
    /// uncropped frame drawn on top of the zoomed crop with everything outside
    /// the crop rectangle dimmed. View-only — never persisted or baked.
    show_mask: bool,
    /// Minimum padding (logical points) kept around the image in crop mode:
    /// the WGSL contain-fit base shrinks by `pad` on every side so a white
    /// margin is always visible; zooming in grows the image into it. `0.0`
    /// (normal mode) is the exact previous layout. Set via [`Self::set_pad`].
    pad: f32,
    /// Monotonic id bumped by the app model on each new detail decode. Used
    /// to detect image changes and rebuild the GPU texture/bind group.
    image_id: u64,
}

impl DetailProgram {
    /// Create a new program for the given mono image.
    ///
    /// `mono` is TRUE sensor-linear data — linear `[0,1]` relative to the
    /// sensor white point — the same for every preset (one `f32` per pixel,
    /// row-major, top-to-bottom). `width`/`height` are the **texture** dims
    /// (the downscale of the source, display-oriented). `src_long_edge` is the
    /// sensor's true long edge AFTER masked-border cropping and BEFORE the
    /// downscale — it restores the full-resolution display-oriented source dims
    /// (`source_dimensions`) that the crop margins are authored against.
    /// `develop` is the frame's resolved density develop (its `exposure_ev`
    /// drives the per-fragment `2^-EV` sensor gain). `crop` is the frame's
    /// stored source-pixel crop (all-zero for a fresh frame). `rotation` is the
    /// cumulative counter-clockwise 90° quarter-turn display rotation on top of
    /// the EXIF-upright texture (`0` = none). The view starts at contain fit
    /// (zoom 1.0, no pan).
    #[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
    pub fn new(
        mono: Vec<f32>,
        width: u32,
        height: u32,
        develop: Develop,
        crop: CropMargins,
        rotation: u8,
        image_id: u64,
        src_long_edge: u32,
    ) -> Self {
        // The texture shares the display source's aspect (resize_area +
        // orientation preserve it within floor-rounding), so the two source
        // axes are recovered from the long edge: the long-edged axis equals
        // `src_long_edge`, the short axis scales its texture counterpart.
        let (src_w, src_h) = if width >= height {
            (
                src_long_edge.max(1),
                (((u64::from(height) * u64::from(src_long_edge)) + u64::from(width) / 2)
                    / u64::from(width.max(1))) as u32,
            )
        } else {
            (
                (((u64::from(width) * u64::from(src_long_edge)) + u64::from(height) / 2)
                    / u64::from(height.max(1))) as u32,
                src_long_edge.max(1),
            )
        };
        Self {
            mono,
            width,
            height,
            src_w,
            src_h,
            develop,
            zoom: 1.0,
            pan: cosmic::iced::Point::default(),
            crop,
            rotation,
            show_mask: false,
            pad: 0.0,
            image_id,
        }
    }

    /// Update the exposure value (called on slider drag): the pointwise develop
    /// reads it as the `2^-EV` sensor gain, so +EV brightens the positive.
    pub fn set_exposure(&mut self, ev: f32) {
        self.develop.exposure_ev = ev;
    }

    /// Update the zoom/pan transform (called on detail-view wheel/drag).
    ///
    /// `zoom` is in `log2` units (1.0 = contain fit). `pan` is in logical
    /// points relative to the widget center.
    pub fn set_view(&mut self, zoom: f32, pan: cosmic::iced::Point) {
        self.zoom = zoom;
        self.pan = pan;
    }

    /// Replace the live density develop (called on editing-drawer sliders). The
    /// uploaded texture is untouched — only the `dev_*` uniforms change and the
    /// WGSL re-develops every fragment.
    pub fn set_develop(&mut self, develop: Develop) {
        self.develop = develop;
    }

    /// Update the live crop margins (called on each keyboard trim).
    ///
    /// Only the four crop uniforms change — the uploaded texture stays intact.
    /// The WGSL zoom-to-fits the CROPPED region into the widget; the overlay
    /// full-frame layer (see [`Self::set_show_mask`]) dims outside the crop.
    pub fn set_crop(&mut self, crop: CropMargins) {
        self.crop = crop;
    }

    /// Update the live display rotation (called on each rotate press/click).
    ///
    /// Only the one uniform changes — the uploaded texture stays intact. The
    /// WGSL rotates the already-cropped display frame counter-clockwise by
    /// `rotation & 3` quarter-turns; `0` is the identity (EXIF-upright).
    pub fn set_rotation(&mut self, rotation: u8) {
        self.rotation = rotation;
    }

    /// Show or hide the "view dimmed crop area" overlay (called on toggle).
    ///
    /// When on, the shader draws the full uncropped frame on top of the zoomed
    /// crop (same zoom/pan) and dims everything outside the crop rectangle,
    /// letting the user see where the crop lands over the whole negative.
    /// View-only state: never persisted or applied to grid/cover bakes.
    pub fn set_show_mask(&mut self, on: bool) {
        self.show_mask = on;
    }

    /// Set the minimum padding (logical points) kept around the image in crop
    /// mode. The WGSL contain-fit base shrinks by `pad` on every side so a
    /// white margin is always visible; zooming in grows the image into it.
    /// `0.0` (normal mode) restores the exact unpadded layout.
    pub fn set_pad(&mut self, pad: f32) {
        self.pad = pad;
    }

    /// The full-resolution display-oriented source dimensions the persisted
    /// crop margins are authored against (the sensor's post-masked-border dims,
    /// oriented, before any downscale). The crop keyboard math must bound its
    /// trims against THIS frame — the downscaled texture dims would store a
    /// texture-pixel crop the thumbnail bake mis-scales.
    #[must_use]
    pub fn source_dimensions(&self) -> (u32, u32) {
        (self.src_w, self.src_h)
    }

    /// The zoom level (in `log2` units, `1.0` = contain fit) at which the image
    /// renders at 100% — one texture pixel per one physical screen pixel.
    ///
    /// Mirrors the WGSL's `contain = min(sc_w/dw, sc_h/dh)` with the live crop
    /// and display rotation applied, so the cap tracks the ACTUAL on-screen
    /// scale. `widget_w`/`widget_h` are the preview area's LOGICAL size and
    /// `scale_factor` converts them to physical pixels, matching the shader's
    /// `sc_w = bounds · scale_factor`. Floored at 1.0 (contain fit) so the cap
    /// never drops below the minimum zoom.
    #[must_use]
    pub fn zoom_100(&self, widget_w: f32, widget_h: f32, scale_factor: f32) -> f32 {
        self.zoom_100_for(self.width, self.height, widget_w, widget_h, scale_factor)
    }

    /// The same 1:1 ("100%") cap math as [`Self::zoom_100`], but for a
    /// hypothetical texture of `tex_w` × `tex_h` under the live crop, display
    /// rotation and pad. Lets the detail view project the cap a pending native
    /// (level-up) decode will unlock while only the coarse overview is
    /// installed, so zooming can continue through the load without ever
    /// passing the point the upcoming texture's pixels cover.
    #[must_use]
    pub fn zoom_100_for(
        &self,
        tex_w: u32,
        tex_h: u32,
        widget_w: f32,
        widget_h: f32,
        scale_factor: f32,
    ) -> f32 {
        let (_, (cw, ch)) = crop_uv_geometry(self.crop, tex_w, tex_h, self.src_w, self.src_h);
        let (dw, dh) = if (self.rotation & 1) != 0 { (ch, cw) } else { (cw, ch) };
        // The 1:1 cap mirrors the WGSL contain-fit, which in crop mode shrinks
        // the available box by the minimum padding on every side (`pad` is in
        // logical points, converted to physical like the scissor rect).
        let sc_w = ((widget_w - 2.0 * self.pad) * scale_factor).max(1.0);
        let sc_h = ((widget_h - 2.0 * self.pad) * scale_factor).max(1.0);
        1.0 + (dw / sc_w).max(dh / sc_h).log2().max(0.0)
    }

    /// Wrap in a `Shader` widget sized to fill the parent.
    pub fn view<M>(&self) -> Shader<M, Self> {
        Shader::new(self.clone())
            .width(Length::Fill)
            .height(Length::Fill)
    }
}

impl Clone for DetailProgram {
    fn clone(&self) -> Self {
        Self {
            mono: self.mono.clone(),
            width: self.width,
            height: self.height,
            src_w: self.src_w,
            src_h: self.src_h,
            develop: self.develop,
            zoom: self.zoom,
            pan: self.pan,
            crop: self.crop,
            rotation: self.rotation,
            show_mask: self.show_mask,
            pad: self.pad,
            image_id: self.image_id,
        }
    }
}

impl std::fmt::Debug for DetailProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetailProgram")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("src_w", &self.src_w)
            .field("src_h", &self.src_h)
            .field("develop", &self.develop)
            .field("zoom", &self.zoom)
            .field("pan", &self.pan)
            .field("crop", &self.crop)
            .field("rotation", &self.rotation)
            .field("show_mask", &self.show_mask)
            .field("pad", &self.pad)
            .field("mono_len", &self.mono.len())
            .field("image_id", &self.image_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// iced::widget::shader::Program implementation
// ---------------------------------------------------------------------------

impl<M> Program<M> for DetailProgram {
    type State = ();
    type Primitive = DetailPrimitive;

    fn draw(
        &self,
        _state: &(),
        _cursor: cosmic::iced::mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        DetailPrimitive {
            mono: self.mono.clone(),
            develop: self.develop,
            zoom: self.zoom,
            pan: self.pan,
            crop: self.crop,
            rotation: self.rotation,
            show_mask: self.show_mask,
            pad: self.pad,
            width: self.width,
            height: self.height,
            src_w: self.src_w,
            src_h: self.src_h,
            image_id: self.image_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Crop-geometry pure helpers
// ---------------------------------------------------------------------------

/// The live-crop sub-rectangle in **texture space** as `(origin, size)`, both
/// `(x, y)` pairs, given the source-pixel crop margins, the downscaled texture
/// dims (`tex_w`/`tex_h`) and the full-resolution display-oriented source dims
/// (`src_w`/`src_h`) the margins are authored against. Each margin axis scales
/// by `tex_axis/src_axis`, so a stored source-pixel crop removes the same
/// *fraction of the frame* at any texture resolution (overview 2048 or native)
/// and matches what the CPU thumbnail bake does. An all-zero crop yields origin
/// `(0, 0)` and size equal to the texture — the identity no-op the WGSL remap
/// leaves untouched.
#[allow(clippy::cast_precision_loss)]
#[must_use]
fn crop_uv_geometry(
    crop: CropMargins,
    tex_w: u32,
    tex_h: u32,
    src_w: u32,
    src_h: u32,
) -> ((f32, f32), (f32, f32)) {
    let sx = tex_w as f32 / src_w.max(1) as f32;
    let sy = tex_h as f32 / src_h.max(1) as f32;
    (
        (crop.left as f32 * sx, crop.top as f32 * sy),
        (
            crop.cropped_width(src_w) as f32 * sx,
            crop.cropped_height(src_h) as f32 * sy,
        ),
    )
}

// ---------------------------------------------------------------------------
// iced::wgpu::primitive::Primitive implementation
// ---------------------------------------------------------------------------

/// Per-frame data sent from `Program::draw()` to the GPU pipeline.
///
/// Carries a clone of the mono data on the first frame so the pipeline can
/// create the GPU texture.  `image_id` lets the pipeline detect when a new
/// image is selected (iceD caches pipelines per type, so we cannot rely on
/// `initialized` alone).
#[derive(Debug, Clone)]
pub struct DetailPrimitive {
    mono: Vec<f32>,
    /// The live pointwise density develop (see [`DetailProgram::develop`]).
    develop: Develop,
    /// Detail-view zoom in `log2` units; 1.0 = contain fit.
    zoom: f32,
    /// Pan offset of the image center from the widget center (logical points).
    pan: cosmic::iced::Point,
    /// Live source-pixel crop margins removed from each edge (uniform UV-remap).
    crop: CropMargins,
    /// User display rotation (cumulative CCW 90° quarter-turns, `0`…`3`),
    /// applied as a uniform UV remap on top of the cropped EXIF-upright frame.
    rotation: u8,
    /// Whether the "view dimmed crop area" overlay is shown (full frame on top
    /// of the zoomed crop, dimmed outside the crop rect).
    show_mask: bool,
    /// Minimum padding (logical points) kept around the image in crop mode.
    pad: f32,
    width: u32,
    height: u32,
    /// Full-resolution display-oriented source dims the crop margins are
    /// authored against (see [`crop_uv_geometry`]).
    src_w: u32,
    src_h: u32,
    image_id: u64,
}

impl Primitive for DetailPrimitive {
    type Pipeline = DetailPipeline;

    // Sequential wgpu uploads (texture, then uniforms) that must run in this
    // exact order each frame; splitting them into helper methods would only
    // scatter the pipeline state they all touch.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &cosmic::iced::wgpu::Device,
        queue: &cosmic::iced::wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        // --- Rebuild texture + bind group if image identity changed ---
        // `iced` caches the `DetailPipeline` across image selections because
        // both selections share the same pipeline type. `initialized` only
        // catches the very first frame; tracking `image_id` catches every
        // subsequent image swap too.
        let needs_new_texture =
            !pipeline.initialized || pipeline.current_image_id != Some(self.image_id);

        if needs_new_texture {
            // Pre-filter the linear mono into a full mip chain so minified
            // views (zoom back out after the native level-up) sample averaged
            // texels instead of aliasing the full-res grain into a moiré. The
            // averaging runs in linear light on the TRUE sensor data, and the
            // WGSL's per-fragment tone/inversion applies after sampling — the
            // same flatten-before-average ordering the CPU thumbnail bake uses.
            let mips = build_mip_chain(&self.mono, self.width, self.height);
            let tex = device.create_texture(&cosmic::iced::wgpu::TextureDescriptor {
                label: Some("exposure mono"),
                size: cosmic::iced::wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips.len() as u32,
                sample_count: 1,
                dimension: cosmic::iced::wgpu::TextureDimension::D2,
                format: cosmic::iced::wgpu::TextureFormat::R16Float,
                usage: cosmic::iced::wgpu::TextureUsages::TEXTURE_BINDING
                    | cosmic::iced::wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            for (level, mip) in mips.iter().enumerate() {
                queue.write_texture(
                    cosmic::iced::wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: cosmic::iced::wgpu::Origin3d::ZERO,
                        aspect: cosmic::iced::wgpu::TextureAspect::All,
                    },
                    &mip.data,
                    cosmic::iced::wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(mip.width * 2),
                        rows_per_image: Some(mip.height),
                    },
                    cosmic::iced::wgpu::Extent3d {
                        width: mip.width,
                        height: mip.height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            let texture_view =
                tex.create_view(&cosmic::iced::wgpu::TextureViewDescriptor::default());

            pipeline.texture = Some(texture_view);
            pipeline.bind_group = Some(device.create_bind_group(
                &cosmic::iced::wgpu::BindGroupDescriptor {
                    label: Some("exposure bg"),
                    layout: &pipeline.bind_group_layout,
                    entries: &[
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 0,
                            resource: cosmic::iced::wgpu::BindingResource::TextureView(
                                pipeline.texture.as_ref().unwrap(),
                            ),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 1,
                            resource: cosmic::iced::wgpu::BindingResource::Sampler(
                                &pipeline.sampler,
                            ),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pipeline.uniform_buf.as_entire_binding(),
                        },
                    ],
                },
            ));
            pipeline.current_image_id = Some(self.image_id);
            pipeline.initialized = true;
        }

        // --- Per-frame: update uniform buffer ---
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let sf = viewport.scale_factor() as f32;
        #[allow(clippy::cast_precision_loss)]
        let tex_w = self.width as f32;
        #[allow(clippy::cast_precision_loss)]
        let tex_h = self.height as f32;
        // Live crop: the texture sub-rectangle to show. Origin is the removed
        // left/top margins and size the kept region, mapped from the stored
        // source-pixel crop onto the DOWNSCALED texture by `tex/src` per axis
        // (see [`crop_uv_geometry`]); at the native level-up the texture IS the
        // source and the map is 1:1. The WGSL contain-fits the crop into the
        // widget (base view) and, when the dim overlay is on, also lays out the
        // full texture to show the crop in context. An all-zero crop → origin 0
        // and size == texture.
        let (crop_origin, (crop_w, crop_h)) =
            crop_uv_geometry(self.crop, self.width, self.height, self.src_w, self.src_h);
        let (crop_l, crop_t) = crop_origin;
        // Display rotation as a float quarter-turn count for the WGSL remap.
        // `0.0` = identity; the WGSL keeps it in {0,1,2,3} by construction.
        let rot = f32::from(self.rotation & 3);
        // The sensor gain is `2^-EV`: +EV lowers the transmission, raises the
        // density, and brightens the positive. Computed once per frame from the
        // develop's exposure; the WGSL reads it as a direct multiplier.
        let exposure = (-self.develop.exposure_ev).exp2();
        let uniforms = Uniforms {
            exposure,
            // Texture dimensions feed the WGSL's contained-fit math so the
            // shader mirrors `widget::image.content_fit(ContentFit::Contain)`.
            tex_w,
            tex_h,
            sc_x: bounds.x * sf,
            sc_y: bounds.y * sf,
            sc_w: bounds.width * sf,
            sc_h: bounds.height * sf,
            // Detail-view transform: zoom in log2 units (1.0 = contain fit);
            // pan in logical points converted to physical pixels, same
            // convention as the scissor rect above.
            zoom: self.zoom,
            pan_x: self.pan.x * sf,
            pan_y: self.pan.y * sf,
            crop_l,
            crop_t,
            crop_w,
            crop_h,
            rot,
            // Whether to draw the full-frame dim overlay (see set_show_mask).
            show_mask: if self.show_mask { 1.0 } else { 0.0 },
            // Crop-mode minimum padding, logical points → physical pixels like
            // the pan (see set_pad).
            pad: self.pad * sf,
            // The density develop block (mirrors `film::Develop`).
            dev_base: self.develop.base,
            dev_black: self.develop.black,
            dev_white: self.develop.white,
            dev_contrast: self.develop.contrast,
            dev_pivot: self.develop.pivot_offset,
        };
        queue.write_buffer(&pipeline.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut cosmic::iced::wgpu::CommandEncoder,
        target: &cosmic::iced::wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let (_tex_view, Some(pipeline_obj), Some(bind_group)) = (
            &pipeline.texture,
            &pipeline.render_pipeline,
            &pipeline.bind_group,
        ) else {
            return;
        };

        {
            let mut pass = encoder.begin_render_pass(&cosmic::iced::wgpu::RenderPassDescriptor {
                label: Some("exposure render"),
                color_attachments: &[Some(cosmic::iced::wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: cosmic::iced::wgpu::Operations {
                        load: cosmic::iced::wgpu::LoadOp::Load,
                        store: cosmic::iced::wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            #[allow(clippy::cast_precision_loss)]
            pass.set_viewport(
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
                0.0,
                1.0,
            );
            pass.set_pipeline(pipeline_obj);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU pipeline (created once per image, stored in iced's primitive storage)
// ---------------------------------------------------------------------------

/// GPU resources shared across all `DetailPrimitive` instances of the same
/// image.  Created lazily on the first `prepare()` call (needs the mono data
/// to build the texture).
pub struct DetailPipeline {
    texture: Option<cosmic::iced::wgpu::TextureView>,
    uniform_buf: cosmic::iced::wgpu::Buffer,
    sampler: cosmic::iced::wgpu::Sampler,
    bind_group_layout: cosmic::iced::wgpu::BindGroupLayout,
    bind_group: Option<cosmic::iced::wgpu::BindGroup>,
    render_pipeline: Option<cosmic::iced::wgpu::RenderPipeline>,
    initialized: bool,
    /// Identity of the image currently installed in the GPU texture. Compared
    /// against the primitive's `image_id` on every `prepare()` call to detect
    /// that the user has selected a different photo.
    current_image_id: Option<u64>,
}

impl std::fmt::Debug for DetailPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetailPipeline")
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl Pipeline for DetailPipeline {
    fn new(
        device: &cosmic::iced::wgpu::Device,
        _queue: &cosmic::iced::wgpu::Queue,
        _format: cosmic::iced::wgpu::TextureFormat,
    ) -> Self {
        let shader = load_shader(device);
        let bind_group_layout = build_bind_group_layout(device);
        let render_pipeline = build_render_pipeline(device, &shader, &bind_group_layout);

        let uniform_buf =
            device.create_buffer_init(&cosmic::iced::wgpu::util::BufferInitDescriptor {
                label: Some("exposure uniforms"),
                contents: bytemuck::bytes_of(&Uniforms::default()),
                usage: cosmic::iced::wgpu::BufferUsages::UNIFORM
                    | cosmic::iced::wgpu::BufferUsages::COPY_DST,
            });

        let sampler = device.create_sampler(&cosmic::iced::wgpu::SamplerDescriptor {
            address_mode_u: cosmic::iced::wgpu::AddressMode::ClampToEdge,
            address_mode_v: cosmic::iced::wgpu::AddressMode::ClampToEdge,
            mag_filter: cosmic::iced::wgpu::FilterMode::Linear,
            min_filter: cosmic::iced::wgpu::FilterMode::Linear,
            // Trilinear minification: the detail texture carries a full mip
            // chain (see [`build_mip_chain`]), so zooming back out past the
            // native level-up minifies through pre-filtered levels instead of
            // aliasing the full-res grain into a moiré pattern.
            mipmap_filter: cosmic::iced::wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Bind group and texture are created lazily in `prepare()` on the
        // first frame (they require the mono data to build the texture).

        Self {
            texture: None,
            uniform_buf,
            sampler,
            bind_group_layout,
            bind_group: None,
            render_pipeline: Some(render_pipeline),
            initialized: false,
            current_image_id: None,
        }
    }

    fn trim(&mut self) {
        // Nothing to trim — the texture lives for the image's lifetime.
    }
}

/// Load the WGSL source from disk and create a shader module.
fn load_shader(device: &cosmic::iced::wgpu::Device) -> cosmic::iced::wgpu::ShaderModule {
    device.create_shader_module(cosmic::iced::wgpu::ShaderModuleDescriptor {
        label: Some("exposure shader"),
        source: cosmic::iced::wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "shader/exposure.wgsl"
        ))),
    })
}

fn build_bind_group_layout(
    device: &cosmic::iced::wgpu::Device,
) -> cosmic::iced::wgpu::BindGroupLayout {
    device.create_bind_group_layout(&cosmic::iced::wgpu::BindGroupLayoutDescriptor {
        label: Some("exposure bgl"),
        entries: &[
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Texture {
                    sample_type: cosmic::iced::wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: cosmic::iced::wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Sampler(
                    cosmic::iced::wgpu::SamplerBindingType::Filtering,
                ),
                count: None,
            },
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: cosmic::iced::wgpu::ShaderStages::VERTEX
                    | cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Buffer {
                    ty: cosmic::iced::wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn build_render_pipeline(
    device: &cosmic::iced::wgpu::Device,
    shader: &cosmic::iced::wgpu::ShaderModule,
    bind_group_layout: &cosmic::iced::wgpu::BindGroupLayout,
) -> cosmic::iced::wgpu::RenderPipeline {
    let pipeline_layout =
        device.create_pipeline_layout(&cosmic::iced::wgpu::PipelineLayoutDescriptor {
            label: Some("exposure pl"),
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

    device.create_render_pipeline(&cosmic::iced::wgpu::RenderPipelineDescriptor {
        label: Some("exposure rp"),
        layout: Some(&pipeline_layout),
        vertex: cosmic::iced::wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: cosmic::iced::wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(cosmic::iced::wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(cosmic::iced::wgpu::ColorTargetState {
                format: cosmic::iced::wgpu::TextureFormat::Bgra8Unorm,
                // Alpha-blend so the WGSL's `alpha=0` letterbox bars let the
                // COSMIC panel background show through. Image pixels have
                // `alpha=1`, so they composite the same as `REPLACE` would.
                blend: Some(cosmic::iced::wgpu::BlendState::ALPHA_BLENDING),
                write_mask: cosmic::iced::wgpu::ColorWrites::ALL,
            })],
            compilation_options: cosmic::iced::wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: cosmic::iced::wgpu::PrimitiveState {
            topology: cosmic::iced::wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: cosmic::iced::wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Uniform buffer layout (must match the WGSL `Uniforms` struct)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Uniforms {
    exposure: f32,
    /// Texture width in pixels — drives the WGSL contained-fit math.
    tex_w: f32,
    /// Texture height in pixels.
    tex_h: f32,
    sc_x: f32,
    sc_y: f32,
    sc_w: f32,
    sc_h: f32,
    /// Detail-view zoom: 1.0 = contain fit, each +1 doubles scale.
    zoom: f32,
    /// Pan offset in physical pixels (logical points × scale factor).
    pan_x: f32,
    pan_y: f32,
    /// Live crop, in source pixels: the sub-rectangle origin (left/top margins
    /// removed) and its size (source − removed margins). All zero/no-op when
    /// `crop_w == tex_w` and `crop_l == 0`. These are the EXIF-upright texture
    /// coordinates; the display rotation below composes after them.
    crop_l: f32,
    crop_t: f32,
    crop_w: f32,
    crop_h: f32,
    /// User display rotation as counter-clockwise 90° quarter-turns (`0`…`3`).
    /// `0.0` is the identity: the EXIF-upright frame, byte-identical render.
    rot: f32,
    /// Non-zero when the "view dimmed crop area" overlay is shown: the shader
    /// draws the full uncropped frame on top of the zoomed crop and dims
    /// everything outside the crop rectangle.
    show_mask: f32,
    /// Minimum padding (physical pixels) kept around the image in crop mode.
    /// The WGSL contain-fit base shrinks by `pad` on every side so a white
    /// margin is always visible; zooming in grows the image into it.
    pad: f32,
    /// Clear-film transmission anchor: `density = log10(dev_base / value)`.
    dev_base: f32,
    /// Density mapped to output black.
    dev_black: f32,
    /// Density mapped to output white.
    dev_white: f32,
    /// Contrast power about the pivot (1.0 = identity).
    dev_contrast: f32,
    /// Signed offset from the `[dev_black, dev_white]` midpoint the contrast
    /// bends around.
    dev_pivot: f32,
}

// SAFETY: Uniforms is repr(C) with all f32 fields.
unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

// ---------------------------------------------------------------------------
// f32 → f16 (IEEE 754 binary16) bit conversion
// ---------------------------------------------------------------------------
//
// `R16Float` is filterable on WebGPU; `R32Float` is not. Half-float has
// 1 sign + 5 exponent + 10 mantissa — plenty of precision for our
// post-inversion linear-light tonal data (mostly 0..1.5, occasionally
// higher after a stop of exposure). The conversion runs once per decode
// (~16 MB f32 → 8 MB u16 at 2048²).
//
// Subnormal f32 inputs flush to zero and overflows flush to infinity;
// round-to-nearest-even is used for mantissa truncation. No half-float
// subnormals are produced (input range is large enough that the small
// loss of precision below ~6e-8 linear is inaudible).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_field = ((bits >> 23) & 0xff).cast_signed();
    let mant = bits & 0x007f_ffff;

    if exp_field == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exp_field == 0 {
        return sign;
    }

    let unbiased = exp_field - 127;

    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        return sign;
    }

    let half_exp = (unbiased + 15) as u32;
    let mut half_mant = mant >> 13;
    let round_part = mant & 0x1fff;

    // Round-to-nearest-even.
    if round_part > 0x1000 || (round_part == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant >= 0x400 {
            // Mantissa overflow → bump exponent (mantissa bits implicitly 0).
            let new_exp = half_exp + 1;
            if new_exp >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((new_exp as u16) << 10);
        }
    }

    sign | ((half_exp as u16) << 10) | (half_mant as u16)
}

/// Decode a half-float bit pattern to f32 — the inverse of [`f32_to_half`].
/// Used by the parity tests to simulate the GPU's half-float LUT texture.
#[cfg(test)]
#[must_use]
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        // Zero or subnormal → flush to zero (matches f32_to_half's subnormal
        // handling: it never emits half subnormals).
        return f32::from_bits(sign << 31);
    }
    if exp == 0x1f {
        return if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7fc0_0000)
        };
    }
    // Re-bias the exponent: half bias 15 → f32 bias 127.
    let f32_bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Convert a `Vec<f32>` of linear mono values into an `R16Float`-compatible
/// byte buffer for `queue.write_texture`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mono_to_half_bytes(mono: &[f32]) -> Vec<u8> {
    let mut half = Vec::with_capacity(mono.len() * 2);
    for &v in mono {
        half.extend_from_slice(&f32_to_half(v).to_le_bytes());
    }
    half
}

/// One level of the detail texture's mip chain, ready for `write_texture`.
struct MipLevel {
    /// Mip width (`max(1, floor(parent/2))`, or the texture width at level 0).
    width: u32,
    /// Mip height, halving like the width.
    height: u32,
    /// Tightly packed `R16Float` (half) bytes: `width × height × 2`.
    data: Vec<u8>,
}

/// Build the full mip chain for a linear mono texture: level 0 is the raw
/// half-float upload, each level below is a 2×2 area-average (box) downsample
/// of the previous in LINEAR light, halving to `max(1, floor(dim/2))` and
/// stopping at 1×1. Edge blocks average only their real contributing texels.
///
/// The averaging happens on the TRUE sensor data before any tone mapping, so
/// the WGSL's per-fragment curve/inversion — which samples the minified texels
/// and THEN applies the tone — sees correctly pre-filtered values (the same
/// flatten-before-average ordering the CPU thumbnail bake uses). Without this
/// chain the single-level texture aliases the high-frequency grain into a
/// moiré whenever the view minifies (notably after the native level-up installs
/// an up-to-8192 texture and the user zooms back out to contain fit).
fn build_mip_chain(mono: &[f32], width: u32, height: u32) -> Vec<MipLevel> {
    let mut levels = Vec::new();
    levels.push(MipLevel {
        width,
        height,
        data: mono_to_half_bytes(mono),
    });

    let mut src = mono.to_vec();
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let (next, nw, nh) = downsample_half(&src, w, h);
        levels.push(MipLevel {
            width: nw,
            height: nh,
            data: mono_to_half_bytes(&next),
        });
        src = next;
        w = nw;
        h = nh;
    }
    levels
}

/// 2×2 area-average downsample of a mono buffer: output pixel `(x, y)` is the
/// mean of the source's `2x`..`2x+2` × `2y`..`2y+2` block, clamped to the
/// source bounds so odd-dimension edges average their actual texels.
#[allow(clippy::cast_precision_loss)]
fn downsample_half(mono: &[f32], width: u32, height: u32) -> (Vec<f32>, u32, u32) {
    let out_w = u32::max(width / 2, 1);
    let out_h = u32::max(height / 2, 1);
    let mut out = vec![0.0_f32; out_w as usize * out_h as usize];
    for y in 0..out_h {
        for x in 0..out_w {
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for dy in 0..2_u32 {
                let sy = y * 2 + dy;
                if sy >= height {
                    continue;
                }
                for dx in 0..2_u32 {
                    let sx = x * 2 + dx;
                    if sx >= width {
                        continue;
                    }
                    sum += mono[sy as usize * width as usize + sx as usize];
                    count += 1;
                }
            }
            out[y as usize * out_w as usize + x as usize] = sum / count as f32;
        }
    }
    (out, out_w, out_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_uv_geometry_zero_crop_is_identity() {
        // Same source and texture dims → the margin mapping is a no-op.
        let (origin, size) = crop_uv_geometry(CropMargins::default(), 400, 300, 400, 300);
        assert_eq!(origin, (0.0, 0.0));
        assert_eq!(size, (400.0, 300.0));
    }

    #[test]
    fn crop_uv_geometry_offsets_origin_by_left_top_and_shrinks_size() {
        // Source == texture (native): margins map 1:1 onto the texture.
        let crop = CropMargins {
            top: 10,
            right: 5,
            bottom: 30,
            left: 15,
        };
        let (origin, size) = crop_uv_geometry(crop, 400, 300, 400, 300);
        assert_eq!(origin, (15.0, 10.0));
        assert_eq!(size, (380.0, 260.0));
    }

    #[test]
    fn crop_uv_geometry_scales_source_px_onto_the_downscaled_texture() {
        // A 6000x4000 source shown at a 0.1× 600x400 texture: a stored
        // source-pixel crop (100 left, 50 top) must remove 10 and 5 texture
        // pixels respectively so the live view and the thumbnail bake stay in
        // the same reference frame. Before the src-aware fix each axis was
        // treated as texture pixels, so the downscaled view truncated ~10× more
        // than the bake revealed.
        let crop = CropMargins {
            top: 50,
            right: 100,
            bottom: 150,
            left: 100,
        };
        let (origin, size) = crop_uv_geometry(crop, 600, 400, 6000, 4000);
        assert_eq!(origin, (10.0, 5.0));
        assert_eq!(size, (580.0, 380.0));
    }

    #[test]
    fn crop_uv_geometry_scales_a_portrait_crop_onto_the_rotated_texture() {
        // Rotate90 display source (4000x6000) downscaled to a 256x384 portrait
        // texture (~0.064×, the THUMB-long-edge ratio is coincidental here).
        let crop = CropMargins {
            top: 300,
            right: 0,
            bottom: 300,
            left: 0,
        };
        let (origin, size) = crop_uv_geometry(crop, 256, 384, 4000, 6000);
        assert_eq!(origin, (0.0, 19.2));
        assert_eq!(size, (256.0, 384.0 - 38.4));
    }

    #[test]
    fn zoom_100_mirrors_the_wgsl_contain_math() {
        // 1:1 (one texture pixel per physical screen pixel) on a 2000×1000
        // texture in a 1000×1000 logical widget at scale factor 1.0: the width
        // axis needs 2× contain, so the cap is 1 + log2(2) = 2.0.
        let program = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            2000);
        assert!((program.zoom_100(1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);

        // An odd display rotation swaps which axis contains: in a 2000×1000
        // widget the cropped height (1000) is the limiting axis at 2× → still
        // 2.0; without rotation the width (2000) exactly fills → cap 1.0.
        let rotated = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            1,
            1,
            2000);
        assert!((rotated.zoom_100(2000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);
        assert!((program.zoom_100(2000.0, 1000.0, 1.0) - 1.0).abs() < 1e-5);

        // Scale factor converts logical → physical: at 2× UI the same texture
        // already fills a 1000-logical widget at 1:1, so the cap is contain.
        assert!((program.zoom_100(1000.0, 1000.0, 2.0) - 1.0).abs() < 1e-5);

        // A crop that halves the frame halves the cap (you magnify the crop).
        let cropped = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins {
                top: 0,
                right: 1000,
                bottom: 0,
                left: 0,
            },
            0,
            1,
            2000);
        assert!((cropped.zoom_100(1000.0, 1000.0, 1.0) - 1.0).abs() < 1e-5);

        // A widget larger than the texture never drops the cap below contain.
        assert!((program.zoom_100(4000.0, 4000.0, 1.0) - 1.0).abs() < 1e-5);

        // Crop-mode padding shrinks the contain-fit base: on a 2000×1000
        // texture in a 1000×1000 widget, 100px padding leaves 800×800 for the
        // image, so the width axis needs 2.5× (2000/800) → cap 1 + log2(2.5).
        let mut padded = program.clone();
        padded.set_pad(100.0);
        assert!((padded.zoom_100(1000.0, 1000.0, 1.0) - (1.0 + 2.5f32.log2())).abs() < 1e-5);

        // A widget smaller than 2× the padding clamps the available box to 1
        // physical pixel on each side (no divide-by-zero / negative cap) —
        // the cap is then just the 1:1 point for a 1px box.
        assert!(
            (padded.zoom_100(50.0, 50.0, 1.0) - (1.0 + 2000.0f32.log2())).abs() < 1e-5
        );

        // Zero padding is the unpadded layout (regression guard for the mirror).
        padded.set_pad(0.0);
        assert!((padded.zoom_100(1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn zoom_100_for_delegates_on_the_own_texture_and_scales_in_log2() {
        // The projection helper with the shader's own texture dims is exactly
        // `zoom_100` (same 2000x1000 texture → cap 2.0 in a 1000x1000 widget).
        let program = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            2000);
        assert!(
            (program.zoom_100_for(2000, 1000, 1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5
        );
        assert!(
            (program.zoom_100_for(2000, 1000, 1000.0, 1000.0, 1.0)
                - program.zoom_100(1000.0, 1000.0, 1.0))
                .abs()
                < 1e-5
        );

        // Doubling both texture axes doubles the physical pixel count at the
        // same scale, so the 1:1 cap grows by exactly one log2 unit — this is
        // what lets the detail view project the native level-up's cap.
        let native = program.zoom_100_for(4000, 2000, 1000.0, 1000.0, 1.0);
        assert!((native - (1.0 + 4.0f32.log2())).abs() < 1e-5);

        // Cropping half the *projected* frame magnifies the projected cap the
        // same way it does the real one (crop is authored in source space).
        let cropped = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins {
                top: 0,
                right: 1000,
                bottom: 0,
                left: 0,
            },
            0,
            1,
            2000);
        assert!((cropped.zoom_100_for(4000, 2000, 1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);

        // The 1:1 projection is floored at contain fit like the real cap.
        assert!((program.zoom_100_for(1000, 500, 4000.0, 4000.0, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn new_restores_display_source_dims_from_the_long_edge() {
        // Landscape source: texture 600x400 (downscale of 6000x4000) → source
        // long edge 6000 lands on the width axis.
        let program = DetailProgram::new(
            vec![0.0; 600 * 400],
            600,
            400,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            6000);
        assert_eq!(program.source_dimensions(), (6000, 4000));

        // Portrait (rotated) source: texture 400x600 → long edge lands height.
        let program = DetailProgram::new(
            vec![0.0; 400 * 600],
            400,
            600,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            6000);
        assert_eq!(program.source_dimensions(), (4000, 6000));

        // Native decode (texture == source): the long edge restores exactly.
        let program = DetailProgram::new(
            vec![0.0; 400 * 600],
            400,
            600,
            crate::film::Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            600);
        assert_eq!(program.source_dimensions(), (400, 600));
    }

    /// Mirrors the WGSL `rotate_uv`/`rotated_dims` display rotation: mapping a
    /// point `(u, v)` over the DISPLAYED (rotated) frame back into the
    /// EXIF-upright texture's `(tu, tv)` normalized coordinates, for a
    /// cumulative counter-clockwise `rot` quarter-turns. The WGSL selection
    /// chain must match this exactly; testing the Rust mirror catches
    /// arithmetic contradictions even though the WGSL itself only parses here.
    fn rot_uv(rot: u8, u: f32, v: f32) -> (f32, f32) {
        match rot & 3 {
            0 => (u, v),
            1 => (1.0 - v, u),
            2 => (1.0 - u, 1.0 - v),
            _ => (v, 1.0 - u),
        }
    }

    #[test]
    fn rotated_dimensions_swap_axes_for_odd_turns() {
        // An odd quarter-turn swaps which frame axis is the display horizontal:
        // the contain-fit (and the thumbnail print) become as tall as the
        // source was wide.
        fn rotated_dims<W: Copy>(w: W, h: W, rot: u8) -> (W, W) {
            match rot & 3 {
                0 | 2 => (w, h),
                _ => (h, w),
            }
        }
        assert_eq!(rotated_dims(3_u32, 2_u32, 0), (3, 2));
        assert_eq!(rotated_dims(3_u32, 2_u32, 1), (2, 3));
        assert_eq!(rotated_dims(3_u32, 2_u32, 2), (3, 2));
        assert_eq!(rotated_dims(3_u32, 2_u32, 3), (2, 3));
    }

    #[test]
    fn rotate_uv_maps_display_quarter_turns_to_texture_uv() {
        // Identity: the displayed (u, v) IS the texture coordinate.
        assert_eq!(rot_uv(0, 0.25, 0.75), (0.25, 0.75));
        // One CCW 90°: the texture's top edge lands on the display's left, so
        // the display's LEFT (u=0) samples the texture's TOP row (tv=0) while
        // moving right across the display walks the texture top→bottom.
        assert_eq!(rot_uv(1, 0.0, 0.0), (1.0, 0.0));
        assert_eq!(rot_uv(1, 0.0, 1.0), (0.0, 0.0));
        assert_eq!(rot_uv(1, 1.0, 1.0), (0.0, 1.0));
        assert_eq!(rot_uv(1, 1.0, 0.0), (1.0, 1.0));
        // Two turns: pure 180° reversal of both axes.
        assert_eq!(rot_uv(2, 0.25, 0.75), (0.75, 0.25));
        // Three turns (one CW): the texture's top edge lands on the display's
        // right; the display's right (u=1) samples the top row (tv=0).
        assert_eq!(rot_uv(3, 1.0, 0.0), (0.0, 0.0));
        assert_eq!(rot_uv(3, 1.0, 1.0), (1.0, 0.0));
        assert_eq!(rot_uv(3, 0.0, 0.0), (0.0, 1.0));
        // Four turns return to the identity, and the count wraps mod 4 the way
        // the shader's `rotation & 3` write keeps it.
        assert_eq!(rot_uv(4, 0.25, 0.75), (0.25, 0.75));
    }

    #[test]
    fn f32_to_half_matches_canonical_values() {
        assert_eq!(f32_to_half(0.0), 0x0000, "+0");
        assert_eq!(f32_to_half(-0.0), 0x8000, "-0");
        assert_eq!(f32_to_half(1.0), 0x3c00, "1.0");
        assert_eq!(f32_to_half(-1.0), 0xbc00, "-1.0");
        assert_eq!(f32_to_half(0.5), 0x3800, "0.5");
        assert_eq!(f32_to_half(2.0), 0x4000, "2.0");
        assert_eq!(f32_to_half(65504.0), 0x7bff, "largest finite f16");
        assert_eq!(f32_to_half(65536.0), 0x7c00, "overflow → +∞");
        assert_eq!(f32_to_half(-65536.0), 0xfc00, "overflow → -∞");
        assert_eq!(f32_to_half(0.1), 0x2e66, "0.1");
        assert_eq!(f32_to_half(-0.1), 0xae66, "-0.1");
        assert_eq!(f32_to_half(0.000_488_281_25), 0x1000, "2^-11");
        assert_eq!(
            f32_to_half(0.000_061_035_156_25),
            0x0400,
            "smallest f16 normal ≈ 2^-14"
        );
        // Values smaller than the smallest f16 normal range flush to zero;
        // we don't produce half-float subnormals.
        assert_eq!(
            f32_to_half(0.000_030_517_578_125),
            0x0000,
            "below smallest normal flushes to 0 (no subnormal handling)"
        );
    }

    #[test]
    fn f32_to_half_underflow_flushes_to_zero() {
        // Anything smaller than the smallest f16 normal (≈ 6.1e-5) flushes
        // to zero; we do not produce half-float subnormals.
        assert_eq!(f32_to_half(1.0e-7), 0x0000, "tiny positive → +0");
        assert_eq!(f32_to_half(-1.0e-7), 0x8000, "tiny negative → -0");
    }

    #[test]
    fn f32_to_half_handles_infinity_and_nan() {
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00, "+∞");
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00, "-∞");
        assert_eq!(f32_to_half(f32::NAN) & 0x7c00, 0x7c00, "NaN keeps exp");
        assert_eq!(f32_to_half(f32::NAN) & 0x0200, 0x0200, "NaN keeps mant bit");
    }

    #[test]
    fn mono_to_half_bytes_lays_out_u16_le() {
        // For 1.0 we expect two bytes: 0x00 0x3C (little-endian u16 0x3C00).
        let bytes = mono_to_half_bytes(&[1.0]);
        assert_eq!(bytes, [0x00, 0x3c]);
    }

    #[test]
    fn build_mip_chain_halves_dimensions_down_to_one() {
        // A 2000×1000 buffer: levels halve per axis (floor) and the chain stops
        // at 1×1. `floor(log2(2000)) + 1 = 11` levels (2000→1000→…→3→1).
        let mono: Vec<f32> = vec![0.0; 2000 * 1000];
        let mips = build_mip_chain(&mono, 2000, 1000);
        assert_eq!(mips.len(), 11);
        assert_eq!((mips[0].width, mips[0].height), (2000, 1000));
        assert_eq!((mips[1].width, mips[1].height), (1000, 500));
        assert_eq!((mips[2].width, mips[2].height), (500, 250));
        assert_eq!((mips[3].width, mips[3].height), (250, 125));
        assert_eq!((mips[4].width, mips[4].height), (125, 62));
        assert_eq!(
            (mips[10].width, mips[10].height),
            (1, 1),
            "last level is 1×1"
        );
        for mip in &mips {
            assert_eq!(
                mip.data.len(),
                mip.width as usize * mip.height as usize * 2,
                "tightly packed half bytes"
            );
        }
    }

    #[test]
    fn build_mip_chain_level0_is_the_raw_half_data() {
        let mono: Vec<f32> = (0..100).map(|v| v as f32 / 99.0).collect();
        let mips = build_mip_chain(&mono, 10, 10);
        assert_eq!(mips[0].width, 10);
        assert_eq!(mips[0].height, 10);
        assert_eq!(mips[0].data, mono_to_half_bytes(&mono));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn build_mip_chain_box_averages_source_blocks_in_linear_light() {
        // A 4×4 ramp: mip 1's (0,0) texel must be the mean of the base's 2×2
        // top-left block, and (1,1) the mean of the 2×2 bottom-right block.
        let mono: Vec<f32> = (0..16).map(|v| v as f32 / 15.0).collect();
        let mips = build_mip_chain(&mono, 4, 4);
        assert_eq!((mips[1].width, mips[1].height), (2, 2));

        let decode = |mip: &MipLevel, x: u32, y: u32| {
            let offset = (y as usize * mip.width as usize + x as usize) * 2;
            half_to_f32(u16::from_le_bytes([mip.data[offset], mip.data[offset + 1]]))
        };
        let mean = |coords: &[(usize, usize)]| {
            let sum: f32 = coords.iter().map(|&(x, y)| mono[y * 4 + x]).sum();
            sum / coords.len() as f32
        };
        assert!((decode(&mips[1], 0, 0) - mean(&[(0, 0), (1, 0), (0, 1), (1, 1)])).abs() < 1e-3);
        assert!((decode(&mips[1], 1, 1) - mean(&[(2, 2), (3, 2), (2, 3), (3, 3)])).abs() < 1e-3);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn build_mip_chain_handles_odd_dimensions() {
        // A 3×2 buffer (row-major: [0,0.1,0.2, 0.3,0.4,0.5]) has no exact 2×2
        // tiling: strict halving drops the odd column/row tail, so the single
        // 1×1 mip 1 averages the 2×2 top-left block — 0.0, 0.1, 0.3, 0.4.
        let mono: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let mips = build_mip_chain(&mono, 3, 2);
        assert_eq!(mips.len(), 2);
        assert_eq!((mips[1].width, mips[1].height), (1, 1));
        let expected = (0.0 + 0.1 + 0.3 + 0.4) / 4.0;
        let decoded = half_to_f32(u16::from_le_bytes([mips[1].data[0], mips[1].data[1]]));
        assert!((decoded - expected).abs() < 1e-3);
    }

    #[test]
    fn build_mip_chain_single_pixel_is_a_single_level() {
        let mips = build_mip_chain(&[0.42], 1, 1);
        assert_eq!(mips.len(), 1);
        assert_eq!((mips[0].width, mips[0].height), (1, 1));
    }

    #[test]
    fn set_develop_updates_the_live_controls() {
        let mut program = DetailProgram::new(
            vec![0.0; 16],
            4,
            4,
            Develop::with_base(crate::film::DEFAULT_FILM_BASE),
            CropMargins::default(),
            0,
            1,
            4,
        );
        program.set_exposure(1.5);
        assert!((program.develop.exposure_ev - 1.5).abs() < 1e-6);
        let mut d = program.develop;
        d.contrast = 2.0;
        program.set_develop(d);
        assert!((program.develop.contrast - 2.0).abs() < 1e-6);
    }

    #[test]
    fn wgsl_source_parses_and_validates() {
        // The shader is compiled by wgpu at first launch, so a parse error
        // would only surface at runtime. Validate the source we embed every
        // test run instead.
        let source = include_str!("shader/exposure.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|err| panic!("WGSL failed to parse: {err}"));

        let stages: Vec<_> = module.entry_points.iter().map(|ep| ep.stage).collect();
        assert!(stages.contains(&naga::ShaderStage::Vertex));
        assert!(stages.contains(&naga::ShaderStage::Fragment));
    }
}
