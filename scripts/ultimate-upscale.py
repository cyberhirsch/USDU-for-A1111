import math
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from modules import processing, shared, images, devices, scripts
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.shared import opts, state
from enum import Enum

elem_id_prefix = "ultimateupscale"

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class USDUpscaler():

    def __init__(self, p, image, upscaler_index:int, save_redraw, save_seams_fix, tile_width, tile_height, save_intermediate=False) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)
        self.save_intermediate = save_intermediate  # New attribute
        self.intermediate_image = None  # To store the intermediate image

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            if self.save_intermediate:
                self.intermediate_image = self.image.copy()
                self.save_image(intermediate=True)
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index+1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        
        # Save or store the intermediate image if required
        if self.save_intermediate:
            self.intermediate_image = self.image.copy()
            self.save_image(intermediate=True)

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self, intermediate=False):
        if intermediate and self.intermediate_image is not None:
            if type(self.p.prompt) != list:
                images.save_image(self.intermediate_image, self.p.outpath_samples, "upscaled_intermediate", self.p.seed, self.p.prompt, opts.samples_format, info=self.initial_info, p=self.p)
            else:
                images.save_image(self.intermediate_image, self.p.outpath_samples, "upscaled_intermediate", self.p.seed, self.p.prompt[0], opts.samples_format, info=self.initial_info, p=self.p)
        else:
            if type(self.p.prompt) != list:
                images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.samples_format, info=self.initial_info, p=self.p)
            else:
                images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt[0], opts.samples_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()
    
        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()
        
        # Optionally, you can also append the intermediate image to result_images for display
        if self.save_intermediate and self.intermediate_image is not None:
            self.result_images.append(self.intermediate_image)

class USDURedraw():
    # Existing class implementation remains unchanged
    # ...

class USDUSeamsFix():
    # Existing class implementation remains unchanged
    # ...

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]

        redrow_modes = [
            "Linear",
            "Chess",
            "None"
        ]

        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image depending on the selected target size type</p>")

        with gr.Row():
            target_size_type = gr.Dropdown(label="Target size type", elem_id=f"{elem_id_prefix}_target_size_type", choices=[k for k in target_size_types], type="index",
                                  value=next(iter(target_size_types)))

            custom_width = gr.Slider(label='Custom width', elem_id=f"{elem_id_prefix}_custom_width", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_height = gr.Slider(label='Custom height', elem_id=f"{elem_id_prefix}_custom_height", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_scale = gr.Slider(label='Scale', elem_id=f"{elem_id_prefix}_custom_scale", minimum=1, maximum=16, step=0.01, value=2, visible=False, interactive=True)

        gr.HTML("<p style=\"margin-bottom:0.75em\">Redraw options:</p>")
        with gr.Row():
            upscaler_index = gr.Radio(label='Upscaler', elem_id=f"{elem_id_prefix}_upscaler_index", choices=[x.name for x in shared.sd_upscalers],
                                value=shared.sd_upscalers[0].name, type="index")
        with gr.Row():
            redraw_mode = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_redraw_mode", choices=[k for k in redrow_modes], type="index", value=next(iter(redrow_modes)))
            tile_width = gr.Slider(elem_id=f"{elem_id_prefix}_tile_width", minimum=0, maximum=2048, step=64, label='Tile width', value=512)
            tile_height = gr.Slider(elem_id=f"{elem_id_prefix}_tile_height", minimum=0, maximum=2048, step=64, label='Tile height', value=0)
            mask_blur = gr.Slider(elem_id=f"{elem_id_prefix}_mask_blur", label='Mask blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(elem_id=f"{elem_id_prefix}_padding", label='Padding', minimum=0, maximum=512, step=1, value=32)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Seams fix:</p>")
        with gr.Row():
            seams_fix_type = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_seams_fix_type", choices=[k for k in seams_fix_types], type="index", value=next(iter(seams_fix_types)))
            seams_fix_denoise = gr.Slider(label='Denoise', elem_id=f"{elem_id_prefix}_seams_fix_denoise", minimum=0, maximum=1, step=0.01, value=0.35, visible=False, interactive=True)
            seams_fix_width = gr.Slider(label='Width', elem_id=f"{elem_id_prefix}_seams_fix_width", minimum=0, maximum=128, step=1, value=64, visible=False, interactive=True)
            seams_fix_mask_blur = gr.Slider(label='Mask blur', elem_id=f"{elem_id_prefix}_seams_fix_mask_blur", minimum=0, maximum=64, step=1, value=4, visible=False, interactive=True)
            seams_fix_padding = gr.Slider(label='Padding', elem_id=f"{elem_id_prefix}_seams_fix_padding", minimum=0, maximum=128, step=1, value=16, visible=False, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Save options:</p>")
        with gr.Row():
            save_upscaled_image = gr.Checkbox(label="Upscaled", elem_id=f"{elem_id_prefix}_save_upscaled_image", value=True)
            save_seams_fix_image = gr.Checkbox(label="Seams fix", elem_id=f"{elem_id_prefix}_save_seams_fix_image", value=False)
            save_intermediate_image = gr.Checkbox(label="Save Intermediate Upscaled Image", elem_id=f"{elem_id_prefix}_save_intermediate_image", value=False)  # New Checkbox

        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2 or fix_index == 3
            width_visible = fix_index == 1

            return [gr.update(visible=all_visible),
                    gr.update(visible=width_visible),
                    gr.update(visible=mask_blur_visible),
                    gr.update(visible=all_visible)]

        seams_fix_type.change(
            fn=select_fix_type,
            inputs=seams_fix_type,
            outputs=[seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding]
        )

        def select_scale_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2

            return [gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_scale),
                    ]

        target_size_type.change(
            fn=select_scale_type,
            inputs=target_size_type,
            outputs=[custom_width, custom_height, custom_scale]
        )

        def init_field(scale_name):
            try:
                scale_index = target_size_types.index(scale_name)
                custom_width.visible = custom_height.visible = scale_index == 1
                custom_scale.visible = scale_index == 2
            except:
                pass

        target_size_type.init_field = init_field

        return [info, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
                seams_fix_type, target_size_type, custom_width, custom_height, custom_scale, save_intermediate_image]  # Added

    def run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale, save_intermediate_image):
    
        # Init
        processing.fix_seed(p)
        devices.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = images.flatten(init_img, opts.img2img_background_color)

        #override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p, 
            init_img, 
            upscaler_index, 
            save_upscaled_image, 
            save_seams_fix_image, 
            tile_width, 
            tile_height,
            save_intermediate=save_intermediate_image  # Pass the new parameter
        )
        upscaler.upscale()
        
        # Optionally, retrieve the intermediate image
        intermediate_image = upscaler.intermediate_image if save_intermediate_image else None

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        # Prepare info
        info_text = upscaler.initial_info if upscaler.initial_info is not None else ""
        
        # Return both intermediate and final images if available
        if intermediate_image:
            result_images.insert(0, intermediate_image)  # Insert at the beginning for easy identification
            info_text += "\nIntermediate upscaled image saved."

        return Processed(p, result_images, seed, info_text if info_text else "")
