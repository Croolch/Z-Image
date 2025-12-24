"""
Z-Image Gradio Web UI
An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer
"""

import torch
import gradio as gr
import random
import time
from PIL import Image
from typing import Optional

# Global pipeline instance
pipe = None


def load_pipeline(model_path: str = "Tongyi-MAI/Z-Image-Turbo"):
    """Load the Z-Image pipeline."""
    global pipe
    
    if pipe is not None:
        return "✅ 模型已加载"
    
    from diffusers import ZImagePipeline
    
    print(f"[Z-Image] Loading model from: {model_path}")
    
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")
    
    print("[Z-Image] Model loaded successfully!")
    return "✅ 模型加载成功！"


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    randomize_seed: bool,
    progress=gr.Progress()
) -> tuple[Image.Image, int, str]:
    """Generate an image using Z-Image."""
    global pipe
    
    if pipe is None:
        load_pipeline()
    
    # Handle seed
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    
    progress(0.1, desc="🎨 准备生成...")
    
    # Create generator
    generator = torch.Generator("cuda").manual_seed(seed)
    
    progress(0.3, desc="⚡ 正在生成图像...")
    
    start_time = time.time()
    
    # Generate image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    elapsed_time = time.time() - start_time
    
    progress(1.0, desc="✅ 完成！")
    
    image = result.images[0]
    info = f"⏱️ 生成用时: {elapsed_time:.2f}秒 | 🎲 Seed: {seed} | 📐 {width}×{height}"
    
    return image, seed, info


# Preset resolutions
RESOLUTIONS = {
    "1:1 方形 (1024×1024)": (1024, 1024),
    "3:4 竖版 (768×1024)": (768, 1024),
    "4:3 横版 (1024×768)": (1024, 768),
    "9:16 手机竖屏 (576×1024)": (576, 1024),
    "16:9 宽屏 (1024×576)": (1024, 576),
    "2:3 竖版 (680×1024)": (680, 1024),
    "3:2 横版 (1024×680)": (1024, 680),
}


def update_resolution(resolution_name: str) -> tuple[int, int]:
    """Update width and height based on preset selection."""
    if resolution_name in RESOLUTIONS:
        return RESOLUTIONS[resolution_name]
    return 1024, 1024


# Example prompts
EXAMPLE_PROMPTS = [
    ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Soft-lit outdoor night background.", "", "1:1 方形 (1024×1024)"],
    ["一位穿着蓝色旗袍的优雅女性，站在江南水乡的石桥上，背景是白墙黛瓦的古建筑，水面倒映着彩色的灯笼，夜幕降临，月光皎洁。", "", "3:4 竖版 (768×1024)"],
    ["A majestic snow leopard resting on a rocky mountain peak at golden hour, dramatic lighting, photorealistic, 8K quality, National Geographic style.", "", "16:9 宽屏 (1024×576)"],
    ["Cyberpunk cityscape at night, neon signs in Chinese and English, flying cars, rain-slicked streets reflecting colorful lights, futuristic architecture.", "", "16:9 宽屏 (1024×576)"],
    ["A cozy coffee shop interior, warm lighting, bookshelves, vintage furniture, steam rising from a ceramic cup, autumn leaves visible through the window.", "", "4:3 横版 (1024×768)"],
    ["「造梦」两个金色大字，霓虹灯效果，赛博朋克风格背景，深蓝色夜空，城市灯火璀璨。", "", "1:1 方形 (1024×1024)"],
]


# Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-hover: #4f46e5;
    --accent-color: #f59e0b;
    --bg-dark: #0f0f1a;
    --bg-card: #1a1a2e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #2d2d44;
    --success-color: #10b981;
}

.gradio-container {
    font-family: 'Noto Sans SC', 'Space Grotesk', sans-serif !important;
    background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a3e 50%, var(--bg-dark) 100%) !important;
    min-height: 100vh;
}

.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
}

.subtitle {
    text-align: center;
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
}

.gr-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
}

.gr-input, .gr-dropdown {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

.gr-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

.image-output {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
}

.info-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    text-align: center !important;
}

.example-row {
    background: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: all 0.2s ease !important;
}

.example-row:hover {
    background: rgba(99, 102, 241, 0.1) !important;
}

footer {
    display: none !important;
}
"""


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(css=custom_css, title="Z-Image | 造相", theme=gr.themes.Base()) as demo:
        
        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px 0;">
                <h1 class="main-title">⚡ Z-Image 造相</h1>
                <p class="subtitle">An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer</p>
            </div>
        """)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                
                # Prompt section
                with gr.Group():
                    gr.Markdown("### 🎨 创作提示")
                    prompt = gr.Textbox(
                        label="提示词 (Prompt)",
                        placeholder="描述你想要生成的图像...\n支持中英文双语，例如：一位穿着汉服的女子...",
                        lines=4,
                        max_lines=8,
                    )
                    negative_prompt = gr.Textbox(
                        label="负面提示词 (Negative Prompt)",
                        placeholder="描述你不想要的内容（可选）",
                        lines=2,
                        max_lines=4,
                    )
                
                # Resolution section
                with gr.Group():
                    gr.Markdown("### 📐 图像设置")
                    resolution = gr.Dropdown(
                        choices=list(RESOLUTIONS.keys()),
                        value="1:1 方形 (1024×1024)",
                        label="预设分辨率",
                    )
                    with gr.Row():
                        width = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="宽度",
                        )
                        height = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="高度",
                        )
                
                # Advanced settings
                with gr.Accordion("⚙️ 高级设置", open=False):
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=9,
                        step=1,
                        label="推理步数 (Turbo模型推荐9步)",
                    )
                    guidance_scale = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.1,
                        label="引导比例 (Turbo模型设为0)",
                    )
                    with gr.Row():
                        seed = gr.Number(
                            value=42,
                            label="随机种子",
                            precision=0,
                        )
                        randomize_seed = gr.Checkbox(
                            label="🎲 随机",
                            value=True,
                        )
                
                # Generate button
                generate_btn = gr.Button(
                    "✨ 生成图像",
                    variant="primary",
                    size="lg",
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="生成结果",
                    type="pil",
                    elem_classes=["image-output"],
                    height=512,
                )
                info_text = gr.Textbox(
                    label="生成信息",
                    interactive=False,
                    elem_classes=["info-box"],
                )
                output_seed = gr.Number(
                    label="使用的种子",
                    interactive=False,
                    visible=False,
                )
        
        # Examples section
        gr.Markdown("### 💡 示例提示词")
        gr.Examples(
            examples=EXAMPLE_PROMPTS,
            inputs=[prompt, negative_prompt, resolution],
            examples_per_page=6,
            elem_id="examples",
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 30px 0; color: #64748b;">
                <p>Powered by <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank" style="color: #6366f1;">Z-Image</a> | 
                Made with ❤️ by Tongyi MAI Team</p>
            </div>
        """)
        
        # Event handlers
        resolution.change(
            fn=update_resolution,
            inputs=[resolution],
            outputs=[width, height],
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                seed,
                randomize_seed,
            ],
            outputs=[output_image, output_seed, info_text],
        )
        
        # Allow Enter key to submit
        prompt.submit(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                seed,
                randomize_seed,
            ],
            outputs=[output_image, output_seed, info_text],
        )
    
    return demo


if __name__ == "__main__":
    # Pre-load the model
    print("[Z-Image] Starting Gradio app...")
    load_pipeline()
    
    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

