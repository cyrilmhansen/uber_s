#version 450

layout(location = 0) in vec2 fragUv;
layout(location = 0) out vec4 outColor;

layout(std140, binding = 0) uniform UbershaderParams {
    int shaderType;    // 0: Solid, 1: BorderedRect, 2: TexturedQuad, 3: TextGlyph
    vec4 baseColor;
    vec4 borderColor;
    float borderThickness;
    vec3 _padding1; // To match Rust struct padding
} params;

// Texture sampler (if texturing is active)
layout(binding = 1) uniform sampler2D texSampler; // Keep for future, not used if shaderType isn't 2 or 3

void main() {
    vec4 local_color = params.baseColor; // Default to base color

    if (params.shaderType == 0) { // Solid Color Rectangle
        // Already set to baseColor
    } else if (params.shaderType == 1) { // Bordered Rectangle
        float border = params.borderThickness;
        if (fragUv.x < border || fragUv.x > (1.0 - border) ||
            fragUv.y < border || fragUv.y > (1.0 - border)) {
            local_color = params.borderColor;
        } else {
            local_color = params.baseColor;
        }
    } else if (params.shaderType == 2) { // Textured Quad
        local_color = texture(texSampler, fragUv) * params.baseColor; // Modulate texture with baseColor (tint)
    } else if (params.shaderType == 3) { // Text Glyph (bitmap font)
        // Assume glyph is in alpha channel of texture
        float alpha = texture(texSampler, fragUv).a;
        local_color = vec4(params.baseColor.rgb, params.baseColor.a * alpha);
    }
    // else: Could be other types or default to transparent/error color

    outColor = local_color;
}
