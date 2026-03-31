"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GRAVITAS_PRESETS = void 0;
exports.GRAVITAS_PRESETS = {
    'default': {
        label: 'Default (Mixed)',
        description: 'Standard configuration',
        config: {
            'coder.general.mode': 'gpu',
            'coder.hardware.contextSize': 8192,
            'coder.hardware.nGpuLayers': 36,
            'reviewer.general.mode': 'cpu',
            'reviewer.hardware.contextSize': 8192
        }
    },
    'high_accuracy': {
        label: 'High Accuracy',
        description: 'Max context, slow sampler',
        config: {
            'coder.general.mode': 'gpu',
            'coder.hardware.contextSize': 8192,
            'coder.sampling.temperature': 0.2,
            'reviewer.general.mode': 'cpu',
            'reviewer.hardware.contextSize': 8192,
            'reviewer.sampling.temperature': 0.1
        }
    },
    'fast_draft': {
        label: 'Fast Draft',
        description: 'Small context, fast sampler',
        config: {
            'coder.general.mode': 'gpu',
            'coder.hardware.contextSize': 1024,
            'coder.sampling.temperature': 0.8,
            'reviewer.general.mode': 'cpu',
            'reviewer.hardware.contextSize': 1024
        }
    },
    'safe_mode': {
        label: 'Safe Mode',
        description: 'Force CPU for all agents',
        config: {
            'coder.general.mode': 'cpu',
            'coder.hardware.nGpuLayers': 0,
            'reviewer.general.mode': 'cpu',
            'reviewer.hardware.nGpuLayers': 0
        }
    }
};
//# sourceMappingURL=presetDefinitions.js.map